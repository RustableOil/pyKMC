from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import (
    System,
    Config,
    NeighborsList,
    AtomicEnvironment,
    ReferenceEventTable,
    PointSetRegistration,
    Reconstruction,
)
from typing import Optional
from ..utils import geometry
from ..rate_constant import RateConstant
import pandas as pd
import copy
import ase.geometry
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Ok, Err, ErrorInfo, ErrorType, BasinOutput, Result
from pykmc.psr.result import PSRError

# Log the size of the basin every this many states during its exploration
LOG_EVERY_N_STATES = 100
# Abandon the basin after this many selected exits failed to be reconstructed
MAX_DROPPED_EXITS = 10
# PSR failures that only invalidate one transition, not the whole basin
PSR_ERRORS = (
    PSRError.NO_MATCH_FOUND,
    PSRError.MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
)

# TODO: StateDate is here to handle state informations, when State Object will be creates, need to remove
# TODO: For the moment Basin uses EnergyThresholdDetector, BasinGenericEventExplorer, FPTASelector, need to deal with possible multiple implementation with builder.
# TODO: Think about parallized exploration
# TODO: Could think of refining transient -> absorbing event when exploring
# TODO : Exit if state 0 leads to all absorbing states because all unknown environments, here FTPA fails but because only have 1 transient state (0), should be a different ERROR.TYPE
# TODO should also check if we apply same event to different central atoms but same saddle position meaning that it s a duplicate event, so remove.


@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList]
    transient: bool = False
    visited: bool = False

    def release_heavy_objects(self) -> None:
        """Release heavy objects"""
        self.neighbors_list = None
        self.environment = None

    def ensure_full_state(self, config: Config) -> None:
        if self.system is not None:
            if self.neighbors_list is None:
                self.neighbors_list = NeighborsList(
                    self.system,
                    config.atomicenvironment.rnei,
                    config.atomicenvironment.rcut,
                )
            if self.environment is None:
                self.environment = AtomicEnvironment(
                    config.atomicenvironment.style,
                    self.neighbors_list.neighbors_list["rnei"],
                    self.neighbors_list.neighbors_list["rcut"],
                    config.atomicenvironment.neighbors_add,
                    coordination_threshold=config.atomicenvironment.coordination_threshold,
                    types=self.system.types,
                    coloring_mode=config.atomicenvironment.atom_coloring_mode,
                )


@dataclass
class MappedEvent:
    """A generic event mapped onto an atom of a basin state."""

    system: System  # copy of the state, positions unchanged
    neighbors: np.ndarray  # rcut environment of the central atom
    initial_positions: np.ndarray  # expected positions of `neighbors`
    saddle_positions: np.ndarray
    final_positions: np.ndarray


class BasinsGenericEvents:
    def __init__(
        self, config: Config, reference_table, known_environments, manager, loggers=None
    ) -> None:
        self.loggers = loggers  # LogKMC, optional
        self.config = config  # Config object with basins parameters
        self.explorer = None  # object to explore a state in the basin
        self.reference_table = reference_table  # Object with reference generic events
        self.manager = manager  # object to do external task (minimize, refine)

        self.connectivity_table = None  # Dataframe of basin connexion state
        self.selected_event = None  # The selected event after basin exploration
        self.current_state = None  # Current state where we're at
        self.states_to_explore = None  # List of state to explore
        self.explored_states = None  # List of state that we already explored
        self.states: dict[int, StateData] = {}  # Dictionnary of StateDate
        self.known_environments = known_environments
        # Refined saddle positions per connectivity row label (several rows, i.e.
        # central atoms, can lead to the same exit state)
        self.absorbing_saddle_positions: dict[int, np.ndarray] = {}
        self.n_dropped_exits = 0  # selected exits that could not be reconstructed
        self.dropped_exit_rate = 0.0  # their total rate
        self.n_psr_skipped = 0  # transitions dropped because of a failed PSR match
        # generic rate dropped for a failed PSR match, per source state
        self.psr_dropped_rate: dict[int, float] = {}
        self.psr_max_rate_fraction = 0.0
        # exits with a negligible share of the exit rate: refined only if selected
        self.deferred_exit_rows: set[int] = set()
        self._rate_constant = None
        # Position of the central atom of the event that triggered the basin, in
        # state 0: the centre of [Basin] max_radius (None: no geometric bound)
        self.center_position: Optional[np.ndarray] = None
        self.max_extent = 0.0  # largest distance from the centre of a moved atom

    def detection(self, params) -> bool:
        """Utility method."""
        return self.detector.detection(**params)

    def execute(self, system, center_atom: Optional[int] = None):
        """
        run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.

        `center_atom` is the central atom of the event that triggered the basin:
        the basin's extent, bounded by [Basin] max_radius, is measured from it.
        """
        # initialize the basin
        self._initialize(system)
        if center_atom is not None:
            self.center_position = np.array(system.positions[center_atom], copy=True)
        # explore the basin
        result = self.construct_connexion_table()
        if not result.is_ok():
            return result
        # reorder states index
        mapping = self.connectivity_table.reorder_states_index()
        self._remap_states(mapping)
        # Refine absorbing states
        result = self.refine_absorbing(system)
        if not result.is_ok():
            return result
        return self.select_exit()

    def select_exit(self) -> "Result[BasinOutput, ErrorInfo]":
        """Select an exit with the FPTA and build the output the KMC loop needs.

        An exit whose generic event no longer maps onto its from state is
        dropped and another exit is selected.
        """
        while True:
            # apply selector algorithm to find t_exit and exit_state
            result = self.selector.select_from_connectivity(self.connectivity_table)
            if not result.is_ok():
                return result
            t_exit = result.ok_value().t_exit
            exit_state = result.ok_value().exit_state

            transition = self.connectivity_table.get_transition_to_state(
                target_state=exit_state, as_tuples=False
            )
            row_idx = transition.index[0]
            from_state, event_idx, central_atom, sym_idx, is_transient = (
                self.connectivity_table.to_tuples(transition)[0]
            )
            if row_idx in self.deferred_exit_rows:
                # refined only now that the selector picked it
                self.deferred_exit_rows.discard(row_idx)
                result = self._submit_exit_refinement(
                    self.connectivity_table.df.loc[row_idx]
                )
                if result.is_ok():
                    result = self._store_exit_refinement(row_idx, result.ok_value())
                if not result.is_ok():
                    result = self._drop_exit(exit_state, result.err_value().message)
                    if not result.is_ok():
                        return result
                    continue
            # Exit states are not built during the exploration: the expected
            # final positions come from the generic event mapped onto the from state
            result = self.map_event(from_state, event_idx, central_atom, sym_idx)
            if not result.is_ok():
                result = self._drop_exit(exit_state, result.err_value().message)
                if not result.is_ok():
                    return result
                continue
            mapped = result.ok_value()
            df = self.connectivity_table.df
            return Ok(
                BasinOutput(
                    initial_system_positions=self.states[from_state].system.positions,
                    central_atom=central_atom,
                    saddle_positions=self.absorbing_saddle_positions[row_idx],
                    final_positions=mapped.final_positions,
                    neighbors=mapped.neighbors,
                    energy_barrier=df.loc[row_idx, "dE_forward"],
                    k_tot=df.loc[df["transient"] == False, "k_forward"].sum(),
                    t_exit=t_exit,
                    exit_state=exit_state,
                    from_state=from_state,
                    num_reference_event=event_idx,
                )
            )

    def reselect_exit(
        self, failed_exit_state: int, reason: str
    ) -> "Result[BasinOutput, ErrorInfo]":
        """Drop an exit the KMC loop could not reconstruct and select another one."""
        result = self._drop_exit(failed_exit_state, reason)
        if not result.is_ok():
            return result
        return self.select_exit()

    def _drop_exit(self, exit_state: int, reason: str) -> "Result[None, ErrorInfo]":
        """Remove an exit from the basin by zeroing the rates leading to it.

        The exit time is then drawn again from the remaining exits; the rate of
        the dropped exits is reported, since it is missing from the escape rate.
        """
        df = self.connectivity_table.df
        exits = df["transient"] == False
        k_exits = (
            df.loc[exits, "k_forward"].astype(float).sum() + self.dropped_exit_rate
        )
        into = df["state_connexion"] == exit_state
        row = df[into].iloc[0]
        k_dropped = df.loc[into, "k_forward"].astype(float).sum()
        df.loc[into, "k_forward"] = 0.0
        self.dropped_exit_rate += k_dropped
        self.n_dropped_exits += 1
        share = self.dropped_exit_rate / k_exits if k_exits > 0 else 1.0
        self._log(
            "\t :=> Basin exit {} (reference event {}, atom {}, dE = {:.4f} eV) "
            "dropped: {}; dropped exits carry {:.1%} of the exit rate".format(
                exit_state,
                row["event_connexion"],
                row["central_atom"],
                float(row["dE_forward"]),
                reason,
                share,
            )
        )
        if (
            self.n_dropped_exits >= MAX_DROPPED_EXITS
            or not (df.loc[exits, "k_forward"].astype(float) > 0).any()
        ):
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_EXIT_FAILED,
                    message="{} basin exits could not be reconstructed".format(
                        self.n_dropped_exits
                    ),
                    variables={
                        "n_dropped_exits": self.n_dropped_exits,
                        "dropped_rate_fraction": share,
                    },
                )
            )
        return Ok(None)

    def _initialize(self, system) -> None:
        """
        Initialize necessary component after entering in basin. We always enter in state == 0.
        """
        self.current_state = 0
        self.states_to_explore = [0]
        self.explored_states = []
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(
            config=self.config, reference_table=self.reference_table
        )
        self.selector = FPTASelector()
        self.n_dropped_exits = 0
        self.dropped_exit_rate = 0.0
        self.n_psr_skipped = 0
        self.psr_dropped_rate = {}
        self.psr_max_rate_fraction = 0.0
        self.deferred_exit_rows = set()
        new_system = System(
            positions=system.positions.copy(),
            types=system.types.copy(),
            cell=system.cell.copy(),
            pbc=system.pbc.copy(),
            index=np.arange(len(system.types)),
        )
        self._add_state(
            state_index=0, system=new_system
        )  # add current state 0 to self.states

    def construct_connexion_table(self):
        """
        explore the basin and construct the connextion table
        """
        # States that lost an outgoing transition because of a failed PSR match
        psr_skipped_from_states = set()

        # Loop over state to explore
        while len(self.states_to_explore) != 0:
            # next state to explore :
            to_explore = self.states_to_explore[0]

            if (
                to_explore not in self.states
            ):  # always true except at the start (to_explore = 0)
                # We need to create the state
                # find a state and an event from which we go to the state that we want to create
                from_state, event_idx, central_atom, sym_idx, is_transient = (
                    self.connectivity_table.get_transition_to_state(
                        target_state=to_explore
                    )
                )

                if not is_transient:
                    # Exits are absorbing and never explored: their state is
                    # only needed if the exit is selected, and the KMC loop then
                    # reconstructs it from the refined saddle (see execute)
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)
                    continue

                # Create new system by applying (reconstruction) the generic event to the from_state
                result = self.system_from_state(
                    from_state, event_idx, central_atom, sym_idx
                )
                if not result.is_ok() and result.err_value().type in PSR_ERRORS:
                    # The generic event does not match this atom's environment:
                    # drop the transition (rows sharing to_explore are symmetric
                    # copies of it)
                    df = self.connectivity_table.df
                    dropped = df["state_connexion"] == to_explore
                    rate = df.loc[dropped, "k_forward"].astype(float).sum()
                    self.connectivity_table.df = df[~dropped]
                    self.states_to_explore.remove(to_explore)
                    self.n_psr_skipped += 1
                    psr_skipped_from_states.add(from_state)
                    self.states[from_state].release_heavy_objects()

                    result = self._check_psr_skip_rate(
                        {from_state: rate}, self.connectivity_table.df
                    )
                    if not result.is_ok():
                        return result
                    continue
                if not result.is_ok():
                    # The state cannot be built from the catalogue event: make
                    # the transition an exit, the KMC loop reconstructs it from
                    # its refined saddle if it is selected
                    self.connectivity_table.change_state_to_absorbing(to_explore)
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)
                    self.states[from_state].release_heavy_objects()
                    self._log(
                        "\t :=> Basin state {} (reference event {}, atom {}) could "
                        "not be built, treated as an exit: {}: {}".format(
                            to_explore,
                            event_idx,
                            central_atom,
                            result.err_value().type.name,
                            result.err_value().message,
                        )
                    )
                    continue
                new_system = result.ok_value()

                # Check if it is a new_system or already in states
                is_new_state = self.is_new_state(new_system)
                if is_new_state != -1:  # It already exists
                    # update table
                    self.connectivity_table.change_state_index(
                        current_index=to_explore, new_index=is_new_state
                    )
                    self.explored_states.append(to_explore)
                    self.states_to_explore.remove(to_explore)

                    # Cleaning
                    self.states[from_state].release_heavy_objects()
                    continue  # Skip the rest

                # add state
                self._add_state(
                    state_index=to_explore, system=new_system, transient=is_transient
                )
                n_states = len(self.states)
                if n_states % LOG_EVERY_N_STATES == 0:
                    self._log(
                        "\t :=> Basin exploration: {} states, {} left to explore".format(
                            n_states, len(self.states_to_explore) - 1
                        )
                    )
                if self.center_position is not None:
                    extent = self._extent(new_system)
                    self.max_extent = max(self.max_extent, extent)
                    max_radius = self.config.basin.max_radius
                    if max_radius > 0 and extent > max_radius:
                        return Err(
                            ErrorInfo(
                                type=ErrorType.BASIN_OUTSIDE_RADIUS,
                                message="Basin state {} has an atom moved {:.2f} A from "
                                "the basin centre, beyond max_radius = {} A; lower "
                                "[Basin] energy_thr if transitions that should be "
                                "simulated explicitly are inside the basin".format(
                                    to_explore, extent, max_radius
                                ),
                                variables={"state": to_explore, "extent": extent},
                            )
                        )
                max_states = self.config.basin.max_states
                if max_states > 0 and n_states >= max_states:
                    return Err(
                        ErrorInfo(
                            type=ErrorType.BASIN_TOO_MANY_STATES,
                            message="Basin reached max_states = {} states with {} left "
                            "to explore; lower [Basin] energy_thr if transitions that "
                            "should be simulated explicitly are inside the basin".format(
                                max_states, len(self.states_to_explore) - 1
                            ),
                            variables={"n_states": n_states},
                        )
                    )

                # ENSURE FULL STATE TO EXPLORE
                self.states[to_explore].ensure_full_state(self.config)
                # Check if unknown atomic environments
                if self.is_states_has_unknown_environments(self.states[to_explore]):
                    # We consider that this state is an absorbing one because we need to search new events (in main KMC loop)
                    # Need to update the connectivity table
                    self.connectivity_table.change_state_to_absorbing(to_explore)
                    self.states[to_explore].transient = False
                    is_transient = False

                if not is_transient:
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)

                    # Cleaning
                    self.states[from_state].release_heavy_objects()
                    self.states[to_explore].release_heavy_objects()

                    continue  # We dont explore/skip the rest

                # Release heavy objet memory
                self.states[from_state].release_heavy_objects()

            # Explore state
            self.current_state = to_explore
            last_state_connectivity = self.get_last_state_index()

            # Ensure full state to explore
            self.states[to_explore].ensure_full_state(self.config)
            self.explorer.explore(
                state=self.states[to_explore],
                state_index=self.current_state,
                start_index=last_state_connectivity,
            )

            # to_explore has been explored :
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            # Merge state connectivity table to basin connectivity table
            self.connectivity_table.merge(self.explorer.connectivity_table)
            # Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()
            # Clean heaby state object :
            self.states[to_explore].release_heavy_objects()

        # A transient state left without outgoing transitions would be counted as
        # absorbing by the selector
        lost_states = psr_skipped_from_states.difference(
            self.connectivity_table.df["state"]
        )
        no_exit = (
            self.n_psr_skipped > 0
            and self.connectivity_table.df["transient"].astype(bool).all()
        )
        if lost_states or no_exit:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_NO_VALID_TRANSITION,
                    message="PSR match failures left the basin without valid transitions",
                    variables={
                        "states": sorted(lost_states),
                        "n_psr_skipped": self.n_psr_skipped,
                    },
                )
            )

        df = self.connectivity_table.get_table()
        self._log(
            "\t :=> Basin explored: {} states, {} exits{}".format(
                len(self.states),
                int((~df["transient"].astype(bool)).sum()),
                ""
                if self.center_position is None
                else ", extent {:.2f} A".format(self.max_extent),
            )
        )
        return Ok(None)

    def _extent(self, system) -> float:
        """Largest distance from the basin centre of an atom of `system` that moved
        by more than [Control] disp_thr from state 0 (0 if none moved)."""
        reference = self.states[0].system.positions
        displacement, _ = ase.geometry.find_mic(
            system.positions - reference, system.cell, pbc=True
        )
        moved = np.linalg.norm(displacement, axis=1) > self.config.control.disp_thr
        if not moved.any():
            return 0.0
        _, distances = ase.geometry.find_mic(
            system.positions[moved] - self.center_position, system.cell, pbc=True
        )
        return float(distances.max())

    def _log(self, message: str) -> None:
        if self.loggers is not None:
            self.loggers.info("log", message)

    def select_event(self):
        """
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self):
        """
        convinient method
        """
        pass

    def get_last_state_index(self):
        if self.current_state == 0:  # connextion table is empty
            new_state_connexion = 1
        else:  # last state connexion +1
            new_state_connexion = int(
                self.connectivity_table.get_table()["state_connexion"].iloc[-1] + 1
            )
        return new_state_connexion

    def update_to_explore(self):
        # Find all state index in the connexion table :
        unique_states = set(self.connectivity_table.get_table()["state"]).union(
            set(self.connectivity_table.get_table()["state_connexion"])
        )
        self.states_to_explore = list(
            unique_states.difference(set(self.explored_states))
        )

    def map_event(
        self, from_state, event_idx, central_atom, sym_idx
    ) -> "Result[MappedEvent, ErrorInfo]":
        """Map a generic event onto `central_atom` of `from_state` (PSR and
        symmetry), without moving any atom."""
        ref_event = self.reference_table.table[
            self.reference_table.table["idx_ref"] == event_idx
        ]  # event where event_idx == idx_ref
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()
        #        ref_event = self.reference_table.table.iloc[event_idx].copy()

        # supposed_initial_positions = ref_event["initial_positions"].copy()
        # supposed_final_positions = ref_event["final_positions"].copy()
        # saddle_positions = ref_event['saddle_positions'].copy()

        supposed_initial_positions = np.array(ref_event["initial_positions"], copy=True)
        supposed_final_positions = np.array(ref_event["final_positions"], copy=True)
        saddle_positions = np.array(ref_event["saddle_positions"], copy=True)

        # Apply the generic event to the current state

        # ENSURE FULL STATE FOR FROM STATE
        self.states[from_state].ensure_full_state(self.config)

        # We start from the from_state
        new_system = System(
            positions=self.states[from_state].system.positions.copy(),
            types=self.states[from_state].system.types,
            cell=self.states[from_state].system.cell,
            pbc=True,
            index=np.arange(len(self.states[from_state].system.types)),
        )
        # new_system = copy.deepcopy(self.states[from_state].system)

        # Apply PSR between event initial position and environment positions of the central_atoms

        # TODO: duplicated in refine_absorbing() below and in Refinement.refine_single().
        # Extract into a shared method once basin.py calls Refinement directly instead
        # of re-implementing this preprocessing.
        # Build the point clouds PointSetRegistration needs: local neighborhood of
        # central_atom in new_system (coords1/typ1) against the reference event's
        # initial positions (coords2/typ2)
        neighbor_list = self.states[from_state].neighbors_list.get_neighbors(
            "rcut", central_atom
        )
        coords1 = new_system.positions[neighbor_list]
        coords2 = ref_event.at["initial_positions"]

        if self.config.atomicenvironment.atom_coloring_mode == "full":
            typ1 = list(np.array(new_system.types)[neighbor_list])
            typ2 = list(ref_event.at["types"])
        else:
            # Grey alloy: all atoms treated identically -> species-blind IRA
            # matching (a single shared dummy label, sized per structure).
            typ1 = ["X"] * len(coords1)
            typ2 = ["X"] * len(coords2)

        # unwrap around the central atom (minimum-image convention, valid for
        # orthorhombic and triclinic cells alike)
        coords1 = geometry.unwrap_relative_to(
            coords1, new_system.positions[central_atom], new_system.cell
        )

        result = PointSetRegistration.create(
            self.config.psr.style, config=self.config.ira
        ).match_and_check(
            len(coords1),
            typ1,
            coords1,
            len(coords2),
            typ2,
            coords2,
            self.config.psr.matching_score_thr,
            candidate1=int(np.where(np.asarray(neighbor_list) == central_atom)[0][0]),
            candidate2=int(ref_event.at["move_atom_idx"]),
        )
        if not result.is_ok():  # PSR Err or matching score not valid
            return result
        else:
            psr_output = result.ok_value()  # get psr results

        # Apply PSR to generic event to move

        # Apply symmetry matrix if sym != 0
        # Symmetries are applied to displacements, as in refinement.py: a
        # displacement does not depend on the origin, and the initial
        # positions are invariant under the symmetry, so they stay unchanged.
        if sym_idx != 0:
            sym_matrix = ref_event["sym_matrix"][sym_idx]
            sym_perm = ref_event["sym_perm"][sym_idx]
            new_displacement_saddle = geometry.transform_positions(
                saddle_positions - supposed_initial_positions, sym_matrix, 0, sym_perm
            )
            new_displacement_final = geometry.transform_positions(
                supposed_final_positions - supposed_initial_positions,
                sym_matrix,
                0,
                sym_perm,
            )
            saddle_positions = supposed_initial_positions + new_displacement_saddle
            supposed_final_positions = (
                supposed_initial_positions + new_displacement_final
            )
        supposed_initial_positions = geometry.transform_positions(
            supposed_initial_positions,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )
        saddle_positions = geometry.transform_positions(
            saddle_positions,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )
        supposed_final_positions = geometry.transform_positions(
            supposed_final_positions,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )

        neighbors = self.states[from_state].neighbors_list.get_neighbors(
            "rcut", central_atom
        )
        return Ok(
            MappedEvent(
                system=new_system,
                neighbors=neighbors,
                initial_positions=supposed_initial_positions,
                saddle_positions=saddle_positions,
                final_positions=supposed_final_positions,
            )
        )

    def system_from_state(self, from_state, event_idx, central_atom, sym_idx):
        """Reconstruct the generic event to generate new state from state"""
        result = self.map_event(from_state, event_idx, central_atom, sym_idx)
        if not result.is_ok():
            return result
        mapped = result.ok_value()
        new_system = mapped.system
        neighbors = mapped.neighbors
        supposed_initial_positions = mapped.initial_positions
        saddle_positions = mapped.saddle_positions
        supposed_final_positions = mapped.final_positions

        # Move system do saddle positions
        if self.config.basin.style == "global":
            new_system.update_positions(supposed_final_positions, atom_idx=neighbors)
            min2_pos, _ = self.manager.group_minimize_with_results(
                config=self.config, positions=new_system.positions.copy()
            )
            new_system.update_positions(min2_pos)

        elif self.config.basin.style == "global/reconstruction":
            new_system.update_positions(saddle_positions, atom_idx=neighbors)

            # Reconstruct the event
            # future = self.manager.minimize_with_results(self.config, positions=new_system.positions)
            # min_pos, _ = future.result()

            result = Reconstruction(self.config, self.manager).reconstruct(
                supposed_initial_positions,
                supposed_final_positions,
                new_system.positions,
                new_system.cell,
                self.config.psr.matching_score_thr,
                neighbors,
                central_atom=central_atom,
            )
            if not result.is_ok():
                return result
            new_system.update_positions(result.ok_value().min2_positions)

        else:
            raise ValueError(f"Unknown {self.config.basin.style} style parameter.")

        return Ok(new_system)

    def refine_absorbing(self, system):
        """Refine the energy barrier and rate of the transient -> absorbing
        transitions; exits with a negligible share of the exit rate are deferred
        and refined only if they are selected."""
        df = self.connectivity_table.df
        exit_rows = [idx for idx in df.index if not bool(df.at[idx, "transient"])]
        self.deferred_exit_rows = {
            idx for idx in exit_rows if self._is_negligible_exit(idx)
        }

        futures_context = {}  # row label -> {"min", "saddle", "neighbors"}
        psr_failed_rows = []  # connectivity rows whose PSR match failed
        for idx in exit_rows:
            if idx in self.deferred_exit_rows:
                continue
            result = self._submit_exit_refinement(df.loc[idx])
            if not result.is_ok():  # PSR Err or matching score not valid
                if result.err_value().type not in PSR_ERRORS:
                    return result
                # Drop this exit transition, keep refining the others
                psr_failed_rows.append(idx)
                continue
            futures_context[idx] = result.ok_value()

        if psr_failed_rows:
            result = self._drop_exit_transitions(psr_failed_rows)
            if not result.is_ok():
                return result

        # modify connectivity table entry future1 hold min energy, future2 holds E_saddle
        for idx, ctx in futures_context.items():
            result = self._store_exit_refinement(idx, ctx)
            if not result.is_ok():
                return result

        # Deferred exits keep their generic barrier, with the same constant prefactor
        df = self.connectivity_table.df
        for idx in self.deferred_exit_rows:
            df.loc[idx, "k_forward"] = self._constant_rate(df.loc[idx, "dE_forward"])
        return Ok(None)

    def _submit_exit_refinement(self, row) -> "Result[dict, ErrorInfo]":
        """Place the generic saddle of exit `row` with PSR and submit its refinement.

        Returns the futures context of the refinement, or the PSR error.
        """
        state = self.states[row["state"]]
        result = self.map_event(
            row["state"], row["event_connexion"], row["central_atom"], row["sym"]
        )
        if not result.is_ok():
            state.release_heavy_objects()
            return result
        mapped = result.ok_value()
        tmp_system = mapped.system
        neighbors = mapped.neighbors

        # get tmp_system energy
        future1 = self.manager.get_total_energy(
            positions=tmp_system.positions.copy()
        )  # Send copy not reference
        if self.config.control.active_volume == True:
            # add a job to manager queue
            future2 = self.manager.partn_refine(
                config=self.config,
                central_atom_idx=row["central_atom"],
                positions=tmp_system.positions.copy(),
                cell=tmp_system.cell,
                types=tmp_system.types.copy(),
                saddle_idx=neighbors.copy(),
                saddle_positions=mapped.saddle_positions.copy(),
            )
        # Move system do saddle positions
        else:
            tmp_system.update_positions(mapped.saddle_positions, atom_idx=neighbors)
            # refine
            future2 = self.manager.partn_refine(
                config=self.config,
                central_atom_idx=row["central_atom"],
                positions=tmp_system.positions.copy(),
                saddle_idx=neighbors.copy(),
            )  # send copy not reference !

        # RELEASE MEMORY :
        state.release_heavy_objects()
        return Ok({"min": future1, "saddle": future2, "neighbors": neighbors})

    def _store_exit_refinement(self, idx, ctx) -> "Result[None, ErrorInfo]":
        """Wait for the refinement of exit row `idx`; store its barrier, rate and saddle."""
        E_min = ctx["min"].result()
        result_sad = ctx["saddle"].result()
        if not result_sad.is_ok():
            return result_sad
        E_sad = result_sad.ok_value().E_saddle
        if self.config.control.active_volume == True:
            dE = E_sad
        else:
            dE = E_sad - E_min

        # also save saddle positions refined
        self.absorbing_saddle_positions[idx] = result_sad.ok_value().saddle_positions[
            ctx["neighbors"]
        ]
        # update connectivity table row
        self.connectivity_table.df.loc[idx, "dE_forward"] = dE
        self.connectivity_table.df.loc[idx, "k_forward"] = self._constant_rate(dE)
        return Ok(None)

    def _constant_rate(self, dE) -> float:
        """Rate of a barrier `dE` inside a basin.

        Basin rates always use the constant prefactor k0, whatever [RateConstant]
        style is set.
        """
        if self._rate_constant is None:
            self._rate_constant = RateConstant.create(
                "constant", config=self.config.rateconstant
            )
        return self._rate_constant.compute_rate(dE)

    def _is_negligible_exit(self, row_idx) -> bool:
        """Whether exit `row_idx` is outside the exits that matter.

        The exits of the row's source state are ranked by rate (generic barrier,
        constant prefactor); those covering `config.basin.exit_refine_thr` of the
        state's exit rate are refined, the others are deferred.
        """
        threshold = self.config.basin.exit_refine_thr
        if threshold >= 1.0:
            return False
        df = self.connectivity_table.df
        exits = df[
            (df["state"] == df.at[row_idx, "state"]) & ~df["transient"].astype(bool)
        ]
        rates = exits["dE_forward"].astype(float).map(self._constant_rate)
        rates = rates.sort_values(ascending=False, kind="stable")
        covered_before = (rates.cumsum() - rates) / rates.sum()
        return bool(covered_before.loc[row_idx] >= threshold)

    def _drop_exit_transitions(self, rows) -> "Result[None, ErrorInfo]":
        """Remove transient -> absorbing rows whose PSR match failed.

        Absorbing states left without incoming transitions are removed and state
        indices are made compact again, since the selector indexes its rate matrix
        by state index. The labels of the remaining rows are kept.
        """
        df = self.connectivity_table.df
        transient_states = set(df["state"])
        dropped = (
            df.loc[rows]
            .astype({"k_forward": float})
            .groupby("state")["k_forward"]
            .sum()
        )
        df = df.drop(index=rows)
        self.n_psr_skipped += len(rows)

        result = self._check_psr_skip_rate(dropped.to_dict(), df)
        if not result.is_ok():
            return result

        lost_states = transient_states.difference(df["state"])
        if lost_states or df["transient"].astype(bool).all():
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_NO_VALID_TRANSITION,
                    message="PSR match failures left the basin without valid exit transitions",
                    variables={
                        "states": sorted(lost_states),
                        "n_psr_skipped": self.n_psr_skipped,
                    },
                )
            )

        self.connectivity_table.df = df
        mapping = self.connectivity_table.reorder_states_index()
        self._remap_states(mapping)
        return Ok(None)

    def _check_psr_skip_rate(self, dropped, df) -> "Result[None, ErrorInfo]":
        """Bound the outgoing rate carried by transitions dropped for a failed PSR match.

        `dropped` maps source states to the generic rate just dropped and `df`
        is the connectivity table without those rows. For each state, the rate
        dropped so far is compared with its total outgoing rate (remaining +
        dropped); above `config.basin.psr_skip_rate_thr` the basin fails so the
        original KMC event is applied, as for any other basin failure.
        """
        threshold = self.config.basin.psr_skip_rate_thr
        for state, rate in dropped.items():
            self.psr_dropped_rate[state] = self.psr_dropped_rate.get(state, 0.0) + rate
            kept = df.loc[df["state"] == state, "k_forward"].astype(float).sum()
            total = kept + self.psr_dropped_rate[state]
            fraction = self.psr_dropped_rate[state] / total if total > 0 else 1.0
            if not np.isfinite(fraction):
                fraction = 1.0
            self.psr_max_rate_fraction = max(self.psr_max_rate_fraction, fraction)
            if fraction > threshold:
                return Err(
                    ErrorInfo(
                        type=ErrorType.BASIN_PSR_SKIP_RATE_ABOVE_THRESHOLD,
                        message="Transitions dropped for a failed PSR match carry too "
                        "much of a basin state's outgoing rate",
                        details="state {}: dropped rate fraction = {:.3g}, threshold = {}".format(
                            state, fraction, threshold
                        ),
                        variables={
                            "state": state,
                            "rate_fraction": fraction,
                            "threshold": threshold,
                            "n_psr_skipped": self.n_psr_skipped,
                        },
                    )
                )
        return Ok(None)

    def _remap_states(self, mapping) -> None:
        """Renumber states and their dropped rates after reorder_states_index."""
        self.states = {
            mapping[old]: val for old, val in self.states.items() if old in mapping
        }
        self.psr_dropped_rate = {
            mapping[old]: rate
            for old, rate in self.psr_dropped_rate.items()
            if old in mapping
        }

    def is_new_state(self, system):
        # Loop over all other system in self.states to see if system is already known

        for state_index, state_data in self.states.items():
            are_equivalent = self.are_structures_equivalent(
                system.positions,
                system.types,
                state_data.system.positions,
                state_data.system.types,
                cell=system.cell,
            )
            if are_equivalent:
                return state_index
        return -1

    def are_structures_equivalent(self, pos1, typ1, pos2, typ2, cell, tol=0.3):

        if len(pos1) != len(pos2):
            return False

        # In full coloring mode, two states with the same geometry but a different
        # species arrangement (e.g. an Fe/Ni swap) are distinct; in grey mode they merge.
        if (
            self.config.atomicenvironment.atom_coloring_mode == "full"
            and not np.array_equal(typ1, typ2)
        ):
            return False

        box = np.diag(cell).tolist()
        tree2 = cKDTree(pos2, boxsize=box)
        distances, _ = tree2.query(pos1, k=1)

        return np.max(distances) < tol

    def is_states_has_unknown_environments(self, state: StateData):
        if (
            set(state.environment.atomic_environment_list).difference(
                self.known_environments
            )
            != set()
        ):
            return True
        else:
            return False

    def _add_state(
        self,
        state_index,
        system=None,
        transient=True,
        applicable_events=None,
        visited=False,
        full=False,
    ):
        """Add a new state in the `self.states` dictionnary."""
        # to fit typing
        neighbors_list = []
        atomic_environment = []

        if full == True:
            neighbors_list = NeighborsList(
                system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
            atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                neighbors_list.neighbors_list["rnei"],
                neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
                coordination_threshold=self.config.atomicenvironment.coordination_threshold,
                types=system.types,
                coloring_mode=self.config.atomicenvironment.atom_coloring_mode,
            )
        else:
            neighbors_list = None
            atomic_environment = None
        new_state = StateData(
            system=system,
            environment=atomic_environment,
            neighbors_list=neighbors_list,
            transient=transient,
            visited=visited,
        )

        self.states[state_index] = new_state
