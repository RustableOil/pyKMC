"""Module to reconstruct an event from saddle positions"""

from pykmc.manager import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
import copy
from pykmc.utils.geometry import push_towards, compute_delr, per_atom_displacement
import ase.geometry

# TODO: Use it in KMC
# TODO: Clean reconstruct/split the method

# Number of worst atoms listed in a reconstruction error message
N_WORST_ATOMS = 8


def deviation_diagnostics(
    supposed_positions, obtained_positions, cell, neighbors, central_atom, thr
):
    """Per-atom deviation between the expected and the obtained environment.

    Each atom's deviation is paired with its distance from the central atom
    (from the geometric center of the expected environment when the central
    atom is unknown), to tell failures at the core of the event from failures
    at the edge of the rcut environment.

    Parameters
    ----------
    supposed_positions : np.ndarray
        Expected positions of the environment's atoms.
    obtained_positions : np.ndarray
        Minimized positions of the same atoms.
    cell : np.ndarray
        Simulation box cell.
    neighbors : list[int]
        System indices of the environment's atoms.
    central_atom : int | None
        System index of the event's central atom.
    thr : float
        Deviation threshold of the reconstruction check.

    Returns
    -------
    tuple[str, dict]
        A one-line summary for the error message, and the per-atom arrays.
    """
    neighbors = np.asarray(neighbors)
    deviation = per_atom_displacement(supposed_positions, obtained_positions, cell)
    local_center = np.where(neighbors == central_atom)[0]
    if len(local_center) == 1:
        center = supposed_positions[local_center[0]]
    else:
        center = np.mean(supposed_positions, axis=0)
    distance = per_atom_displacement(
        np.tile(center, (len(supposed_positions), 1)), supposed_positions, cell
    )

    worst = np.argsort(deviation)[::-1]
    above = worst[deviation[worst] > thr]
    summary = "{}/{} atoms above {} A; deviation@distance from center (A): {}".format(
        len(above),
        len(deviation),
        thr,
        ", ".join(
            "{:.3f}@{:.2f}".format(deviation[i], distance[i])
            for i in above[:N_WORST_ATOMS]
        ),
    )
    if len(local_center) == 1:
        summary += "; central atom deviation {:.3f}".format(deviation[local_center[0]])
    variables = {
        "atom_indices": neighbors,
        "deviation": deviation,
        "distance_from_center": distance,
    }
    return summary, variables


class Reconstruction:
    def __init__(self, config: Config, manager: Manager, types=None) -> None:
        self.config = config
        self.manager = manager  # Manager objet that can perform minimization and return minimized positions
        self.types = types

    def reconstruct(
        self,
        supposed_min1_positions,
        supposed_min2_positions,
        saddle_positions,
        cell,
        delr_thr,
        neighbors=None,
        central_atom=None,
    ):
        """From a saddle point, try to reconstruct the event to see if it matches the
        supposed min1 and min2 positions, and that the to minima are connected.

        Since we generaly save only the atomic environment of the central atom
        we can specified neighbors which correspond to the list index of atoms in
        saddle positions that we need to modifie to go toward min pos.

        The reconstruction procede as follow :
        From the saddle positions
        Move the system toward the first minimum (with fraction)
        Minimize and compare minimized positions with supposed min1 positions
        same for min2


        Parameters
        ----------
        supposed_min1_positions : _type_
            _description_
        supposed_min2_positions : _type_
            _description_
        saddle_positions : _type_
            _description_
        cell :
        neighbors : _type_, optional
            _description_, by default None
            typically the neighors list of the in the atomic environment of the atom on which we apply the event
        central_atom : int, optional
            System index of the event's central atom, only used to report the
            distance of each mismatched atom from it when the check fails.
        """

        if neighbors is None:  # len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle_positions))

        # Saddle positions
        tmp_positions = copy.deepcopy(saddle_positions)

        # Move toward min1 positions
        saddle_toward_min1_pos = push_towards(
            saddle_positions[neighbors],
            supposed_min1_positions,
            fraction=self.config.reconstruction.push_fraction,
            cell=cell,
        )
        tmp_positions[neighbors] = saddle_toward_min1_pos
        # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
        min1_pos, _ = self.manager.group_minimize_with_results(
            config=self.config, positions=tmp_positions, types=self.types
        )
        #        min1_pos, _ = future.result()

        # compaire min1_pos with system current positions
        t1 = ase.geometry.wrap_positions(positions=min1_pos, cell=cell, pbc=True)
        delr1 = compute_delr(
            supposed_min1_positions, t1[neighbors], cell
        )  # I guess we need to be carefull here, if atom_modify sort 0 it's ok
        if delr1 > self.config.psr.matching_score_thr:
            summary, diagnostics = deviation_diagnostics(
                supposed_min1_positions,
                t1[neighbors],
                cell,
                neighbors,
                central_atom,
                self.config.psr.matching_score_thr,
            )
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                    message="did not retreive initial minimum : delr1 = {} ({})".format(
                        delr1, summary
                    ),
                    variables={"delr1": delr1, **diagnostics},
                )
            )
        else:
            # positions towards min2 :
            saddle_toward_min2_pos = push_towards(
                saddle_positions[neighbors],
                supposed_min2_positions,
                fraction=self.config.reconstruction.push_fraction,
                cell=cell,
            )
            tmp_positions[neighbors] = saddle_toward_min2_pos
            # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
            min2_pos, min2_etot = self.manager.group_minimize_with_results(
                config=self.config, positions=tmp_positions, types=self.types
            )
            #            min2_pos, _ = future.result()

            # Compare min2pos with expected final_positions
            t2 = ase.geometry.wrap_positions(positions=min2_pos, cell=cell, pbc=True)
            # delr2 = compute_delr(supposed_min2_positions, min2_pos[neighbors], cell)
            delr2 = compute_delr(supposed_min2_positions, t2[neighbors], cell)
            if delr2 > self.config.psr.matching_score_thr:
                summary, diagnostics = deviation_diagnostics(
                    supposed_min2_positions,
                    t2[neighbors],
                    cell,
                    neighbors,
                    central_atom,
                    self.config.psr.matching_score_thr,
                )
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message="did not retreive expected final minimum : delr2 = {} ({})".format(
                            delr2, summary
                        ),
                        variables={"delr2": delr2, **diagnostics},
                    )
                )

            else:
                return Ok(
                    ReconstructionOutput(
                        min1_positions=min1_pos,
                        saddle_positions=saddle_positions,
                        min2_positions=min2_pos,
                        min2_etot=min2_etot,
                    )
                )
