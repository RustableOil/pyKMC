"""A failed PSR match drops one basin transition instead of aborting the basin."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.basins import basin as basin_module
from pykmc.result import Err, ErrorInfo, ErrorType, Ok, PSROutput

PSR_ERR = Err(
    ErrorInfo(
        type=ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
        message="PSR found a match but matching score is above acceptance threshold",
    )
)


def row(state, state_connexion, event, central_atom, transient):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": event,
        "central_atom": central_atom,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.1,
        "k_forward": 1.0,
        "dE_backward": 0.1,
        "k_backward": 1.0,
    }


def make_basin(config=None, manager=None):
    return BasinsGenericEvents(
        config=config or Mock(),
        reference_table=Mock(),
        known_environments=set(),
        manager=manager or Mock(),
    )


@pytest.fixture
def no_heavy_objects(monkeypatch):
    """StateData without neighbors list / environment rebuilds."""
    monkeypatch.setattr(StateData, "ensure_full_state", lambda self, config: None)
    monkeypatch.setattr(StateData, "release_heavy_objects", lambda self: None)


class FakeExplorer:
    """Give state 0 two absorbing exits (events 10 and 11)."""

    def __init__(self):
        self.connectivity_table = BasinStatesConnectivity()

    def explore(self, state, state_index, start_index):
        self.connectivity_table.df = pd.DataFrame(
            [
                row(state_index, start_index, 10, 1, False),
                row(state_index, start_index + 1, 11, 2, False),
            ]
        )

    def clear(self):
        self.connectivity_table.clear()


def explore_from_state_0(basin, system_from_state):
    basin.current_state = 0
    basin.states_to_explore = [0]
    basin.explored_states = []
    basin.n_psr_skipped = 0
    basin.connectivity_table = BasinStatesConnectivity()
    basin.explorer = FakeExplorer()
    basin.states = {0: StateData(system=Mock(), environment=None, neighbors_list=None)}
    basin.system_from_state = system_from_state
    basin.is_new_state = lambda system: -1
    basin.is_states_has_unknown_environments = lambda state: False
    return basin.construct_connexion_table()


class TestConstructConnexionTable:
    def test_psr_failure_drops_only_that_transition(self, no_heavy_objects):
        basin = make_basin()
        result = explore_from_state_0(
            basin,
            lambda from_state, event_idx, central_atom, sym_idx: (
                PSR_ERR if event_idx == 10 else Ok(Mock())
            ),
        )

        assert result.is_ok()
        assert basin.n_psr_skipped == 1
        assert list(basin.connectivity_table.df["event_connexion"]) == [11]
        assert 1 not in basin.states and 2 in basin.states

    def test_all_transitions_failing_is_an_error(self, no_heavy_objects):
        basin = make_basin()
        result = explore_from_state_0(basin, lambda *args: PSR_ERR)

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.BASIN_NO_VALID_TRANSITION
        assert basin.n_psr_skipped == 2

    def test_non_psr_error_still_aborts(self, no_heavy_objects):
        reconstruction_err = Err(
            ErrorInfo(type=ErrorType.RECONSTRUCTION_INVALID_MIN2, message="min2")
        )
        basin = make_basin()
        result = explore_from_state_0(basin, lambda *args: reconstruction_err)

        assert result is reconstruction_err
        assert basin.n_psr_skipped == 0


class TestDropExitTransitions:
    def test_absorbing_states_are_reindexed(self):
        basin = make_basin()
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(
            [
                row(0, 1, 1, 0, True),
                row(1, 0, 2, 0, True),
                row(0, 2, 3, 0, False),
                row(1, 3, 4, 0, False),
                row(0, 4, 5, 0, False),
            ]
        )
        basin.states = {i: f"state{i}" for i in range(5)}

        result = basin._drop_exit_transitions([3])  # 1 -> 3

        df = basin.connectivity_table.df
        assert result.is_ok()
        assert list(df.index) == [0, 1, 2, 4]  # labels kept for futures_context
        assert sorted(set(df["state"]) | set(df["state_connexion"])) == [0, 1, 2, 3]
        assert basin.states == {0: "state0", 1: "state1", 2: "state2", 3: "state4"}
        assert df.loc[4, "state_connexion"] == 3

    @pytest.mark.parametrize(
        "rows, drop",
        [
            # state 1 loses its only transition
            (
                [
                    row(0, 1, 1, 0, True),
                    row(0, 2, 2, 0, False),
                    row(1, 3, 3, 0, False),
                ],
                [2],
            ),
            # no exit left
            (
                [
                    row(0, 1, 1, 0, True),
                    row(1, 0, 2, 0, True),
                    row(0, 2, 3, 0, False),
                ],
                [2],
            ),
        ],
    )
    def test_invalid_basin_is_an_error(self, rows, drop):
        basin = make_basin()
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(rows)
        before = basin.connectivity_table.df.copy()

        result = basin._drop_exit_transitions(drop)

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.BASIN_NO_VALID_TRANSITION
        pd.testing.assert_frame_equal(basin.connectivity_table.df, before)


class TestRefineAbsorbing:
    def test_psr_failure_skips_exit_and_refines_others(
        self, monkeypatch, no_heavy_objects
    ):
        class FakePSR:
            def __init__(self, config, system, ref_event, neighbors_list, atom):
                self.atom = atom

            def match(self):
                if self.atom == 99:
                    return PSR_ERR
                return Ok(
                    PSROutput(
                        rotation_matrix=np.eye(3),
                        translation_matrix=np.zeros(3),
                        permutation_matrix=np.arange(2),
                        matching_score=0.0,
                    )
                )

        monkeypatch.setattr(basin_module, "PointSetRegistration", FakePSR)

        config = SimpleNamespace(
            control=SimpleNamespace(active_volume=False),
            psr=SimpleNamespace(matching_score_thr=0.1),
            rateconstant=SimpleNamespace(k0=10.0, T=300.0),
        )
        saddle = np.full((2, 3), 0.5)
        manager = Mock()
        manager.get_total_energy.return_value = Mock(result=Mock(return_value=-1.0))
        manager.partn_refine.return_value = Mock(
            result=Mock(
                return_value=Ok(
                    SimpleNamespace(E_saddle=-0.7, saddle_positions=np.zeros((2, 3)))
                )
            )
        )
        basin = make_basin(config=config, manager=manager)
        basin.reference_table.table = pd.DataFrame(
            {"idx_ref": [5], "saddle_positions": [saddle], "sym_matrix": [None]}
        )
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(
            [
                row(0, 1, 5, 0, True),
                row(1, 0, 5, 0, True),
                row(0, 2, 5, 99, False),  # PSR fails
                row(1, 3, 5, 1, False),
            ]
        )
        neighbors_list = Mock(get_neighbors=Mock(return_value=np.array([0, 1])))
        basin.states = {
            i: StateData(
                system=System(
                    types=["Si", "Si"],
                    positions=np.zeros((2, 3)),
                    cell=np.eye(3) * 10,
                    pbc=True,
                    index=np.arange(2),
                ),
                environment=None,
                neighbors_list=neighbors_list,
            )
            for i in range(4)
        }

        result = basin.refine_absorbing(system=None)

        df = basin.connectivity_table.df
        assert result.is_ok()
        assert basin.n_psr_skipped == 1
        assert manager.partn_refine.call_count == 1
        assert list(df["central_atom"]) == [0, 0, 1]
        assert df.loc[3, "state_connexion"] == 2  # old state 3 reindexed
        assert df.loc[3, "dE_forward"] == pytest.approx(0.3)
        assert set(basin.absorbing_saddle_positions) == {3}  # row label


class TestExecuteExitTransition:
    def test_output_comes_from_one_transition(self, no_heavy_objects):
        """Two central atoms lead to the same exit state: saddle, neighbors and
        barrier must all belong to the selected transition."""
        basin = make_basin()
        basin._initialize = lambda system: None
        basin.construct_connexion_table = lambda: Ok(None)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(
            [
                row(0, 1, 1, 0, True),
                row(1, 0, 1, 0, True),
                row(0, 2, 2, 7, False),
                row(0, 2, 3, 8, False),
            ]
        )
        basin.connectivity_table.df.loc[2, "dE_forward"] = 0.5
        basin.connectivity_table.df.loc[3, "dE_forward"] = 0.9
        neighbors = {7: np.arange(3), 8: np.arange(4)}
        neighbors_list = Mock(get_neighbors=lambda cutoff, atom: neighbors[atom])
        basin.states = {
            i: StateData(
                system=SimpleNamespace(positions=np.zeros((5, 3))),
                environment=None,
                neighbors_list=neighbors_list,
            )
            for i in range(3)
        }

        def refine(system):
            basin.absorbing_saddle_positions = {2: np.ones((3, 3)), 3: np.ones((4, 3))}
            return Ok(None)

        basin.refine_absorbing = refine
        basin.selector = Mock(
            select_from_connectivity=Mock(
                return_value=Ok(SimpleNamespace(t_exit=1.0, exit_state=2))
            )
        )

        out = basin.execute(system=None).ok_value()

        assert out.central_atom == 7
        assert len(out.neighbors) == len(out.saddle_positions) == 3
        assert out.energy_barrier == 0.5
