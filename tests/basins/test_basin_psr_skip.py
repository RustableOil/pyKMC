"""A failed PSR match drops one basin transition, within a bound on the rate it
carries, instead of aborting the basin."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.basins.basin import MappedEvent
from pykmc.psr.result import PSRError
from pykmc.result import Err, ErrorInfo, ErrorType, Ok

PSR_ERR = Err(
    ErrorInfo(
        type=PSRError.MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
        message="PSR found a match but matching score is above acceptance threshold",
    )
)


def row(state, state_connexion, event, central_atom, transient, k=1.0):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": event,
        "central_atom": central_atom,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.1,
        "k_forward": k,
        "dE_backward": 0.1,
        "k_backward": 1.0,
    }


def basin_config(psr_skip_rate_thr=1.0, exit_refine_thr=1.0, **sections):
    return SimpleNamespace(
        basin=SimpleNamespace(
            psr_skip_rate_thr=psr_skip_rate_thr,
            exit_refine_thr=exit_refine_thr,
            max_states=0,
        ),
        **sections,
    )


def make_basin(config=None, manager=None):
    return BasinsGenericEvents(
        config=config or basin_config(),
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
    """State 0 hops to two transient states, through event 10 (rate k10) and
    event 11 (rate 1); every other state has a single exit."""

    def __init__(self, k10):
        self.k10 = k10
        self.connectivity_table = BasinStatesConnectivity()

    def explore(self, state, state_index, start_index):
        if state_index == 0:
            rows = [
                row(0, start_index, 10, 1, True, k=self.k10),
                row(0, start_index + 1, 11, 2, True),
            ]
        else:
            rows = [row(state_index, start_index, 12, 3, False)]
        self.connectivity_table.df = pd.DataFrame(rows)

    def clear(self):
        self.connectivity_table.clear()


def explore_from_state_0(basin, system_from_state, k10=1.0):
    basin.current_state = 0
    basin.states_to_explore = [0]
    basin.explored_states = []
    basin.connectivity_table = BasinStatesConnectivity()
    basin.explorer = FakeExplorer(k10)
    basin.states = {0: StateData(system=Mock(), environment=None, neighbors_list=None)}
    basin.system_from_state = system_from_state
    basin.is_new_state = lambda system: -1
    basin.is_states_has_unknown_environments = lambda state: False
    return basin.construct_connexion_table()


def fail_event(event, err=PSR_ERR):
    return lambda from_state, event_idx, central_atom, sym_idx: (
        err if event_idx == event else Ok(Mock())
    )


class TestConstructConnexionTable:
    def test_psr_failure_drops_only_that_transition(self, no_heavy_objects):
        basin = make_basin()
        result = explore_from_state_0(basin, fail_event(10))

        assert result.is_ok()
        assert basin.n_psr_skipped == 1
        assert list(basin.connectivity_table.df["event_connexion"]) == [11, 12]
        assert 1 not in basin.states and 2 in basin.states

    def test_all_transitions_failing_is_an_error(self, no_heavy_objects):
        basin = make_basin()
        result = explore_from_state_0(basin, lambda *args: PSR_ERR)

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.BASIN_NO_VALID_TRANSITION
        assert basin.n_psr_skipped == 2

    def test_other_failure_makes_the_transition_an_exit(self, no_heavy_objects):
        reconstruction_err = Err(
            ErrorInfo(type=ErrorType.RECONSTRUCTION_INVALID_MIN2, message="min2")
        )
        basin = make_basin()
        result = explore_from_state_0(basin, fail_event(10, reconstruction_err))

        assert result.is_ok()
        assert basin.n_psr_skipped == 0
        df = basin.connectivity_table.df
        assert not df.loc[df["event_connexion"] == 10, "transient"].any()


class TestSkipRateThreshold:
    def test_small_dropped_rate_is_skipped(self, no_heavy_objects):
        basin = make_basin(basin_config(psr_skip_rate_thr=0.01))
        result = explore_from_state_0(basin, fail_event(10), k10=0.005)

        assert result.is_ok()
        assert basin.n_psr_skipped == 1
        assert basin.psr_max_rate_fraction == pytest.approx(0.005 / 1.005)

    def test_large_dropped_rate_aborts(self, no_heavy_objects):
        basin = make_basin(basin_config(psr_skip_rate_thr=0.01))
        result = explore_from_state_0(basin, fail_event(10), k10=0.5)

        assert not result.is_ok()
        err = result.err_value()
        assert err.type == ErrorType.BASIN_PSR_SKIP_RATE_ABOVE_THRESHOLD
        assert err.variables["state"] == 0
        assert err.variables["rate_fraction"] == pytest.approx(0.5 / 1.5)

    def test_zero_threshold_aborts_on_any_failure(self, no_heavy_objects):
        basin = make_basin(basin_config(psr_skip_rate_thr=0.0))
        result = explore_from_state_0(basin, fail_event(10), k10=1e-9)

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.BASIN_PSR_SKIP_RATE_ABOVE_THRESHOLD

    def test_dropped_rate_accumulates_per_state(self):
        """A state's earlier drops count towards the bound of a later one."""
        basin = make_basin(basin_config(psr_skip_rate_thr=0.01))
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(
            [
                row(0, 1, 1, 0, True),
                row(1, 0, 2, 0, True, k=0.994),
                row(1, 2, 3, 0, False, k=0.006),  # dropped
                row(0, 3, 4, 0, False),
            ]
        )
        basin.psr_dropped_rate = {1: 0.006}  # dropped while building states

        result = basin._drop_exit_transitions([2])

        assert not result.is_ok()
        err = result.err_value()
        assert err.type == ErrorType.BASIN_PSR_SKIP_RATE_ABOVE_THRESHOLD
        assert err.variables["rate_fraction"] == pytest.approx(0.012 / 1.006)

    def test_dropped_rate_follows_state_renumbering(self):
        basin = make_basin(basin_config(psr_skip_rate_thr=0.5))
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(
            [
                row(0, 2, 1, 0, True),
                row(2, 0, 2, 0, True),
                row(0, 1, 3, 0, False),
                row(2, 3, 4, 0, False),
            ]
        )
        basin.states = {0: "state0", 2: "state2"}
        basin.psr_dropped_rate = {2: 0.1}

        # Absorbing state 1 loses its only transition: transient 2 becomes 1
        result = basin._drop_exit_transitions([2])

        assert result.is_ok()
        assert basin.psr_dropped_rate == {0: 1.0, 1: 0.1}
        assert basin.states == {0: "state0", 1: "state2"}


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
        basin.states = {0: "state0", 1: "state1"}

        result = basin._drop_exit_transitions([3])  # 1 -> 3

        df = basin.connectivity_table.df
        assert result.is_ok()
        assert list(df.index) == [0, 1, 2, 4]  # labels kept for futures_context
        assert sorted(set(df["state"]) | set(df["state_connexion"])) == [0, 1, 2, 3]
        assert basin.states == {0: "state0", 1: "state1"}
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


def small_system():
    return System(
        types=["Si", "Si"],
        positions=np.zeros((2, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
        index=np.arange(2),
    )


def mapped_event():
    return Ok(
        MappedEvent(
            system=small_system(),
            neighbors=np.array([0, 1]),
            initial_positions=np.zeros((2, 3)),
            saddle_positions=np.full((2, 3), 0.5),
            final_positions=np.ones((2, 3)),
        )
    )


def refine_manager(E_saddle=-0.7):
    manager = Mock()
    manager.get_total_energy.return_value = Mock(result=Mock(return_value=-1.0))
    manager.partn_refine.return_value = Mock(
        result=Mock(
            return_value=Ok(
                SimpleNamespace(E_saddle=E_saddle, saddle_positions=np.zeros((2, 3)))
            )
        )
    )
    return manager


class TestRefineAbsorbing:
    def test_psr_failure_skips_exit_and_refines_others(self, no_heavy_objects):
        config = basin_config(
            psr_skip_rate_thr=0.01,
            control=SimpleNamespace(active_volume=False),
            rateconstant=SimpleNamespace(k0=10.0, T=300.0),
        )
        manager = refine_manager()
        basin = make_basin(config=config, manager=manager)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(
            [
                row(0, 1, 5, 0, True),
                row(1, 0, 5, 0, True),
                row(0, 2, 5, 99, False, k=0.005),  # PSR fails, 0.5% of state 0
                row(1, 3, 5, 1, False),
            ]
        )
        basin.states = {
            i: StateData(system=small_system(), environment=None, neighbors_list=None)
            for i in range(2)
        }
        basin.map_event = lambda state, event, atom, sym: (
            PSR_ERR if atom == 99 else mapped_event()
        )

        result = basin.refine_absorbing(system=None)

        df = basin.connectivity_table.df
        assert result.is_ok()
        assert basin.n_psr_skipped == 1
        assert basin.psr_max_rate_fraction == pytest.approx(0.005 / 1.005)
        assert manager.partn_refine.call_count == 1
        assert list(df["central_atom"]) == [0, 0, 1]
        assert df.loc[3, "state_connexion"] == 2  # old state 3 reindexed
        assert df.loc[3, "dE_forward"] == pytest.approx(0.3)
        assert set(basin.absorbing_saddle_positions) == {3}  # row label
