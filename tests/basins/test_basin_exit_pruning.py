"""Basin exits with a negligible share of the exit rate are neither built nor
refined unless the selector picks one."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.basins import basin as basin_module
from pykmc.result import Ok, PSROutput

K0, T = 10.0, 500.0
KB = 8.6173303e-05


def rate(dE):
    return K0 * np.exp(-dE / (KB * T))


def row(state, state_connexion, central_atom, transient, dE):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": 5,
        "central_atom": central_atom,
        "sym": 0,
        "transient": transient,
        "dE_forward": dE,
        "k_forward": 1.0,
        "dE_backward": 0.1,
        "k_backward": 1.0,
    }


def make_basin(exit_refine_thr=0.9999, manager=None):
    config = SimpleNamespace(
        basin=SimpleNamespace(psr_skip_rate_thr=0.01, exit_refine_thr=exit_refine_thr),
        rateconstant=SimpleNamespace(k0=K0, T=T),
        control=SimpleNamespace(active_volume=False),
        psr=SimpleNamespace(matching_score_thr=0.1),
    )
    return BasinsGenericEvents(
        config=config,
        reference_table=Mock(),
        known_environments=set(),
        manager=manager or Mock(),
    )


def small_system():
    return System(
        types=["Si", "Si"],
        positions=np.zeros((2, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
        index=np.arange(2),
    )


@pytest.fixture
def no_heavy_objects(monkeypatch):
    monkeypatch.setattr(StateData, "ensure_full_state", lambda self, config: None)
    monkeypatch.setattr(StateData, "release_heavy_objects", lambda self: None)


# Exits of state 0: two low barriers carry the escape rate, two high ones do not
STATE0_ROWS = [
    row(0, 1, 10, True, 0.02),  # internal transition, never an exit
    row(0, 2, 11, False, 0.18),
    row(0, 3, 12, False, 0.20),
    row(0, 4, 13, False, 2.90),
    row(0, 5, 14, False, 3.00),
]


class TestNegligibleExit:
    def test_high_barrier_exits_are_negligible(self):
        basin = make_basin(exit_refine_thr=0.9999)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(STATE0_ROWS)

        negligible = [basin._is_negligible_exit(i) for i in (1, 2, 3, 4)]

        assert negligible == [False, False, True, True]

    def test_threshold_one_keeps_every_exit(self):
        basin = make_basin(exit_refine_thr=1.0)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(STATE0_ROWS)

        assert not any(basin._is_negligible_exit(i) for i in (1, 2, 3, 4))


class FakeExplorer:
    def __init__(self):
        self.connectivity_table = BasinStatesConnectivity()

    def explore(self, state, state_index, start_index):
        rows = [
            row(state_index, start_index + i, r["central_atom"], False, r["dE_forward"])
            for i, r in enumerate(STATE0_ROWS[1:])
        ]
        self.connectivity_table.df = pd.DataFrame(rows)

    def clear(self):
        self.connectivity_table.clear()


def test_construct_does_not_build_negligible_exits(no_heavy_objects):
    basin = make_basin()
    basin.current_state = 0
    basin.states_to_explore = [0]
    basin.explored_states = []
    basin.connectivity_table = BasinStatesConnectivity()
    basin.explorer = FakeExplorer()
    basin.states = {0: StateData(system=Mock(), environment=None, neighbors_list=None)}
    built = []
    basin.system_from_state = lambda from_state, event, atom, sym: (
        built.append(atom) or Ok(Mock())
    )
    basin.is_new_state = lambda system: -1
    basin.is_states_has_unknown_environments = lambda state: False

    result = basin.construct_connexion_table()

    df = basin.connectivity_table.df
    assert result.is_ok()
    assert built == [11, 12]  # only the exits carrying the escape rate
    assert set(df.loc[list(basin.deferred_exit_rows), "central_atom"]) == {13, 14}
    assert len(df) == 4  # deferred exits stay in the table


def refine_ready_basin(manager):
    basin = make_basin(manager=manager)
    basin.reference_table.table = pd.DataFrame(
        {"idx_ref": [5], "saddle_positions": [np.zeros((2, 3))], "sym_matrix": [None]}
    )
    basin.connectivity_table = BasinStatesConnectivity()
    basin.connectivity_table.df = pd.DataFrame(STATE0_ROWS)
    basin.deferred_exit_rows = {3, 4}
    neighbors_list = Mock(get_neighbors=Mock(return_value=np.array([0, 1])))
    basin.states = {
        i: StateData(
            system=small_system(), environment=None, neighbors_list=neighbors_list
        )
        for i in range(4)
    }
    return basin


def refine_manager(E_saddle=-0.7):
    manager = Mock()
    manager.get_total_energy.return_value = Mock(result=Mock(return_value=-1.0))
    manager.partn_refine.return_value = Mock(
        result=Mock(
            return_value=Ok(
                SimpleNamespace(E_saddle=E_saddle, saddle_positions=np.ones((2, 3)))
            )
        )
    )
    return manager


@pytest.fixture
def identity_psr(monkeypatch):
    class FakePSR:
        def __init__(self, *args):
            pass

        def match(self):
            return Ok(
                PSROutput(
                    rotation_matrix=np.eye(3),
                    translation_matrix=np.zeros(3),
                    permutation_matrix=np.arange(2),
                    matching_score=0.0,
                )
            )

    monkeypatch.setattr(basin_module, "PointSetRegistration", FakePSR)


def test_refine_skips_deferred_exits(no_heavy_objects, identity_psr):
    manager = refine_manager()
    basin = refine_ready_basin(manager)

    result = basin.refine_absorbing(system=None)

    df = basin.connectivity_table.df
    assert result.is_ok()
    assert manager.partn_refine.call_count == 2  # exits 1 and 2 only
    assert set(basin.absorbing_saddle_positions) == {1, 2}
    # deferred exits keep their generic barrier, with the basin's constant prefactor
    assert df.loc[3, "dE_forward"] == 2.90
    assert df.loc[3, "k_forward"] == pytest.approx(rate(2.90))
    assert df.loc[1, "k_forward"] == pytest.approx(rate(0.3))  # refined


def test_selected_deferred_exit_is_built_and_refined(no_heavy_objects, identity_psr):
    manager = refine_manager()
    basin = refine_ready_basin(manager)
    basin._initialize = lambda system: None
    basin.construct_connexion_table = lambda: Ok(None)
    basin.refine_absorbing = lambda system: Ok(None)
    basin.selector = Mock(
        select_from_connectivity=Mock(
            return_value=Ok(SimpleNamespace(t_exit=1.0, exit_state=4))
        )
    )
    built = []
    basin.system_from_state = lambda from_state, event, atom, sym: (
        built.append(atom) or Ok(small_system())
    )

    out = basin.execute(system=None).ok_value()

    assert built == [13]
    assert 4 in basin.states and basin.states[4].transient is False
    assert manager.partn_refine.call_count == 1
    assert out.central_atom == 13
    assert out.saddle_positions.shape == (2, 3)
    assert out.energy_barrier == pytest.approx(0.3)  # refined on demand
