"""Basin exits with a negligible share of the exit rate are not refined unless
the selector picks one."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.basins.basin import MappedEvent
from pykmc.result import Ok

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
    row(1, 0, 10, True, 0.02),
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

        negligible = [basin._is_negligible_exit(i) for i in (2, 3, 4, 5)]

        assert negligible == [False, False, True, True]

    def test_threshold_one_keeps_every_exit(self):
        basin = make_basin(exit_refine_thr=1.0)
        basin.connectivity_table = BasinStatesConnectivity()
        basin.connectivity_table.df = pd.DataFrame(STATE0_ROWS)

        assert not any(basin._is_negligible_exit(i) for i in (2, 3, 4, 5))


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


def refine_ready_basin(manager):
    basin = make_basin(manager=manager)
    basin.connectivity_table = BasinStatesConnectivity()
    basin.connectivity_table.df = pd.DataFrame(STATE0_ROWS)
    basin.states = {
        i: StateData(system=small_system(), environment=None, neighbors_list=None)
        for i in range(2)
    }
    mapped = []

    def map_event(state, event, atom, sym):
        mapped.append(atom)
        return Ok(
            MappedEvent(
                system=small_system(),
                neighbors=np.array([0, 1]),
                initial_positions=np.zeros((2, 3)),
                saddle_positions=np.zeros((2, 3)),
                final_positions=np.full((2, 3), float(atom)),
            )
        )

    basin.map_event = map_event
    return basin, mapped


def test_refine_skips_deferred_exits(no_heavy_objects):
    manager = refine_manager()
    basin, mapped = refine_ready_basin(manager)

    result = basin.refine_absorbing(system=None)

    df = basin.connectivity_table.df
    assert result.is_ok()
    assert basin.deferred_exit_rows == {4, 5}
    assert mapped == [11, 12]
    assert manager.partn_refine.call_count == 2  # exits 2 and 3 only
    assert set(basin.absorbing_saddle_positions) == {2, 3}
    # deferred exits keep their generic barrier, with the basin's constant prefactor
    assert df.loc[4, "dE_forward"] == 2.90
    assert df.loc[4, "k_forward"] == pytest.approx(rate(2.90))
    assert df.loc[2, "k_forward"] == pytest.approx(rate(0.3))  # refined


def test_selected_deferred_exit_is_refined(no_heavy_objects):
    manager = refine_manager()
    basin, mapped = refine_ready_basin(manager)
    basin.deferred_exit_rows = {4, 5}
    basin.selector = Mock(
        select_from_connectivity=Mock(
            return_value=Ok(SimpleNamespace(t_exit=1.0, exit_state=4))
        )
    )

    out = basin.select_exit().ok_value()

    assert manager.partn_refine.call_count == 1
    assert basin.deferred_exit_rows == {5}
    assert out.central_atom == 13
    assert out.saddle_positions.shape == (2, 3)
    assert out.energy_barrier == pytest.approx(0.3)  # refined on demand
    np.testing.assert_array_equal(out.final_positions, np.full((2, 3), 13.0))
