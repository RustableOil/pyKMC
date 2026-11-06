"""A basin whose states move atoms farther than [Basin] max_radius from the basin
centre is abandoned, and the exploration reports how far the basin extends."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.result import ErrorType

CELL = np.diag([100.0, 20.0, 20.0])
N_ATOMS = 30
SPACING = 2.0  # atom i sits at x = 2 i


def row(state, state_connexion, transient):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": 1,
        "central_atom": 0,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.1,
        "k_forward": 1.0,
        "dE_backward": 0.1,
        "k_backward": 1.0,
    }


class ChainExplorer:
    """Each explored state leads to one new state; the chain ends with an exit
    after `length` states."""

    def __init__(self, length):
        self.length = length
        self.connectivity_table = BasinStatesConnectivity()

    def explore(self, state, state_index, start_index):
        self.connectivity_table.df = pd.DataFrame(
            [row(state_index, start_index, start_index < self.length)]
        )

    def clear(self):
        self.connectivity_table.clear()


def chain_positions():
    positions = np.zeros((N_ATOMS, 3))
    positions[:, 0] = SPACING * np.arange(N_ATOMS)
    positions[:, 1:] = 10.0
    return positions


def make_basin(monkeypatch, max_radius, length=10, center=0):
    """State n has atoms 1 to n moved by 0.5 A along y: its extent from atom 0
    is that of atom n, about 2 n A."""
    monkeypatch.setattr(StateData, "ensure_full_state", lambda self, config: None)
    monkeypatch.setattr(StateData, "release_heavy_objects", lambda self: None)
    loggers = Mock()
    config = SimpleNamespace(
        basin=SimpleNamespace(max_states=0, max_radius=max_radius),
        control=SimpleNamespace(disp_thr=0.05),
    )
    basin = BasinsGenericEvents(
        config=config,
        reference_table=Mock(),
        known_environments=set(),
        manager=Mock(),
        loggers=loggers,
    )
    basin.current_state = 0
    basin.states_to_explore = [0]
    basin.explored_states = []
    basin.connectivity_table = BasinStatesConnectivity()
    basin.explorer = ChainExplorer(length)
    state0 = SimpleNamespace(positions=chain_positions(), cell=CELL)
    basin.states = {0: StateData(system=state0, environment=None, neighbors_list=None)}
    if center is not None:
        basin.center_position = state0.positions[center].copy()

    def system_from_state(from_state, event_idx, central_atom, sym_idx):
        n = len(basin.states)  # index of the state being built
        positions = chain_positions()
        positions[1 : n + 1, 1] += 0.5
        return SimpleNamespace(is_ok=lambda: True, ok_value=lambda: system(positions))

    def system(positions):
        return SimpleNamespace(positions=positions, cell=CELL)

    basin.system_from_state = system_from_state
    basin.is_new_state = lambda system: -1
    basin.is_states_has_unknown_environments = lambda state: False
    return basin, loggers


def extent_of_state(n):
    return float(np.hypot(SPACING * n, 0.5))


def logged(loggers):
    return [c.args[1] for c in loggers.info.call_args_list]


def test_exploration_stops_beyond_max_radius(monkeypatch):
    basin, _ = make_basin(monkeypatch, max_radius=7.0)

    result = basin.construct_connexion_table()

    err = result.err_value()
    assert err.type == ErrorType.BASIN_OUTSIDE_RADIUS
    assert err.variables["state"] == 4  # first state with an atom beyond 7 A
    assert err.variables["extent"] == pytest.approx(extent_of_state(4))
    assert "max_radius = 7.0 A" in err.message


def test_basin_within_max_radius_reports_its_extent(monkeypatch):
    basin, loggers = make_basin(monkeypatch, max_radius=25.0, length=10)

    assert basin.construct_connexion_table().is_ok()
    assert basin.max_extent == pytest.approx(extent_of_state(9))
    assert logged(loggers)[-1] == (
        "\t :=> Basin explored: 10 states, 1 exits, extent {:.2f} A".format(
            extent_of_state(9)
        )
    )


def test_zero_radius_is_no_limit(monkeypatch):
    basin, _ = make_basin(monkeypatch, max_radius=0.0, length=10)

    assert basin.construct_connexion_table().is_ok()


def test_extent_across_periodic_boundary(monkeypatch):
    """The centre is the last atom of the chain, at x = 58 A: atoms 1 to n are
    nearer through the boundary than within the cell."""
    basin, _ = make_basin(monkeypatch, max_radius=0.0, center=N_ATOMS - 1)

    extent = basin._extent(
        SimpleNamespace(
            positions=chain_positions()
            + np.where(np.arange(N_ATOMS) == 1, 0.5, 0.0)[:, None] * [0, 1, 0],
            cell=CELL,
        )
    )

    # atom 1 at x = 2, centre at x = 58: 56 A in the cell, 44 A through the boundary
    assert extent == pytest.approx(np.hypot(44.0, 0.5))


def test_no_center_no_check(monkeypatch):
    basin, loggers = make_basin(monkeypatch, max_radius=1.0, center=None)

    assert basin.construct_connexion_table().is_ok()
    assert logged(loggers)[-1] == "\t :=> Basin explored: 10 states, 1 exits"
