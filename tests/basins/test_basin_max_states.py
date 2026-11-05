"""A basin whose exploration keeps finding new transient states stops at
[Basin] max_states, and the exploration reports its size in the log."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc.basins import basin as basin_module
from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.result import ErrorType, Ok


def row(state, state_connexion, transient):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": 1,
        "central_atom": 0,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.4,
        "k_forward": 1.0,
        "dE_backward": 0.4,
        "k_backward": 1.0,
    }


class ChainExplorer:
    """Each explored state leads to one new state; the chain ends with an exit
    after `length` states (never if None)."""

    def __init__(self, length=None):
        self.length = length
        self.connectivity_table = BasinStatesConnectivity()

    def explore(self, state, state_index, start_index):
        transient = self.length is None or start_index < self.length
        self.connectivity_table.df = pd.DataFrame(
            [row(state_index, start_index, transient)]
        )

    def clear(self):
        self.connectivity_table.clear()


def make_basin(monkeypatch, max_states, length=None):
    monkeypatch.setattr(StateData, "ensure_full_state", lambda self, config: None)
    monkeypatch.setattr(StateData, "release_heavy_objects", lambda self: None)
    loggers = Mock()
    basin = BasinsGenericEvents(
        config=SimpleNamespace(basin=SimpleNamespace(max_states=max_states)),
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
    basin.states = {0: StateData(system=Mock(), environment=None, neighbors_list=None)}
    basin.system_from_state = lambda *args: Ok(
        SimpleNamespace(positions=np.zeros((1, 3)))
    )
    basin.is_new_state = lambda system: -1
    basin.is_states_has_unknown_environments = lambda state: False
    return basin, loggers


def logged(loggers):
    return [c.args[1] for c in loggers.info.call_args_list]


def test_exploration_stops_at_max_states(monkeypatch):
    basin, loggers = make_basin(monkeypatch, max_states=250)

    result = basin.construct_connexion_table()

    err = result.err_value()
    assert err.type == ErrorType.BASIN_TOO_MANY_STATES
    assert err.variables == {"n_states": 250}
    assert len(basin.states) == 250
    progress = [m for m in logged(loggers) if "Basin exploration" in m]
    assert progress == [
        "\t :=> Basin exploration: 100 states, 0 left to explore",
        "\t :=> Basin exploration: 200 states, 0 left to explore",
    ]


@pytest.mark.parametrize("max_states", [0, 1000])
def test_finite_basin_is_not_limited(monkeypatch, max_states):
    basin, loggers = make_basin(monkeypatch, max_states=max_states, length=150)

    assert basin.construct_connexion_table().is_ok()
    assert len(basin.states) == 150  # states 0 to 149, the exit is not built
    assert logged(loggers)[-1] == "\t :=> Basin explored: 150 states, 1 exits"


def test_no_limit(monkeypatch):
    """max_states = 0 explores past the default limit."""
    basin, _ = make_basin(monkeypatch, max_states=0, length=1200)

    assert basin.construct_connexion_table().is_ok()
    assert len(basin.states) == 1200


def test_basin_without_loggers(monkeypatch):
    basin, _ = make_basin(monkeypatch, max_states=1000, length=3)
    basin.loggers = None

    assert basin.construct_connexion_table().is_ok()
