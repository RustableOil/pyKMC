"""A selected basin exit that cannot be reconstructed is dropped, its rate is
removed from the escape rate, and another exit is selected."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.basins.basin import MappedEvent
from pykmc.basins.selection import FPTASelector
from pykmc.psr.result import PSRError
from pykmc.result import Err, ErrorInfo, ErrorType, Ok


def row(state, state_connexion, event, transient, k):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": event,
        "central_atom": event,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.02 if transient else 0.2,
        "k_forward": k,
        "dE_backward": 0.02,
        "k_backward": k,
    }


def make_basin(unmappable=()):
    basin = BasinsGenericEvents(
        config=Mock(), reference_table=Mock(), known_environments=set(), manager=Mock()
    )
    basin._initialize = lambda system: None
    basin.loggers = Mock()
    basin.connectivity_table = BasinStatesConnectivity()
    # two transient states and three exits (states 2, 3, 4)
    basin.connectivity_table.df = pd.DataFrame(
        [
            row(0, 1, 1, True, 5.0),
            row(1, 0, 2, True, 5.0),
            row(0, 2, 10, False, 1.0),
            row(1, 3, 11, False, 1.0),
            row(0, 4, 12, False, 2.0),
        ]
    )
    basin.selector = FPTASelector()
    basin.states = {
        i: StateData(
            system=SimpleNamespace(positions=np.zeros((20, 3))),
            environment=None,
            neighbors_list=None,
        )
        for i in range(2)
    }
    basin.absorbing_saddle_positions = {i: np.ones((2, 3)) for i in range(5)}

    def map_event(from_state, event_idx, central_atom, sym_idx):
        if event_idx in unmappable:
            return Err(ErrorInfo(type=PSRError.NO_MATCH_FOUND, message="no match"))
        return Ok(
            MappedEvent(
                system=None,
                neighbors=np.array([0, 1]),
                initial_positions=np.zeros((2, 3)),
                saddle_positions=np.ones((2, 3)),
                final_positions=np.full((2, 3), float(event_idx)),
            )
        )

    basin.map_event = map_event
    return basin


def logged(basin):
    return [c.args[1] for c in basin.loggers.info.call_args_list]


def test_failed_exit_is_dropped_and_another_selected():
    np.random.seed(0)
    basin = make_basin()
    first = basin.select_exit().ok_value()

    second = basin.reselect_exit(first.exit_state, "reconstruction failed")

    out = second.ok_value()
    assert out.exit_state != first.exit_state
    df = basin.connectivity_table.df
    assert (df.loc[df["state_connexion"] == first.exit_state, "k_forward"] == 0).all()
    assert basin.n_dropped_exits == 1
    assert "dropped: reconstruction failed" in logged(basin)[-1]


def test_dropped_rate_is_removed_from_k_tot():
    np.random.seed(1)
    basin = make_basin()
    first = basin.select_exit().ok_value()
    assert first.k_tot == 4.0

    out = basin.reselect_exit(first.exit_state, "reconstruction failed").ok_value()

    assert out.k_tot == 4.0 - basin.dropped_exit_rate


def test_unmappable_exit_is_skipped():
    """An exit whose generic event no longer maps is never returned."""
    for seed in range(10):
        np.random.seed(seed)
        basin = make_basin(unmappable=(12,))
        out = basin.select_exit().ok_value()
        assert out.num_reference_event != 12


def test_basin_fails_when_every_exit_is_dropped():
    np.random.seed(2)
    basin = make_basin()
    result = basin.select_exit()
    for _ in range(3):
        result = basin.reselect_exit(result.ok_value().exit_state, "failed")
        if not result.is_ok():
            break

    err = result.err_value()
    assert err.type == ErrorType.BASIN_EXIT_FAILED
    assert err.variables["n_dropped_exits"] == 3
    assert err.variables["dropped_rate_fraction"] == 1.0
