"""Exit states of a basin are not built during its exploration: only transient
states are reconstructed, and the selected exit's final positions come from the
generic event mapped onto its from state."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.result import Err, ErrorInfo, ErrorType, Ok


def row(state, state_connexion, event, transient, k=1.0):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": event,
        "central_atom": 0,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.1 if transient else 1.0,
        "k_forward": k,
        "dE_backward": 0.1,
        "k_backward": 1.0,
    }


class TwoStateExplorer:
    """State 0 hops to the transient state 1 and has two exits; state 1 has
    one exit."""

    def __init__(self):
        self.connectivity_table = BasinStatesConnectivity()

    def explore(self, state, state_index, start_index):
        if state_index == 0:
            rows = [row(0, 1, 10, True), row(0, 2, 20, False), row(0, 3, 30, False)]
        else:
            rows = [row(state_index, start_index, 40, False)]
        self.connectivity_table.df = pd.DataFrame(rows)

    def clear(self):
        self.connectivity_table.clear()


def make_basin(monkeypatch):
    monkeypatch.setattr(StateData, "ensure_full_state", lambda self, config: None)
    monkeypatch.setattr(StateData, "release_heavy_objects", lambda self: None)
    basin = BasinsGenericEvents(
        config=SimpleNamespace(basin=SimpleNamespace(max_states=0)),
        reference_table=Mock(),
        known_environments=set(),
        manager=Mock(),
    )
    basin.current_state = 0
    basin.states_to_explore = [0]
    basin.explored_states = []
    basin.connectivity_table = BasinStatesConnectivity()
    basin.explorer = TwoStateExplorer()
    basin.states = {0: StateData(system=Mock(), environment=None, neighbors_list=None)}
    built = []

    def system_from_state(from_state, event_idx, central_atom, sym_idx):
        built.append(event_idx)
        return Ok(SimpleNamespace(positions=np.full((1, 3), float(event_idx))))

    basin.system_from_state = system_from_state
    basin.is_new_state = lambda system: -1
    basin.is_states_has_unknown_environments = lambda state: False
    return basin, built


def test_only_transient_states_are_built(monkeypatch):
    basin, built = make_basin(monkeypatch)

    assert basin.construct_connexion_table().is_ok()

    assert built == [10]
    assert sorted(basin.states) == [0, 1]
    df = basin.connectivity_table.df
    assert int((~df["transient"].astype(bool)).sum()) == 3
    # every exit is still in the table, as an absorbing state
    assert set(df.loc[~df["transient"].astype(bool), "state_connexion"]) == {2, 3, 4}


def test_transient_state_that_cannot_be_built_becomes_an_exit(monkeypatch):
    basin, built = make_basin(monkeypatch)
    basin.loggers = Mock()
    basin.system_from_state = lambda *args: Err(
        ErrorInfo(type=ErrorType.RECONSTRUCTION_INVALID_MIN2, message="delr2 = 0.4")
    )

    assert basin.construct_connexion_table().is_ok()

    assert sorted(basin.states) == [0]
    df = basin.connectivity_table.df
    assert not df["transient"].astype(bool).any()
    assert set(df["state_connexion"]) == {1, 2, 3}
    messages = [c.args[1] for c in basin.loggers.info.call_args_list]
    assert any(
        "Basin state 1 (reference event 10, atom 0) could not be built, treated as "
        "an exit: RECONSTRUCTION_INVALID_MIN2: delr2 = 0.4" in m
        for m in messages
    )
