"""When several central atoms lead to the same exit state, the basin output
(saddle, central atom, neighbors, barrier) comes from a single transition."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.basins import BasinsGenericEvents, BasinStatesConnectivity, StateData
from pykmc.basins.basin import MappedEvent
from pykmc.result import Ok


def row(state, state_connexion, event, central_atom, transient, dE):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": event,
        "central_atom": central_atom,
        "sym": 0,
        "transient": transient,
        "dE_forward": dE,
        "k_forward": 1.0,
        "dE_backward": 0.1,
        "k_backward": 1.0,
    }


def test_output_comes_from_one_transition(monkeypatch):
    monkeypatch.setattr(StateData, "ensure_full_state", lambda self, config: None)
    basin = BasinsGenericEvents(
        config=Mock(), reference_table=Mock(), known_environments=set(), manager=Mock()
    )
    basin._initialize = lambda system: None
    basin.construct_connexion_table = lambda: Ok(None)
    basin.connectivity_table = BasinStatesConnectivity()
    basin.connectivity_table.df = pd.DataFrame(
        [
            row(0, 1, 1, 0, True, 0.02),
            row(1, 0, 1, 0, True, 0.02),
            # two exits from state 1 to the same exit state 2, through central
            # atoms with environments of different sizes
            row(1, 2, 173, 7, False, 0.18),
            row(1, 2, 371, 8, False, 0.19),
        ]
    )
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
        # refined in row order: the saddle of row 3 is stored last
        basin.absorbing_saddle_positions = {2: np.ones((3, 3)), 3: np.ones((4, 3))}
        return Ok(None)

    basin.refine_absorbing = refine

    def map_event(from_state, event_idx, central_atom, sym_idx):
        n = len(neighbors[central_atom])
        return Ok(
            MappedEvent(
                system=None,
                neighbors=neighbors[central_atom],
                initial_positions=np.zeros((n, 3)),
                saddle_positions=np.ones((n, 3)),
                final_positions=np.full((n, 3), 2.0),
            )
        )

    basin.map_event = map_event
    basin.selector = Mock(
        select_from_connectivity=Mock(
            return_value=Ok(SimpleNamespace(t_exit=1.0, exit_state=2))
        )
    )

    out = basin.execute(system=None).ok_value()

    assert out.central_atom == 7
    assert out.num_reference_event == 173
    assert len(out.neighbors) == len(out.saddle_positions) == 3
    assert len(out.final_positions) == 3
    assert out.energy_barrier == 0.18
