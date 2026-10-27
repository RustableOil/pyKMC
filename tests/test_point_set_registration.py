"""IRA point set registration is anchored on the event's central atom."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import PointSetRegistration, System
from pykmc import point_set_registration as psr_module

CONFIG = SimpleNamespace(
    psr=SimpleNamespace(style="ira"),
    atomicenvironment=SimpleNamespace(atom_coloring_mode="grey"),
    ira=SimpleNamespace(kmax_factor=1.8),
)


def make_psr(env_positions, neighbors, central_atom, ref_event):
    system = System(
        types=np.array(["Si"] * len(env_positions)),
        positions=env_positions,
        cell=np.eye(3) * 50.0,
        pbc=np.array([True, True, True]),
        index=np.arange(len(env_positions)),
    )
    neighbors_list = Mock(get_neighbors=Mock(return_value=neighbors))
    return PointSetRegistration(CONFIG, system, ref_event, neighbors_list, central_atom)


@pytest.fixture
def cluster():
    rng = np.random.default_rng(0)
    return rng.uniform(20.0, 30.0, size=(12, 3))


class TestCandidates:
    @pytest.fixture
    def fake_ira(self, monkeypatch):
        ira = Mock()
        ira.match.return_value = (np.eye(3), np.zeros(3), np.arange(5), 0.0)
        monkeypatch.setattr(psr_module.ira_mod, "IRA", Mock(return_value=ira))
        return ira

    def test_anchored_on_central_atoms(self, fake_ira, cluster):
        ref_event = pd.Series(
            {"initial_positions": cluster[:5].copy(), "move_atom_idx": 3}
        )
        make_psr(cluster, [4, 7, 2, 9, 1], 9, ref_event).match()

        kwargs = fake_ira.match.call_args.kwargs
        assert kwargs["candidate1"] == 3  # position of atom 9 in its neighbor list
        assert kwargs["candidate2"] == 3

    def test_geometric_centre_without_move_atom_idx(self, fake_ira, cluster):
        ref_event = pd.Series({"initial_positions": cluster[:5].copy()})
        make_psr(cluster, [4, 7, 2, 9, 1], 9, ref_event).match()

        kwargs = fake_ira.match.call_args.kwargs
        assert kwargs["candidate1"] is None and kwargs["candidate2"] is None


def test_central_atom_maps_onto_reference_central_atom(cluster):
    # Reference = rotated, translated and shuffled copy of the environment
    neighbors = np.arange(len(cluster))
    central = 5
    rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    order = np.random.default_rng(2).permutation(len(cluster))
    ref_positions = (cluster @ rotation.T + 1.5)[order]
    ref_event = pd.Series(
        {
            "initial_positions": ref_positions,
            "move_atom_idx": int(np.flatnonzero(order == central)[0]),
        }
    )

    result = make_psr(cluster, neighbors, central, ref_event).match()

    out = result.ok_value()
    assert out.matching_score == pytest.approx(0.0, abs=1e-6)
    # ref atoms transformed onto the environment: the central atoms coincide
    aligned = (ref_positions @ out.rotation_matrix.T + out.translation_matrix)[
        out.permutation_matrix
    ]
    np.testing.assert_allclose(aligned, cluster, atol=1e-6)
