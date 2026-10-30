"""A failed reconstruction reports each atom's deviation from the expected
minimum together with its distance from the event's central atom."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pykmc.reconstruction import Reconstruction, deviation_diagnostics
from pykmc.result import ErrorType

CELL = np.diag([20.0, 20.0, 20.0])
THR = 0.1


def environment():
    """Central atom (system index 7) at (10, 10, 10) with neighbors at 1 to 5 A."""
    offsets = np.array(
        [[0, 2, 0], [1, 0, 0], [0, 0, 0], [0, 0, 3], [-4, 0, 0], [0, -5, 0]], float
    )
    return np.array([4, 9, 7, 2, 11, 5]), 10.0 + offsets


def test_deviation_diagnostics_reports_edge_atom():
    neighbors, expected = environment()
    obtained = expected.copy()
    obtained[5, 0] += 0.3  # atom 5 A from the center
    obtained[1, 2] += 0.15  # atom 1 A from the center, the central atom is index 2

    summary, variables = deviation_diagnostics(
        expected, obtained, CELL, neighbors, 7, THR
    )

    assert summary.startswith("2/6 atoms above 0.1 A")
    assert "0.300@5.00, 0.150@1.00" in summary
    assert summary.endswith("central atom deviation 0.000")
    np.testing.assert_allclose(variables["deviation"][[1, 5]], [0.15, 0.3])
    np.testing.assert_allclose(variables["distance_from_center"], [2, 1, 0, 3, 4, 5])
    np.testing.assert_array_equal(variables["atom_indices"], neighbors)


def test_deviation_diagnostics_across_periodic_boundary():
    neighbors, expected = environment()
    expected = (expected + 9.5) % 20.0  # central atom at 19.5, wraps the box
    obtained = expected.copy()
    obtained[4, 1] += 0.2

    summary, variables = deviation_diagnostics(
        expected, obtained, CELL, neighbors, 7, THR
    )

    assert "0.200@4.00" in summary
    np.testing.assert_allclose(variables["distance_from_center"], [2, 1, 0, 3, 4, 5])


def test_deviation_diagnostics_without_central_atom():
    neighbors, expected = environment()
    summary, variables = deviation_diagnostics(
        expected, expected.copy(), CELL, neighbors, None, THR
    )

    center = expected.mean(axis=0)
    np.testing.assert_allclose(
        variables["distance_from_center"], np.linalg.norm(expected - center, axis=1)
    )
    assert summary.startswith("0/6 atoms above")
    assert "central atom" not in summary


def test_reconstruct_min2_error_carries_diagnostics():
    neighbors, expected = environment()
    system_positions = np.full((12, 3), 1.0)
    system_positions[neighbors] = expected
    min2_pos = system_positions.copy()
    min2_pos[neighbors[5], 0] += 0.3  # edge atom settles differently

    manager = Mock()
    manager.group_minimize_with_results.side_effect = [
        (system_positions.copy(), -1.0),
        (min2_pos, -2.0),
    ]
    config = SimpleNamespace(
        psr=SimpleNamespace(matching_score_thr=THR),
        reconstruction=SimpleNamespace(push_fraction=0.15),
    )

    result = Reconstruction(config, manager).reconstruct(
        expected, expected, system_positions, CELL, THR, neighbors, central_atom=7
    )

    err = result.err_value()
    assert err.type == ErrorType.RECONSTRUCTION_INVALID_MIN2
    assert "delr2 = 0.3" in err.message
    assert "1/6 atoms above 0.1 A" in err.message
    assert "0.300@5.00" in err.message
    np.testing.assert_allclose(err.variables["deviation"][5], 0.3)
