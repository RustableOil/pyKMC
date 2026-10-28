"""Event symmetries are applied about the center SOFI computed them for."""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from pykmc.basins.basin import apply_event_symmetry
from pykmc.symmetries import unique_symmetries
from pykmc.utils import geometry


@pytest.fixture
def ref_event():
    """Si vacancy-like event centered in a 43.45 A box, as stored by pyKMC."""
    atoms = bulk("Si", "diamond", a=5.431, cubic=True).repeat(3)
    positions = atoms.get_positions()
    center = positions[np.argmin(np.linalg.norm(positions - positions.mean(0), axis=1))]
    cluster = positions[np.linalg.norm(positions - center, axis=1) < 6.0]
    initial = cluster - center + 43.447598 / 2
    # The central atom hops along z: keeps the 4-fold axis, breaks the others
    final = initial.copy()
    final[np.argmin(np.linalg.norm(initial - initial.mean(0), axis=1)), 2] += 1.0
    sym_matrix, sym_perm = unique_symmetries(initial, final, sym_thr=0.01)
    return pd.Series(
        {"initial_positions": initial, "sym_matrix": sym_matrix, "sym_perm": sym_perm}
    )


def test_event_has_non_trivial_symmetries(ref_event):
    assert len(ref_event["sym_matrix"]) > 1


def test_symmetries_map_initial_structure_onto_itself(ref_event):
    initial = ref_event["initial_positions"]
    for i in range(len(ref_event["sym_matrix"])):
        np.testing.assert_allclose(
            apply_event_symmetry(initial, ref_event, i), initial, atol=0.05
        )


def test_rotation_about_origin_shifts_by_box_vector(ref_event):
    """The previous behaviour: off by c - R c, a whole box vector here."""
    initial = ref_event["initial_positions"]
    shifts = [
        np.abs(
            geometry.transform_positions(
                initial, ref_event["sym_matrix"][i], 0, ref_event["sym_perm"][i]
            )
            - initial
        ).max()
        for i in range(1, len(ref_event["sym_matrix"]))
    ]
    assert max(shifts) > 40.0
