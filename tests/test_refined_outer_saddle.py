"""The reconstruction of a refined event starts from its refined saddle for the
atoms outside the rcut environment, not from the initial minimum."""

import concurrent.futures
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc import System, kmc as kmc_module
from pykmc.config import RateConstantConfig
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.rate_constant import RateConstant
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput, Ok

L = 20.0
N = 10
NEIGHBORS = [0, 1, 2]


def make_system():
    rng = np.random.default_rng(0)
    return System(
        types=["Si"] * N,
        positions=rng.uniform(1.0, L - 1.0, (N, 3)),
        cell=np.diag([L, L, L]),
        pbc=np.array([True, True, True]),
    )


def make_refinement(system):
    refinement = Refinement.__new__(Refinement)
    refinement.system = system
    return refinement


def make_table(rows=()):
    table = ActiveEventTable(
        Mock(),
        rate_constant=RateConstant.create(
            "constant", config=RateConstantConfig(style="constant")
        ),
    )
    for event in rows:
        table.add_events(event)
    return table


def test_outer_saddle_keeps_moved_atoms_outside_the_environment():
    system = make_system()
    saddle = system.positions.copy()
    saddle[NEIGHBORS] += 0.5  # environment, excluded
    saddle[4] += [0.2, 0.0, 0.0]  # moved outer atom
    saddle[6] += [1e-4, 0.0, 0.0]  # numerical noise, excluded
    saddle[7, 0] = system.positions[7, 0] + 0.1 - L  # moved across the boundary

    indices, positions = make_refinement(system).outer_saddle(
        system.positions, saddle, NEIGHBORS
    )

    np.testing.assert_array_equal(indices, [4, 7])
    np.testing.assert_allclose(positions, saddle[[4, 7]])


def test_outer_saddle_none_when_only_the_environment_moved():
    system = make_system()
    saddle = system.positions.copy()
    saddle[NEIGHBORS] += 0.5

    assert (
        make_refinement(system).outer_saddle(system.positions, saddle, NEIGHBORS)
        is None
    )


def make_kmc(system, seen):
    kmc = KMC.__new__(KMC)
    kmc.system = system
    kmc.config = SimpleNamespace(psr=SimpleNamespace(matching_score_thr=0.1))
    kmc.manager = Mock()
    kmc.neighbors_list = Mock()
    kmc.neighbors_list.get_neighbors.return_value = NEIGHBORS

    class FakeReconstruction:
        def __init__(self, *args, **kwargs):
            pass

        def reconstruct(self, min1, min2, saddle_positions, *args, **kwargs):
            seen.append(saddle_positions.copy())
            return Ok("reconstructed")

    return kmc, FakeReconstruction


def test_reconstruction_starts_from_the_outer_saddle(monkeypatch):
    system = make_system()
    initial = system.positions.copy()
    env_saddle = initial[NEIGHBORS] + 0.3
    outer_idx = np.array([5, 8])
    outer_positions = initial[outer_idx] + 0.2
    event = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=env_saddle,
        E_saddle=0.5,
        min2_positions=initial[NEIGHBORS] + 0.6,
        dE_forward=0.5,
        num_reference_event=0,
        refined="T",
        outer_saddle=(outer_idx, outer_positions),
    )
    seen = []
    kmc, fake = make_kmc(system, seen)
    monkeypatch.setattr(kmc_module, "Reconstruction", fake)

    result = kmc._reconstruction_active_event(0, make_table([event]))

    assert result.ok_value() == "reconstructed"
    start = seen[0]
    np.testing.assert_allclose(start[NEIGHBORS], env_saddle)
    np.testing.assert_allclose(start[outer_idx], outer_positions)
    untouched = [i for i in range(N) if i not in NEIGHBORS and i not in outer_idx]
    np.testing.assert_allclose(start[untouched], initial[untouched])
    # the system is back at the initial minimum afterwards
    np.testing.assert_allclose(system.positions, initial)


def test_reconstruction_without_outer_saddle(monkeypatch):
    system = make_system()
    initial = system.positions.copy()
    event = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=initial[NEIGHBORS] + 0.3,
        E_saddle=0.5,
        min2_positions=initial[NEIGHBORS] + 0.6,
        dE_forward=0.5,
        num_reference_event=0,
        refined="F",
    )
    seen = []
    kmc, fake = make_kmc(system, seen)
    monkeypatch.setattr(kmc_module, "Reconstruction", fake)

    kmc._reconstruction_active_event(0, make_table([event]))

    outside = [i for i in range(N) if i not in NEIGHBORS]
    np.testing.assert_allclose(seen[0][outside], initial[outside])


def test_recycled_rows_drop_the_outer_saddle():
    outer = (np.array([5]), np.zeros((1, 3)))
    events = [
        EventRefinementOutput(
            central_atom_index=i,
            saddle_positions=np.zeros((3, 3)),
            E_saddle=0.5,
            min2_positions=np.zeros((3, 3)),
            dE_forward=0.5,
            num_reference_event=0,
            refined="T",
            outer_saddle=outer,
        )
        for i in range(3)
    ]
    table = make_table(events)
    table.recycler = Mock()
    table.recycler.select_recyclable.side_effect = lambda t, *args: t.table.iloc[
        1:
    ].reset_index(drop=True)

    table.prune_for_recycling(0, Mock(), None)

    assert len(table.table) == 2
    assert table.table["outer_saddle"].isna().all()


def run_refinement(system, full_saddle, refined, keep=True):
    """Run Refinement.execute on one stubbed AV refinement returning full_saddle."""
    refinement = make_refinement(system)
    refinement.config = SimpleNamespace(
        reconstruction=SimpleNamespace(refined_outer_saddle=keep),
        control=SimpleNamespace(active_volume=True),
        eventsearch=SimpleNamespace(refined_energy_thr=0.05),
    )
    refinement.loggers = Mock()
    refinement.rate_constant = SimpleNamespace(per_event_prefactor=False)
    refinement.atomic_environment = Mock()
    refinement.atomic_environment.get_atoms_with_id.return_value = [0]
    refinement.get_total_refinements_todo = lambda df: (1, 1.0)
    refinement.get_energy_thr_refine = lambda df, ktot: 10.0

    def refine_single(at_idx, dfevent, total_energy, future_context, *args):
        f = concurrent.futures.Future()
        f.set_result(
            Ok(
                EventRefinementOutput(
                    central_atom_index=at_idx,
                    saddle_positions=full_saddle.copy(),
                    E_saddle=0.5,
                    refined=refined,
                )
            )
        )
        future_context[f] = {
            "min2_positions": np.zeros((3, 3)),
            "num_reference_event": 0,
            "reference_energy_barrier": 0.5,
            "neighbors": list(NEIGHBORS),
        }
        return f

    refinement.refine_single = refine_single
    df = pd.DataFrame([{"event_id": "a", "idx_ref": 0}])
    refinement.execute(df, total_energy=0.0)
    return refinement.results[0].ok_value()


def test_execute_stores_the_outer_saddle():
    system = make_system()
    saddle = system.positions.copy()
    saddle[NEIGHBORS] += 0.3
    saddle[5] += 0.2

    output = run_refinement(system, saddle, "T")

    np.testing.assert_allclose(output.saddle_positions, saddle[NEIGHBORS])
    np.testing.assert_array_equal(output.outer_saddle[0], [5])
    np.testing.assert_allclose(output.outer_saddle[1], saddle[[5]])


def test_execute_without_outer_saddle_when_disabled_or_not_refined():
    system = make_system()
    saddle = system.positions.copy()
    saddle[5] += 0.2

    assert run_refinement(system, saddle, "T", keep=False).outer_saddle is None
    assert run_refinement(system, saddle, "F").outer_saddle is None
