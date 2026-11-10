"""Tests for the nu0 cache of refined active events ([RateConstant] nu0_cache)."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pykmc.config import RateConstantConfig
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.rate_constant import PrefactorOutcome, RateConstant
from pykmc.rate_constant.nu0_cache import Nu0Cache
from pykmc.result import EventRefinementOutput

CELL = np.diag([20.0, 20.0, 20.0])
KEY = ("env-a", "sad-a", "fin-a")


def _minimum(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 20.0, size=(n, 3))


def _saddle(minimum, hop=(1.2, 0.0, 0.0), atom=3, shift=(0.0, 0.0, 0.0)):
    saddle = minimum + np.asarray(shift)  # a rigid shift leaves the profile alone
    saddle[atom] += np.asarray(hop)
    saddle[atom + 1] += 0.3 * np.asarray(hop)
    return saddle % 20.0


def _cache(**kw):
    return Nu0Cache(de_thr=kw.get("de_thr", 0.01), dr_thr=kw.get("dr_thr", 0.05))


# --- Nu0Cache ----------------------------------------------------------------


def test_profile_is_sorted_largest_displacements_and_pbc_aware():
    cache = _cache()
    minimum = _minimum()
    minimum[3] = [19.9, 5.0, 5.0]
    saddle = minimum.copy()
    saddle[3] = [0.3, 5.0, 5.0]  # 0.4 A across the boundary
    saddle[7] += [0.0, 0.1, 0.0]
    profile = cache.profile(minimum, saddle, CELL)
    assert profile.shape == (16,)
    np.testing.assert_allclose(profile[:2], [0.4, 0.1], atol=1e-9)
    assert np.all(profile[2:] == 0.0)


def test_lookup_matches_same_event_only():
    cache = _cache()
    minimum = _minimum()
    profile = cache.profile(minimum, _saddle(minimum), CELL)
    cache.add(KEY, 0.405, profile, 21.5)

    # same event, other image (permuted atoms), barrier within de_thr
    other = _minimum(seed=1)
    profile_other = cache.profile(other, _saddle(other, hop=(0, -1.2, 0), atom=9), CELL)
    assert cache.lookup(KEY, 0.409, profile_other) == 21.5
    # barrier too far
    assert cache.lookup(KEY, 0.42, profile_other) is None
    # other reference event
    assert cache.lookup(("env-b", "sad-a", "fin-a"), 0.405, profile) is None
    # other displacement profile
    profile_long = cache.profile(minimum, _saddle(minimum, hop=(1.5, 0, 0)), CELL)
    assert cache.lookup(KEY, 0.405, profile_long) is None


def test_add_keeps_first_of_matching_events():
    cache = _cache()
    profile = cache.profile(_minimum(), _saddle(_minimum()), CELL)
    cache.add(KEY, 0.405, profile, 21.5)
    cache.add(KEY, 0.406, profile, 99.0)
    assert len(cache) == 1
    assert cache.lookup(KEY, 0.405, profile) == 21.5


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "nu0.pickle")
    cache = Nu0Cache(de_thr=0.01, dr_thr=0.05, signature="av rmov=15; pot A")
    profile = cache.profile(_minimum(), _saddle(_minimum()), CELL)
    cache.add(KEY, 0.405, profile, 21.5)
    cache.save(path)

    again = Nu0Cache(de_thr=0.01, dr_thr=0.05, signature="av rmov=15; pot A")
    assert again.load(path) == (True, "")
    assert again.lookup(KEY, 0.405, profile) == 21.5

    other = Nu0Cache(de_thr=0.01, dr_thr=0.05, signature="av rmov=10; pot A")
    loaded, why = other.load(path)
    assert not loaded and "other settings" in why
    assert len(other) == 0


def test_load_missing_or_corrupt_file(tmp_path):
    cache = _cache()
    assert cache.load(str(tmp_path / "none.pickle"))[0] is False
    bad = tmp_path / "bad.pickle"
    bad.write_bytes(b"not a pickle")
    assert cache.load(str(bad))[0] is False
    with open(tmp_path / "old.pickle", "wb") as file:
        pickle.dump({"format": 0}, file)
    assert cache.load(str(tmp_path / "old.pickle"))[0] is False


# --- KMC.add_active_prefactors with the cache --------------------------------


class _Loggers:
    def __init__(self):
        self.lines = []

    def info(self, name, message):
        self.lines.append(message)

    warning = error = info


class _RateConstant:
    """htst rates, with a compute_prefactors that records what it is asked."""

    per_event_prefactor = True

    def __init__(self, nu0s):
        self._rc = RateConstant.create(
            "htst", config=RateConstantConfig(style="htst", k0=10.0, T=500.0)
        )
        self._nu0s = list(nu0s)
        self.calls = []

    def compute_rate(self, dE, prefactor=None):
        return self._rc.compute_rate(dE, prefactor=prefactor)

    def compute_prefactors(self, requests, system, frozen_ids):
        self.calls.append(len(requests))
        return [[PrefactorOutcome(self._nu0s.pop(0), "ok")] for _ in requests]


def _kmc(rate_constant, cache):
    kmc = KMC.__new__(KMC)
    kmc.rate_constant = rate_constant
    kmc.loggers = _Loggers()
    kmc.frozen_ae = None
    kmc.nu0_cache = cache
    kmc._nu0_pending, kmc._nu0_followers = {}, {}
    kmc._ref_nu0_tried = set()
    minimum = _minimum()
    kmc.system = SimpleNamespace(positions=minimum, cell=CELL)
    kmc.reference_table = SimpleNamespace(
        table=pd.DataFrame(
            {
                "idx_ref": [5, 1],
                "event_id": ["env-a", "env-b"],
                "id_saddle": ["sad-a", "sad-b"],
                "id_final": ["fin-a", "fin-b"],
            }
        )
    )
    kmc.active_table = ActiveEventTable(
        SimpleNamespace(), rate_constant=rate_constant
    )
    return kmc


def _refined(kmc, atom, ref, dE, hop):
    full = _saddle(kmc.system.positions, hop=hop, atom=atom)
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=full[:2],
        E_saddle=dE,
        min2_positions=None,
        dE_forward=dE,
        num_reference_event=ref,
        refined="T",
        prefactor_status="pending",
        full_saddle=full,
    )


def _step(kmc, events):
    kmc.active_table.table = kmc.active_table.table.iloc[0:0]
    kmc.active_table.add_events(events)
    kmc.add_active_prefactors()
    return kmc.active_table.table


def test_kmc_reuses_cached_nu0_across_steps(tmp_path):
    rc = _RateConstant([21.5, 13.0, 99.0])
    kmc = _kmc(rc, _cache())

    # step 1: two images of ref 5 (one Hessian pair) and one ref 1 event
    table = _step(
        kmc,
        [
            _refined(kmc, 3, 5, 0.4053, (1.2, 0, 0)),
            _refined(kmc, 11, 5, 0.4055, (0, 1.2, 0)),
            _refined(kmc, 20, 1, 0.5453, (0.9, 0.9, 0)),
        ],
    )
    assert rc.calls == [2]
    np.testing.assert_allclose(table["prefactor"], [21.5, 21.5, 13.0])
    assert list(table["prefactor_status"]) == ["ok"] * 3
    assert all(v is None for v in table["full_saddle"])
    expected_k = [rc.compute_rate(dE, nu0) for dE, nu0 in
                  zip(table["energy_barrier"], table["prefactor"])]
    np.testing.assert_allclose(table["k"], expected_k)

    # step 2: the same events elsewhere: no Hessian at all
    table = _step(
        kmc,
        [
            _refined(kmc, 30, 5, 0.4059, (0, 0, -1.2)),
            _refined(kmc, 8, 1, 0.5456, (0, 0.9, 0.9)),
        ],
    )
    assert rc.calls == [2]  # compute_prefactors not called again
    np.testing.assert_allclose(table["prefactor"], [21.5, 13.0])

    # step 3: a different refined saddle of ref 5 is computed
    table = _step(kmc, [_refined(kmc, 3, 5, 0.47, (1.6, 0, 0))])
    assert rc.calls == [2, 1]
    np.testing.assert_allclose(table["prefactor"], [99.0])
    assert len(kmc.nu0_cache) == 3


def test_failed_nu0_is_not_cached_and_followers_share_it():
    class _Failing(_RateConstant):
        def compute_prefactors(self, requests, system, frozen_ids):
            self.calls.append(len(requests))
            return [[PrefactorOutcome(None, "hessian_failed")] for _ in requests]

    rc = _Failing([])
    kmc = _kmc(rc, _cache())
    table = _step(
        kmc,
        [
            _refined(kmc, 3, 5, 0.4053, (1.2, 0, 0)),
            _refined(kmc, 11, 5, 0.4053, (0, 1.2, 0)),
        ],
    )
    assert rc.calls == [1]
    assert list(table["prefactor_status"]) == ["hessian_failed"] * 2
    assert table["prefactor"].isna().all()
    assert len(kmc.nu0_cache) == 0


# --- KMC.fill_reference_prefactors -------------------------------------------

import concurrent.futures  # noqa: E402

from pykmc import kmc as kmc_module  # noqa: E402
from pykmc._core.result import Err, ErrorInfo, ErrorType, Ok  # noqa: E402
from pykmc.event_table import ReferenceEventTable  # noqa: E402


class _RefRateConstant(_RateConstant):
    settings_record = "av rmov=15 ract=21"

    def compute_prefactors(self, requests, system, frozen_ids):
        self.calls.append([r.names[0] for r in requests])
        assert all(r.reference for r in requests)
        return [[PrefactorOutcome(self._nu0s.pop(0), "ok")] for _ in requests]


def _fill_kmc(monkeypatch, refined_barriers, cache=None):
    """KMC with refs 5 (env-a, present), 1 (env-b, present), 7 (env-c, absent)."""
    rc = _RefRateConstant([21.5, 13.0])
    kmc = _kmc(rc, cache)
    kmc.config = SimpleNamespace(
        control=SimpleNamespace(active_volume=True),
        eventsearch=SimpleNamespace(refined_energy_thr=0.05),
    )
    kmc.total_energy = -100.0
    kmc.neighbors_list = kmc.manager = None
    ref = ReferenceEventTable.__new__(ReferenceEventTable)
    ref.rate_constant = rc
    ref.table = pd.DataFrame(
        {
            "idx_ref": [5, 1, 7],
            "event_id": ["env-a", "env-b", "env-c"],
            "id_saddle": ["sad-a", "sad-b", "sad-c"],
            "id_final": ["fin-a", "fin-b", "fin-c"],
            "energy_barrier": [0.405, 0.545, 0.9],
            "k": [0.0, 0.0, 0.0],
            "sym_matrix": [[np.eye(3), -np.eye(3)]] * 3,
            "sym_perm": [[np.arange(3), np.arange(3)]] * 3,
            "prefactor": [np.nan] * 3,
            "prefactor_status": ["none"] * 3,
            "prefactor_settings": ["unknown"] * 3,
        }
    )
    kmc.reference_table = ref
    atoms = {"env-a": [3, 11], "env-b": [20]}
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a"] * 2 + ["env-b"] + ["crystal"] * 5,
        get_atoms_with_id=lambda env: atoms.get(env, []),
    )
    refined = []

    class FakeRefinement:
        def __init__(self, *args):
            pass

        def refine_single(self, at_idx, dfevent, total_energy, ctx, e_thr, k_thr):
            assert e_thr == np.inf and k_thr is None
            assert len(dfevent["sym_matrix"]) == 1  # one image only
            refined.append((at_idx, int(dfevent["idx_ref"])))
            barrier = refined_barriers[int(dfevent["idx_ref"])]
            f = concurrent.futures.Future()
            if barrier is None:
                f.set_result(Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="no saddle")))
            else:
                full = _saddle(kmc.system.positions, atom=at_idx)
                f.set_result(Ok(EventRefinementOutput(
                    central_atom_index=at_idx, saddle_positions=full,
                    E_saddle=barrier, refined="T")))
            return [f]

    monkeypatch.setattr(kmc_module, "Refinement", FakeRefinement)
    return kmc, rc, refined


def test_fill_reference_nu0_sets_present_rows_once(monkeypatch):
    kmc, rc, refined = _fill_kmc(monkeypatch, {5: 0.406, 1: 0.546}, _cache())
    kmc.fill_reference_prefactors()

    assert refined == [(3, 5), (20, 1)]  # first atom of each present env
    assert len(rc.calls) == 1 and len(rc.calls[0]) == 2
    t = kmc.reference_table.table.set_index("idx_ref")
    assert t.at[5, "prefactor"] == 21.5 and t.at[1, "prefactor"] == 13.0
    assert t.at[5, "prefactor_status"] == "ok"
    assert t.at[5, "prefactor_settings"] == "av rmov=15 ract=21"
    assert t.at[5, "k"] == rc.compute_rate(0.405, 21.5)
    assert np.isnan(t.at[7, "prefactor"])  # env-c absent: left for later
    # seeded into the cache, so this step's refined events reuse them
    assert len(kmc.nu0_cache) == 2

    kmc.fill_reference_prefactors()  # nothing left to do
    assert len(rc.calls) == 1


def test_fill_reference_nu0_rejects_other_saddle_and_failures(monkeypatch):
    kmc, rc, refined = _fill_kmc(monkeypatch, {5: 0.60, 1: None})
    kmc.fill_reference_prefactors()

    assert rc.calls == []  # no Hessians for a wrong or failed refinement
    t = kmc.reference_table.table.set_index("idx_ref")
    assert t.at[5, "prefactor_status"] == "refined_other_saddle"
    assert t.at[1, "prefactor_status"] == "refinement_failed"
    assert np.isnan(t.at[5, "prefactor"])
    kmc.fill_reference_prefactors()  # tried once per run
    assert len(refined) == 2
