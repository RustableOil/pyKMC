"""Cache of the HTST prefactor nu0 of refined active events.

A refined active event gets its own nu0 from two Hessians (minimum and saddle).
A KMC run keeps meeting the same few events (the same defect hop in the same
local environment), and each new meeting would cost the same two Hessians again.
The cache keeps the nu0 of each refined event and reuses it for a later refined
event that is the same physical event:

- the same reference event, identified by its environment IDs (`event_id`,
  `id_saddle`, `id_final`), which stay valid if the table is re-indexed or
  loaded in another run;
- a refined barrier within `de_thr` (eV) of the cached one;
- the same displacement profile from minimum to saddle: the `n_profile` largest
  per-atom displacements, sorted, each within `dr_thr` (Angstrom) of the cached
  ones. This is invariant to translation, atom permutation and the symmetry
  operations that map one image of the event onto another, which leave nu0
  unchanged.

Only nu0 with status 'ok' are cached; a failed Hessian is computed again the
next time. The cache is written as a pickle with the Hessian region settings and
the potential it was computed with; a file written with other settings is
ignored.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..utils.geometry import per_atom_displacement

CACHE_FORMAT = 1

EventKey = tuple[str, str, str]


@dataclass
class Nu0Entry:
    """One cached nu0."""

    energy_barrier: float
    profile: np.ndarray
    nu0: float
    n_reused: int = 0


@dataclass
class Nu0Cache:
    """nu0 of refined events, reused for later matches of the same event.

    Parameters
    ----------
    de_thr : float
        Largest difference (eV) between refined barriers of a match.
    dr_thr : float
        Largest difference (Angstrom) between matching entries of the
        displacement profiles.
    n_profile : int
        Number of largest per-atom displacements in a profile.
    signature : str
        The Hessian region settings and the potential; a file with another
        signature is not loaded.

    """

    de_thr: float
    dr_thr: float
    n_profile: int = 16
    signature: str = ""
    entries: dict[EventKey, list[Nu0Entry]] = field(default_factory=dict)

    def profile(
        self, minimum: np.ndarray, saddle: np.ndarray, cell: np.ndarray
    ) -> np.ndarray:
        """Return the `n_profile` largest per-atom displacements, largest first.

        Parameters
        ----------
        minimum, saddle : np.ndarray
            Shape (N, 3) positions of the whole system at the minimum and saddle.
        cell : np.ndarray
            3x3 cell (orthorhombic, minimum image).

        Returns
        -------
        np.ndarray
            Shape (n_profile,), zero-padded if the system has fewer atoms.

        """
        disp = per_atom_displacement(
            np.asarray(minimum, dtype=float), np.asarray(saddle, dtype=float), cell
        )
        top = np.sort(disp)[::-1][: self.n_profile]
        out = np.zeros(self.n_profile)
        out[: len(top)] = top
        return out

    def _match(
        self, key: EventKey, energy_barrier: float, profile: np.ndarray
    ) -> Optional[Nu0Entry]:
        for entry in self.entries.get(key, []):
            if abs(entry.energy_barrier - energy_barrier) > self.de_thr:
                continue
            if np.max(np.abs(entry.profile - profile)) > self.dr_thr:
                continue
            return entry
        return None

    def lookup(
        self, key: EventKey, energy_barrier: float, profile: np.ndarray
    ) -> Optional[float]:
        """Return the cached nu0 (THz) of a matching event, or None."""
        entry = self._match(key, energy_barrier, profile)
        if entry is None:
            return None
        entry.n_reused += 1
        return entry.nu0

    def add(
        self, key: EventKey, energy_barrier: float, profile: np.ndarray, nu0: float
    ) -> None:
        """Cache the nu0 of an event, unless a matching event is cached already."""
        if self._match(key, energy_barrier, profile) is not None:
            return
        self.entries.setdefault(key, []).append(
            Nu0Entry(float(energy_barrier), np.asarray(profile, dtype=float), float(nu0))
        )

    def __len__(self) -> int:
        return sum(len(v) for v in self.entries.values())

    def save(self, path: str) -> None:
        """Write the cache atomically (a crash mid-write keeps the old file)."""
        payload = {
            "format": CACHE_FORMAT,
            "signature": self.signature,
            "n_profile": self.n_profile,
            "entries": {
                key: [
                    (e.energy_barrier, e.profile, e.nu0, e.n_reused) for e in entries
                ]
                for key, entries in self.entries.items()
            },
        }
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as file:
            pickle.dump(payload, file)
        os.replace(tmp, path)

    def load(self, path: str) -> tuple[bool, str]:
        """Read the entries of a cache file into this cache.

        Returns
        -------
        tuple[bool, str]
            Whether entries were loaded, and why not otherwise.

        """
        if not os.path.exists(path):
            return False, f"no file {path}"
        try:
            with open(path, "rb") as file:
                payload = pickle.load(file)
        except Exception as exc:  # noqa: BLE001 - a bad cache is only skipped
            return False, f"{path} cannot be read ({exc})"
        if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
            return False, f"{path} has an unknown format"
        if payload.get("signature") != self.signature:
            return False, (
                f"{path} was computed with other settings "
                f"('{payload.get('signature')}', this run '{self.signature}')"
            )
        if payload.get("n_profile") != self.n_profile:
            return False, f"{path} has another profile length"
        for key, entries in payload["entries"].items():
            for energy_barrier, profile, nu0, n_reused in entries:
                self.entries.setdefault(tuple(key), []).append(
                    Nu0Entry(energy_barrier, np.asarray(profile), nu0, n_reused)
                )
        return True, ""
