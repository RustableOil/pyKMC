"""Checkpoints and automatic resume of a KMC run.

A checkpoint (``[Control] restart_output``, written when ``checkpoint = True``)
holds the last completed step, the simulated time, the wall time, the positions
and the sizes of the output files at that moment. It is written atomically at
the end of every step, after the reference table, the visited environments and
the trajectory, so a run killed at any point (e.g. at the wall-time limit of a
batch job) restarts from its last completed step: the output files are cut back
to their checkpointed sizes, which removes the lines of a step that was
interrupted.

With ``resume = True`` the same input can be run again and again (one batch job
after another): see :func:`apply_resume`.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

# Logger files appended at every step (see pykmc.log.LOGGING_CONFIG); pykmc.log
# is not cut, it keeps the history of every job.
STEP_OUTPUT_FILES = ("pykmc.out", "pykmc.events", "pykmc.info")


def output_files(config) -> list[str]:
    return [*STEP_OUTPUT_FILES, config.control.trajectory_output]


def atomic_pickle(obj, path: str) -> None:
    """Pickle to path through a temporary file (a crash mid-write keeps the old file)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def write_checkpoint(
    path: str,
    last_step: int,
    last_time: float,
    total_wall_time: float,
    positions: np.ndarray,
    files: list[str],
) -> None:
    """Write the checkpoint atomically."""
    names = [f for f in files if os.path.exists(f)]
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            last_step=last_step,
            last_time=last_time,
            total_wall_time=total_wall_time,
            positions=np.asarray(positions, float),
            file_names=np.array(names, dtype=str),
            file_sizes=np.array([os.path.getsize(n) for n in names], dtype=np.int64),
        )
    os.replace(tmp, path)


def truncate_outputs(restart_info) -> list[str]:
    """Cut the output files back to their checkpointed sizes; returns the files cut."""
    if "file_names" not in restart_info:
        return []
    cut = []
    for name, size in zip(restart_info["file_names"], restart_info["file_sizes"]):
        name, size = str(name), int(size)
        if os.path.exists(name) and os.path.getsize(name) > size:
            os.truncate(name, size)
            cut.append(name)
    return cut


def write_done(path: str, message: str) -> None:
    with open(path, "w") as f:
        f.write(message.rstrip("\n") + "\n")


def apply_resume(config) -> str | None:
    """Prepare ``config`` for an automatic resume (``[Control] resume = True``).

    Called on every rank before the engines start; only reads files, except for
    writing done_file / removing stale outputs, which only rank 0 should do
    (see :func:`pykmc.run.main`).

    Returns
    -------
    str | None
        A message if there is nothing left to run (the caller exits), else None.
    """
    c = config.control
    if not c.resume:
        return None
    if os.path.exists(c.done_file):
        return f"{c.done_file} exists: the run has finished, nothing to do."
    if c.restart_file is None and os.path.exists(c.restart_output):
        with np.load(c.restart_output) as info:
            last_step = int(info["last_step"])
        remaining = c.n_steps - last_step
        if remaining <= 0:
            return f"done: {last_step} of n_steps = {c.n_steps} steps already run."
        c.restart_file = c.restart_output
        c.n_steps = remaining
        c.reference_table = (
            c.reference_table_output
            if os.path.exists(c.reference_table_output)
            else c.reference_table
        )
        c.visited_environments = (
            c.visited_environments_output
            if os.path.exists(c.visited_environments_output)
            else c.visited_environments
        )
    return None


def remove_stale_outputs(config) -> list[str]:
    """Fresh start under resume = True: remove output files of an earlier attempt
    that died before its first checkpoint (they would be appended to)."""
    c = config.control
    if not c.resume or c.restart_file is not None:
        return []
    removed = []
    for name in output_files(config):
        if os.path.exists(name):
            os.remove(name)
            removed.append(name)
    return removed
