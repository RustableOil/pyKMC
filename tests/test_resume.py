"""Checkpoints and automatic resume ([Control] checkpoint / resume)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.log import LogKMC
from pykmc.resume import (
    apply_resume,
    remove_stale_outputs,
    truncate_outputs,
    write_checkpoint,
)


def _config(tmp_path, **kw):
    control = dict(
        resume=True,
        restart_file=None,
        restart_output=str(tmp_path / "restart.npz"),
        done_file=str(tmp_path / "DONE"),
        n_steps=100,
        reference_table="./start_table.pickle",
        reference_table_output=str(tmp_path / "reference_table.pickle"),
        visited_environments=None,
        visited_environments_output=str(tmp_path / "visited_environments.pickle"),
        trajectory_output=str(tmp_path / "trajkmc.xyz"),
    )
    control.update(kw)
    return SimpleNamespace(control=SimpleNamespace(**control))


def test_checkpoint_cuts_interrupted_step(tmp_path):
    out = tmp_path / "pykmc.out"
    out.write_text("header\nstep 1\n")
    pos = np.random.default_rng(0).random((5, 3))
    path = str(tmp_path / "restart.npz")
    write_checkpoint(path, 1, 2.5e-9, 10.0, pos, [str(out), str(tmp_path / "missing")])
    # an interrupted step 2 appended a line
    with open(out, "a") as f:
        f.write("step 2 (interrupted)\n")
    with np.load(path) as info:
        info = {k: info[k] for k in info.files}
    assert int(info["last_step"]) == 1
    np.testing.assert_array_equal(info["positions"], pos)
    assert truncate_outputs(info) == [str(out)]
    assert out.read_text() == "header\nstep 1\n"
    assert not (tmp_path / "restart.npz.tmp").exists()


def test_resume_from_checkpoint(tmp_path):
    cfg = _config(tmp_path)
    write_checkpoint(cfg.control.restart_output, 40, 1e-9, 5.0, np.zeros((2, 3)), [])
    (tmp_path / "reference_table.pickle").write_bytes(b"x")
    assert apply_resume(cfg) is None
    c = cfg.control
    assert c.restart_file == c.restart_output
    assert c.n_steps == 60  # remaining of the total
    assert c.reference_table == c.reference_table_output
    assert c.visited_environments is None  # output not written yet: unchanged


def test_resume_finished(tmp_path):
    cfg = _config(tmp_path)
    write_checkpoint(cfg.control.restart_output, 100, 1e-9, 5.0, np.zeros((2, 3)), [])
    assert "already run" in apply_resume(cfg)
    (tmp_path / "DONE").write_text("n_steps completed at step 100\n")
    assert "nothing to do" in apply_resume(_config(tmp_path))


def test_fresh_start_removes_stale_outputs(tmp_path):
    cfg = _config(tmp_path)
    traj = tmp_path / "trajkmc.xyz"
    traj.write_text("old")
    assert apply_resume(cfg) is None
    assert cfg.control.restart_file is None
    assert str(traj) in remove_stale_outputs(cfg)
    assert not traj.exists()


def test_no_resume_leaves_config(tmp_path):
    cfg = _config(tmp_path, resume=False)
    write_checkpoint(cfg.control.restart_output, 40, 1e-9, 5.0, np.zeros((2, 3)), [])
    assert apply_resume(cfg) is None
    assert cfg.control.restart_file is None and cfg.control.n_steps == 100


@pytest.mark.parametrize("name,width,fmt", LogKMC.OUTPUT_COLUMNS)
def test_output_columns_stay_separable(name, width, fmt):
    # readers split the header on 2+ spaces: each name is padded by at least one
    assert width > len(name)
