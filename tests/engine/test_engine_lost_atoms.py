"""Setting positions far from the current ones must not lose atoms.

scatter_atoms only overwrites coordinates: LAMMPS wraps an atom by at most one
box length and hands it to a neighboring subdomain only. Run serially, and with
mpirun -n N (N >= 3) for the multi-subdomain case:

    mpirun -n 4 python -m pytest tests/engine/test_engine_lost_atoms.py
"""

from dataclasses import dataclass

import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("lammps")

from mpi4py import MPI  # noqa: E402

from pykmc.engine import Engine  # noqa: E402


@dataclass
class LammpsConfig:
    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


@pytest.fixture
def ni_column():
    """FCC Ni elongated along z, so subdomains are thin slabs in z."""
    return bulk("Ni", "fcc", a=3.524, cubic=True).repeat([3, 3, 12])


def make_engine(atoms, comm, processors=None):
    engine = Engine.create("lammps", config=LammpsConfig(), comm=comm)
    engine.start()
    engine.initialize_parameters()
    if processors is not None:
        engine.command(processors)
    engine.initialize_system(
        types=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )
    engine.initialize_potential()
    return engine


def test_positions_beyond_one_box_length(ni_column):
    """Serial: coordinates more than a box length outside are wrapped, not lost."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial test: run without mpirun")
    engine = make_engine(ni_column, comm=None)
    e_ref = engine.get_total_energy()
    shifted = ni_column.get_positions() + 2.0 * np.diag(ni_column.get_cell())

    e_shifted = engine.get_total_energy(positions=shifted)

    assert e_shifted == pytest.approx(e_ref, rel=1e-10)
    assert engine.lmp.get_natoms() == len(ni_column)
    engine.close()


@pytest.mark.mpi
def test_rigid_shift_across_subdomains(ni_column):
    """MPI: a half-box shift along z moves every atom past its neighbor slab."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 3:
        pytest.skip("requires mpirun -n N with N >= 3")
    engine = make_engine(
        ni_column, comm, processors=f"processors 1 1 {comm.Get_size()}"
    )
    e_ref = engine.get_total_energy()
    shifted = ni_column.get_positions() + [0.0, 0.0, 0.5 * ni_column.cell[2, 2]]

    e_shifted = engine.get_total_energy(positions=shifted)
    positions = engine.get_positions()

    if comm.Get_rank() == 0:
        assert e_shifted == pytest.approx(e_ref, rel=1e-10)
        assert len(positions) == len(ni_column)
    engine.close()
    comm.Barrier()
