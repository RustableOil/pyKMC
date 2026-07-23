# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pyKMC is a Python framework for **adaptive off-lattice Kinetic Monte Carlo (aKMC)** simulations of atomistic systems. It couples to external, compiled scientific codes that have no pure-Python fallback:

- **LAMMPS** (compiled with `PKG_BASIC`, `PKG_EXTRA_COMPUTE`, `PKG_PLUGIN`) — energy/force evaluations and minimization.
- **pARTn** — a LAMMPS plugin providing saddle-point search (ARTn) for finding transition events.
- **IRA** — shape-matching library used for point-set registration (mapping searched events back onto the live system) and symmetry detection.

These are heavy, source-built dependencies (see `docs/install/`). In a bare environment (no LAMMPS/pARTn/IRA/mpi4py/pynauty installed), `import pykmc` itself fails, since `pykmc/__init__.py` eagerly imports the environments module, which imports `pynauty`. Don't assume `pytest`/imports will work without the full stack — check what's actually installed before treating an ImportError as a code bug.

## Commands

```bash
# Install (dev extras: ruff, ty, mkdocs stack)
pip install -e ".[dev]"

# Format (CI-enforced via .github/workflows/format.yml — this is the ONLY CI check, there is no test CI)
ruff format .
ruff format --check .      # what CI runs

# Lint (ruff.toml: select E4,E7,E9,F,B,Q,ANN,D)
ruff check .

# Type check
ty check

# Tests (pytest, no pytest.ini — defaults apply)
pytest tests/
pytest tests/test_geometry.py::test_name -v   # single test

# Docs (mkdocs + mkdocs-material, versioned with mike)
mkdocs serve
mkdocs build

# Run an actual simulation (requires MPI + a compiled LAMMPS/pARTn/IRA stack)
mpirun -n 8 python -m pykmc -in input.in
```

Tests under `tests/` mostly build synthetic ASE `bulk("Ni", ...)` systems (see `tests/conftest.py`) so they don't require a real LAMMPS/pARTn install. A few (`tests/engine/test_engine_lammps.py`, `tests/test_lammps_engine_api_mpi.py`) do exercise the real engine and need the full compiled stack. `tests/conftest.py` also defines an `mpi_test(nproc=2)` decorator that re-execs the current test file under `mpirun` when not already inside an MPI job — use it as a reference when adding MPI-dependent tests.

## Architecture

### Entry point and main loop

`python -m pykmc -in input.in` → `pykmc/run.py:main()`:
1. `Config.from_ini_file(path)` parses the INI input file into Pydantic models (`pykmc/config.py` — one `BaseModel` per INI section: `ControlConfig`, `AtomicEnvironmentConfig`, `EventSearchConfig`, etc.).
2. `ManagerFactory(...).launch()` splits MPI ranks into engine sessions and returns a `Manager` (only on rank 0 — every other rank blocks inside `engine.start()` as an engine worker, see below).
3. `KMC(config)` is built and `.run()` executes the simulation loop (`pykmc/kmc.py`).

The `KMC.run()` loop (per step): find atomic environments not yet explored → search new generic events on a sample of atoms with each unexplored environment (`EventSearch`, delegates saddle-point search to the engine/pARTn) → validate and merge into the persistent `ReferenceEventTable` (generic, transferable events keyed by environment) → refine the subset of reference events applicable to the *current* system into concrete candidates (`Refinement` → `ActiveEventTable`) → pick one via rejection-free KMC (`algorithms.rejection_free`, optionally steered by `Bias`) → reconstruct the chosen event's saddle/final geometry onto the live atom indices via point-set registration (IRA, `point_set_registration.py`/`reconstruction.py`) → if the reconstructed move looks like a trapped "basin" (`basins/detection.py:DetectorThreshold`), hand off to `basins/` to accelerate escape → update positions, log, snapshot trajectory, loop.

Domain-expected failures (failed reconstruction, invalid event, etc.) are propagated as `Result`/`Ok`/`Err` values (`pykmc/result.py`), not exceptions — exceptions are reserved for programming errors.

### MPI master-worker engine layer (`pykmc/enginemanager/`)

`ManagerFactory` (`enginemanager/lmpi/pool/factory.py`) splits `MPI.COMM_WORLD` into N "sessions," each owning a chunk of ranks that runs an `MpiApiEngine` (a LAMMPS instance wrapped for the master-worker protocol). Rank 0 stays outside every chunk and drives everything via a `Manager` (`enginemanager/lmpi/pool/manager.py`) holding one `MpiApiSession` per chunk plus a `global_session` spanning all engine ranks. Communication is abstracted behind `MpiMessenger`/`QueueMessenger` (`enginemanager/messenger.py`) — `QueueMessenger` is used when rank 0 is inside a chunk (in-process), `MpiMessenger` otherwise. `Manager.use_local()`/`use_global()` switch whether subsequent calls target per-session engines or the single global one.

### Registrable / facade-strategy pattern (`pykmc/_core/`)

Nearly every pluggable piece of pyKMC (engines, atomic-environment strategies, etc.) is built on `Registrable` (`_core/registrable.py`): a root ABC declared with `class Xxx(Registrable, root=True)` gets its own `_registry: dict[name, class]`; concrete subclasses declare a class attribute `name = "..."` and are auto-registered at class-definition time via `__init_subclass__`. `_core/discovery.py:autodiscover(package_name, package_path)` imports every submodule of a package so its `Registrable` subclasses register themselves — this is how new strategies/engines become available without touching a central switch statement. Import failures during autodiscovery are swallowed and only raised lazily if that specific name is requested via `.create(name, **kwargs)`.

Two flavors of this pattern:
- **Engines** (`pykmc/engine/base.py`, `pykmc/engine/lammps.py`): a fixed computational backend, not one of several equivalent choices. `Engine` also supports `EngineExtension` — attach extra methods to an engine instance post-hoc (`super().__init__(engine)` in the extension registers it and its public methods become directly callable on the engine via `__getattr__` delegation). Data-extraction engine methods return real values only on MPI rank 0 of the engine's communicator; other ranks get `None` — always account for this when calling into an engine directly.
- **Strategies** (facade/strategy split, documented in `docs/dev_strategy_pattern.md`): a user-facing facade class holds data and delegates to a swappable `XxxStrategy(Registrable, root=True)` implementation, each strategy declaring its config needs as a `typing.Protocol` rather than depending on a concrete config class (so tests can pass stubs/mocks instead of real `Config` objects — see `tests/conftest.py:mock_config`). Adding a new strategy/engine is just: new file in the right package with a `name = "..."` class — no registration boilerplate.

Full details: `docs/dev_strategy_pattern.md` and `docs/dev_engine.md`.

### Config (`pykmc/config.py`)

Input files are classic INI (`configparser`), one section per Pydantic `BaseModel` (`ControlConfig`, `AtomicEnvironmentConfig`, `EventSearchConfig`, `BasinConfig`, `EventRecyclingConfig`, `BiasConfig`, `PsrConfig`, `IraConfig`, ...), assembled into a single `Config` via `Config.from_ini_file(path)`. `tests/data/input.in` is a good reference for the file format; `docs/parameters.md`/`docs/parameters_details.md` (generated by `scripts/generate_parameters_doc.py`) document individual fields.

### Other notable modules

- `pykmc/event_table.py` — `ReferenceEventTable` (generic, environment-keyed events) and `ActiveEventTable` (concrete per-atom candidate events for the current step); both backed by `pandas.DataFrame`.
- `pykmc/event_recycling.py` — optional carry-over of unperturbed active-table rows between KMC steps (`Recycling`/`DistanceRecycling`), enabled by `config.control.recycle`.
- `pykmc/basins/` — superbasin detection/exploration/exit-rate solving for systems trapped oscillating between a small set of states.
- `pykmc/symmetries.py` + `pykmc/environments/graph_nauty.py` — symmetry-equivalence checks for atomic environments/events, via `pynauty` graph automorphism and IRA.
- `pykmc/bias.py` — pluggable event-selection bias (`DirectionBias`, `PointBias`, `TopoBias`) layered on top of the rejection-free algorithm.
