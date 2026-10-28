from pykmc.engine import Engine, EngineExtension
import inspect
import numpy as np
import pytest
from ase.geometry import find_mic


class EngineContractTests:
    """
    Test suite defining the contract that every Engine implementation
    must satisfy. Do not instantiate directly.

    Subclasses must implement `make_engine()`, which returns a configured
    but not yet started engine instance.
    """

    def make_engine(self) -> Engine:
        raise NotImplementedError

    @property
    def is_rank0(self) -> bool:
        """True if the current rank should validate assertions."""
        return True

    def test_start_does_not_raise(self):
        """Test open and close engine."""
        engine = self.make_engine()
        engine.start()
        engine.close()

    def initialize(self, engine):
        """Initialization parameters and system convenience method."""
        engine.initialize_parameters()
        engine.initialize_system(
            types=self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        engine.initialize_potential()

    def test_initialize(self):
        """Test initialization parameters and system."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        engine.close()

    def test_set_get_positions(self):
        """set_positions() followed by get_positions() returns the same positions,
        up to periodic images (engines may wrap atoms into the cell)."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        positions = self.system.positions.copy()
        positions[0, 0] += 0.2
        engine.set_positions(positions)
        result = engine.get_positions()
        if self.is_rank0:
            difference, _ = find_mic(
                result - positions, self.system.cell, self.system.pbc
            )
            np.testing.assert_allclose(difference, 0.0, atol=1e-10)
        else:
            assert result is None
        engine.close()

    def test_get_potential_energy(self):
        """get_potential_energy() returns a float."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        pe = engine.get_potential_energy()
        if self.is_rank0:
            assert isinstance(pe, float)
        else:
            assert pe is None
        engine.close()

    def test_get_total_energy(self):
        """get_total_energy() returns a float."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        tot_e = engine.get_total_energy()
        if self.is_rank0:
            assert isinstance(tot_e, float)
        else:
            assert tot_e is None
        engine.close()

    def test_minimize(self):
        """minimize() reduces the energy of a perturbed configuration."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        rng = np.random.default_rng(seed=0)
        perturbed = self.system.positions.copy() + rng.uniform(
            -0.05, 0.05, size=self.system.positions.shape
        )
        e_before = engine.get_potential_energy(positions=perturbed)
        engine.minimize()
        e_after = engine.get_potential_energy()
        if self.is_rank0:
            assert e_after < e_before
        engine.close()

    def test_minimize_with_results(self):
        """minimize_with_results() reduces the energy and returns positions + energy."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        rng = np.random.default_rng(seed=42)
        perturbed = self.system.positions.copy() + rng.uniform(
            -0.1, 0.1, size=self.system.positions.shape
        )
        e_perturbed = engine.get_potential_energy(positions=perturbed)
        result = engine.minimize_with_results(positions=perturbed)
        if self.is_rank0:
            min_positions, e_min = result
            assert min_positions.shape == self.system.positions.shape
            assert e_min < e_perturbed
        else:
            assert result is None
        engine.close()

    def make_test_extension(self, engine) -> EngineExtension:
        """Return a concrete extension compatible with this engine."""
        raise NotImplementedError

    def make_conflicting_extension(self, engine) -> EngineExtension:
        """Return an extension whose methods conflict with those from make_test_extension()."""
        raise NotImplementedError

    def test_extension_registers_and_delegates(self):
        """A registered extension exposes its public callable methods through the engine."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        ext = self.make_test_extension(engine)
        public_callables = [
            m
            for m in dir(ext)
            if not m.startswith("_") and callable(getattr(ext, m, None))
        ]
        assert all(hasattr(engine, m) for m in public_callables)
        engine.close()

    def test_extension_methods_visible_in_dir(self):
        """Extension callable methods appear in dir(engine) after registration."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        ext = self.make_test_extension(engine)
        public_callables = [
            m
            for m in dir(ext)
            if not m.startswith("_") and callable(getattr(ext, m, None))
        ]
        engine_dir = dir(engine)
        assert all(m in engine_dir for m in public_callables)
        engine.close()

    def test_extension_with_property_registers_without_crash(self):
        """Registration must not execute @property getters — subclass fields may not exist yet."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)

        class _ExtWithProperty(EngineExtension):
            def __init__(self, eng):
                super().__init__(eng)  # register() runs here, before my_param is set
                self.my_param = 1.0

            @property
            def tolerance(self):
                return (
                    self.my_param * 0.01
                )  # would crash if getter ran during registration

            def my_op(self):
                return self.tolerance

        ext = _ExtWithProperty(engine)
        assert not hasattr(engine, "tolerance")  # property stays on the extension
        assert hasattr(engine, "my_op")
        engine.close()

    def test_extension_cannot_shadow_native_method(self):
        """Registering an extension that shadows a native engine method raises ValueError."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)

        class _ShadowsClose(EngineExtension):
            def close(self):
                pass

        with pytest.raises(ValueError, match="shadows native"):
            _ShadowsClose(engine)
        engine.close()

    def test_extension_methods_visible_to_inspect(self):
        """inspect.getmembers(engine, callable) includes extension methods."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        ext = self.make_test_extension(engine)
        public_callables = [
            m
            for m in dir(ext)
            if not m.startswith("_") and callable(getattr(ext, m, None))
        ]
        discovered = {name for name, _ in inspect.getmembers(engine, callable)}
        assert all(m in discovered for m in public_callables)
        engine.close()

    def test_extension_conflict_raises(self):
        """Registering two extensions with the same method name raises ValueError."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        self.make_test_extension(engine)
        with pytest.raises(ValueError):
            self.make_conflicting_extension(engine)
        engine.close()
