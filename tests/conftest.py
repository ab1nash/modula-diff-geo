"""
Pytest configuration and shared fixtures for geometric covariance tests.

This module provides JAX-aware fixtures and configuration for testing
differential geometry extensions to modula.
"""
import pytest
import jax
import jax.numpy as jnp

# Ensure reproducible tests across runs
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def base_key():
    """Root PRNG key for all tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def key(base_key, request):
    """Per-test PRNG key derived from test name for reproducibility."""
    test_id = hash(request.node.nodeid) % (2**31)
    return jax.random.fold_in(base_key, test_id)


@pytest.fixture(params=[2, 3, 5, 8])
def dim(request):
    """Parametrized dimension for testing across scales."""
    return request.param


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    """Parametrized dtype for numerical precision tests."""
    return jnp.float32 if request.param == "float32" else jnp.float64


@pytest.fixture
def tolerance(dtype):
    """Numerical tolerance based on dtype."""
    # Use relaxed tolerances due to matrix conditioning in random transforms
    # Matrix inversion and composition accumulate errors, especially at higher dims
    return 1e-4 if dtype == jnp.float32 else 1e-6


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "phase1: core type system tests")
    config.addinivalue_line("markers", "phase2: dualization tests")
    config.addinivalue_line("markers", "phase3: atomic primitive tests")
    config.addinivalue_line("markers", "phase4: composition tests")
    config.addinivalue_line("markers", "phase5: information geometry tests")
    config.addinivalue_line("markers", "invariant: mathematical invariant verification")

