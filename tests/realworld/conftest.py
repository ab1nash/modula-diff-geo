"""
Shared fixtures for real-world hypothesis tests.
"""
import pytest
import jax


@pytest.fixture
def key():
    """Provide a reproducible JAX random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def keys():
    """Provide multiple JAX random keys for tests that need them."""
    base = jax.random.PRNGKey(42)
    return jax.random.split(base, 10)

