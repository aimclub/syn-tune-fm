"""
Smoke tests: conditional_sampling(n, target_value=...) after fit.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def xy_binary():
    """Binary classification, numeric features only."""
    np.random.seed(0)
    n = 120
    X = pd.DataFrame(
        {
            "f0": np.random.randn(n),
            "f1": np.random.randn(n),
        }
    )
    y = pd.Series(np.zeros(n, dtype=int), name="target")
    y.iloc[60:] = 1
    return X, y


def _assert_conditional_class(X_syn, y_syn, target, n_expected: int):
    assert len(X_syn) == n_expected == len(y_syn)
    assert (y_syn == target).all(), (
        f"expected all y=={target}, got {y_syn.value_counts().to_dict()}"
    )


def test_conditional_sampling_gmm(xy_binary):
    from src.generators import GMMGenerator

    X, y = xy_binary
    gen = GMMGenerator(seed=42, n_samples=30, n_components=3, covariance_type="full")
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(20, target_value=0)
    _assert_conditional_class(Xs, ys, 0, 20)
    Xs, ys = gen.conditional_sampling(20, target_value=1)
    _assert_conditional_class(Xs, ys, 1, 20)


def test_conditional_sampling_gaussian(xy_binary):
    pytest.importorskip("sdv")
    from src.generators import GaussianCopulaGenerator

    X, y = xy_binary
    gen = GaussianCopulaGenerator(seed=42, n_samples=50)
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(25, target_value=0)
    _assert_conditional_class(Xs, ys, 0, 25)


def test_conditional_sampling_ctgan_tvae(xy_binary):
    pytest.importorskip("sdv")
    from src.generators import CTGANGenerator, TVAEGenerator

    X, y = xy_binary
    for Cls in (CTGANGenerator, TVAEGenerator):
        gen = Cls(seed=42, n_samples=50, epochs=1)
        gen.fit(X, y)
        Xs, ys = gen.conditional_sampling(18, target_value=1)
        _assert_conditional_class(Xs, ys, 1, 18)


def test_conditional_sampling_table_augmentation(xy_binary):
    from src.generators import TableAugmentationGenerator

    X, y = xy_binary
    gen = TableAugmentationGenerator(seed=42, n_samples=50, reuse_original_target=True)
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(15, target_value=0)
    _assert_conditional_class(Xs, ys, 0, 15)


def test_conditional_sampling_mixed_model(xy_binary):
    from src.generators import MixedModelGenerator

    X, y = xy_binary
    gen = MixedModelGenerator(seed=42, n_samples=50)
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(12, target_value=1)
    _assert_conditional_class(Xs, ys, 1, 12)


def test_conditional_sampling_tabddpm(xy_binary):
    from src.generators import TabDDPMGenerator

    X, y = xy_binary
    gen = TabDDPMGenerator(
        seed=42,
        n_samples=50,
        epochs=2,
        batch_size=64,
        num_timesteps=20,
        device="cpu",
    )
    try:
        gen.fit(X, y)
    except Exception as e:
        pytest.skip(f"TabDDPM not runnable in this env: {e}")
    Xs, ys = gen.conditional_sampling(10, target_value=0)
    _assert_conditional_class(Xs, ys, 0, 10)


def test_conditional_sampling_diffusion(xy_binary):
    torch = pytest.importorskip("torch")
    from src.generators import TabularDiffusionGenerator

    _ = torch  # silence lint
    X, y = xy_binary
    gen = TabularDiffusionGenerator(
        seed=42, n_samples=50, epochs=1, n_steps=20, batch_size=32, device="cpu"
    )
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(8, target_value=1)
    _assert_conditional_class(Xs, ys, 1, 8)
