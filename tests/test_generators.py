import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({
        "a": np.random.randn(n).cumsum(),
        "b": np.random.randint(0, 5, size=n),
        "c": np.random.rand(n),
    })
    y = pd.Series(np.random.randint(0, 2, size=n), name="target")
    return X, y


def test_gaussian_generator(sample_data):
    try:
        from src.generators import GaussianCopulaGenerator
    except ImportError:
        pytest.skip("SDV not installed")
    X, y = sample_data
    gen = GaussianCopulaGenerator(n_samples=20, seed=42)
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=15)
    assert isinstance(X_syn, pd.DataFrame)
    assert isinstance(y_syn, pd.Series)
    assert len(X_syn) == 15
    assert len(y_syn) == 15
    assert list(X_syn.columns) == list(X.columns)


def test_ctgan_generator(sample_data):
    try:
        from src.generators import CTGANGenerator
    except ImportError:
        pytest.skip("SDV not installed")
    X, y = sample_data
    gen = CTGANGenerator(n_samples=20, epochs=1, seed=42)
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=10)
    assert isinstance(X_syn, pd.DataFrame)
    assert len(X_syn) == 10
    assert list(X_syn.columns) == list(X.columns)


def test_tvae_generator(sample_data):
    try:
        from src.generators import TVAEGenerator
    except ImportError:
        pytest.skip("SDV not installed")
    X, y = sample_data
    gen = TVAEGenerator(n_samples=20, epochs=1, seed=42)
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=10)
    assert isinstance(X_syn, pd.DataFrame)
    assert len(X_syn) == 10
    assert list(X_syn.columns) == list(X.columns)


def test_gmm_generator(sample_data):
    from src.generators import GMMGenerator
    X, y = sample_data
    gen = GMMGenerator(n_samples=20, seed=42, n_components=3, covariance_type="full")
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=10)
    assert isinstance(X_syn, pd.DataFrame)
    assert len(X_syn) == 10
    assert list(X_syn.columns) == list(X.columns)


def test_mixed_model_generator(sample_data):
    from src.generators import MixedModelGenerator
    X, y = sample_data
    gen = MixedModelGenerator(n_samples=20, seed=42)
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=10)
    assert isinstance(X_syn, pd.DataFrame)
    assert len(X_syn) == 10


def test_tableaugmentation_generator(sample_data):
    from src.generators import TableAugmentationGenerator
    X, y = sample_data
    gen = TableAugmentationGenerator(n_samples=20, seed=42, reuse_original_target=True)
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=15)
    assert isinstance(X_syn, pd.DataFrame)
    assert len(X_syn) == 15
    assert len(y_syn) == 15
    for c in X_syn.columns:
        assert c in X.columns
    assert gen.get_selected_columns() is not None


def test_generators_accept_params_dict(sample_data):
    from src.generators import GMMGenerator
    X, y = sample_data
    params = {"seed": 123, "n_samples": 30, "n_components": 2}
    gen = GMMGenerator(**params)
    gen.fit(X, y)
    X_syn, y_syn = gen.generate(n_samples=10)
    assert gen.seed == 123
    assert gen.params.get("n_samples") == 30
    assert len(X_syn) == 10


def test_generate_without_fit_raises():
    """generate() must not be called before fit(); same contract as CatRepBench."""
    from src.generators import GMMGenerator
    import pytest
    gen = GMMGenerator(seed=42, n_components=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(n_samples=10)


def test_runner_factory_resolves_generators():
    from omegaconf import OmegaConf
    from src.pipeline.runner import ExperimentRunner

    cfg = OmegaConf.create({
        "dataset": {"name": "openml", "params": {"dataset_id": 37, "target_column": "class", "test_size": 0.2}},
        "generator": {"name": "gaussian", "params": {"n_samples": 50, "seed": 42}},
        "model": {"name": "tabpfn", "params": {"device": "cpu", "N_ensemble_configurations": 2}},
        "evaluation": {"metrics": ["accuracy"]},
    })
    runner = ExperimentRunner(cfg)
    gen = runner._get_generator()
    assert gen is not None
    assert gen.params.get("n_samples") == 50
