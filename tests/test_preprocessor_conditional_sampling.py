"""TabularPreprocessor + generators: same transform applies to real and conditional synthetic X."""
import numpy as np
import pandas as pd
import pytest

from src.data_processor import TabularPreprocessor
from src.data_processor.schema import TabularSchema


def _fit_prep(X: pd.DataFrame, y: pd.Series) -> TabularPreprocessor:
    df = X.copy()
    tcol = y.name if getattr(y, "name", None) not in (None, "") else "target"
    df[tcol] = y.reset_index(drop=True).to_numpy()
    schema = TabularSchema.infer_from_dataframe(df, target_col=tcol)
    prep = TabularPreprocessor()
    prep.fit(df, schema)
    return prep


def _feature_matrix(prep: TabularPreprocessor, X: pd.DataFrame) -> np.ndarray:
    out = prep.transform(X)
    return out[prep.schema.feature_cols].to_numpy(dtype=np.float64)


@pytest.fixture
def xy_numeric():
    """Numeric-only features (GMM-friendly)."""
    np.random.seed(1)
    n = 100
    X = pd.DataFrame(
        {
            "f0": np.random.randn(n),
            "f1": np.random.randn(n),
        }
    )
    y = pd.Series(np.zeros(n, dtype=int), name="target")
    y.iloc[50:] = 1
    return X, y


@pytest.fixture
def xy_mixed():
    """Numeric + categorical columns (SDV / table aug)."""
    np.random.seed(1)
    n = 100
    X = pd.DataFrame(
        {
            "num_a": np.random.randn(n),
            "num_b": np.random.randint(0, 5, size=n),
            "cat": np.random.choice(list("xyz"), size=n),
        }
    )
    y = pd.Series(np.zeros(n, dtype=int), name="target")
    y.iloc[50:] = 1
    return X, y


def test_preprocessor_conditional_sampling_gmm(xy_numeric):
    from src.generators import GMMGenerator

    X, y = xy_numeric
    prep = _fit_prep(X, y)

    gen = GMMGenerator(seed=42, n_samples=20, n_components=3, covariance_type="full")
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(15, target_value=0)
    assert len(Xs) == 15 and (ys == 0).all()

    real_arr = _feature_matrix(prep, X.iloc[:20])
    syn_arr = _feature_matrix(prep, Xs)
    assert real_arr.shape == (20, 2)
    assert syn_arr.shape == (15, 2)
    assert np.isfinite(real_arr).all() and np.isfinite(syn_arr).all()


def test_preprocessor_conditional_sampling_gaussian(xy_mixed):
    pytest.importorskip("sdv")
    from src.generators import GaussianCopulaGenerator

    X, y = xy_mixed
    prep = _fit_prep(X, y)

    gen = GaussianCopulaGenerator(seed=42, n_samples=30)
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(12, target_value=1)
    assert len(Xs) == 12 and (ys == 1).all()

    syn_arr = _feature_matrix(prep, Xs)
    assert syn_arr.shape == (12, 3)


def test_preprocessor_conditional_sampling_table_augmentation(xy_mixed):
    from src.generators import TableAugmentationGenerator

    X, y = xy_mixed
    prep = _fit_prep(X, y)

    gen = TableAugmentationGenerator(seed=42, n_samples=20, reuse_original_target=True)
    gen.fit(X, y)
    Xs, ys = gen.conditional_sampling(10, target_value=0)
    assert len(Xs) == 10

    # Table augmentation may return a subset of feature columns
    assert set(Xs.columns).issubset(set(X.columns))
    cols = [c for c in prep.schema.feature_cols if c in Xs.columns]
    out = prep.transform(Xs)
    sub = out[cols].to_numpy(dtype=np.float64)
    assert sub.shape[0] == 10
    assert sub.shape[1] == len(cols)
