from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.generators.base import BaseDataGenerator, train_subset_from_conditions

LABEL_COL = "target"


def _sample_bgm_params(rng):
    def log_uniform(a, b):
        return 10 ** rng.uniform(np.log10(a), np.log10(b))
    return dict(
        n_components=rng.randint(1, 31),
        covariance_type=rng.choice(["full", "tied", "diag", "spherical"]),
        tol=log_uniform(1e-5, 1e-1),
        reg_covar=log_uniform(1e-7, 1e-4),
        max_iter=rng.randint(100, 1001),
        n_init=rng.randint(1, 11),
        init_params="kmeans" if rng.rand() > 0.5 else "random",
        weight_concentration_prior_type=rng.choice(["dirichlet_process", "dirichlet_distribution"]),
        mean_precision_prior=rng.uniform(0.1, 10.0),
        warm_start=bool(rng.randint(0, 2)),
        verbose=0,
    )


def _sample_classifier(rng):
    model_name = rng.choice(["RandomForest", "DecisionTree", "MLP", "SVC", "HistGradientBoosting"])
    def log_int(a, b):
        return int(round(10 ** rng.uniform(np.log10(a), np.log10(b))))
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=log_int(10, 500),
            criterion=rng.choice(["gini", "log_loss", "entropy"]),
            max_depth=log_int(10, 100),
            min_samples_split=rng.randint(2, 21),
            min_samples_leaf=rng.randint(1, 11),
            max_leaf_nodes=rng.randint(10, 101),
            bootstrap=bool(rng.randint(0, 2)),
            n_jobs=-1,
        )
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier(
            criterion=rng.choice(["gini", "entropy", "log_loss"]),
            splitter=rng.choice(["best", "random"]),
            max_depth=log_int(5, 100),
            min_samples_split=rng.randint(2, 21),
            min_samples_leaf=rng.randint(1, 11),
            max_features=rng.choice([0.1, 0.25, 0.5, 0.75, 1.0, "sqrt", "log2", None]),
        )
    elif model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(rng.randint(1, 101),),
            activation=rng.choice(["relu", "logistic", "tanh"]),
            solver=rng.choice(["adam", "sgd", "lbfgs"]),
            alpha=rng.uniform(0.0001, 0.1),
            batch_size=rng.choice([32, 64, 128, "auto"]),
            learning_rate=rng.choice(["constant", "invscaling", "adaptive"]),
            learning_rate_init=rng.uniform(0.0001, 0.01),
            max_iter=rng.randint(100, 1001),
            momentum=rng.uniform(0.5, 0.95),
            nesterovs_momentum=bool(rng.randint(0, 2)),
            early_stopping=bool(rng.randint(0, 2)),
        )
    elif model_name == "SVC":
        def log_float(a, b):
            return 10 ** rng.uniform(np.log10(a), np.log10(b))
        return SVC(
            kernel=rng.choice(["linear", "rbf", "poly", "sigmoid"]),
            C=log_float(1e-6, 1e6),
            degree=rng.randint(1, 6),
            gamma=rng.choice(["scale", "auto"]),
            coef0=rng.uniform(-1, 1),
            shrinking=bool(rng.randint(0, 2)),
            probability=True,
            tol=10 ** rng.uniform(-5, -2),
            class_weight=rng.choice([None, "balanced"]),
            max_iter=rng.randint(100, 1001),
            break_ties=bool(rng.randint(0, 2)),
        )
    else:
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=rng.uniform(0.01, 1.0),
            max_iter=rng.randint(50, 1001),
            max_leaf_nodes=rng.randint(5, 101),
            max_depth=rng.randint(3, 16),
            min_samples_leaf=rng.randint(5, 101),
            l2_regularization=rng.uniform(0.0, 1.0),
            max_bins=rng.randint(10, 256),
        )


def _get_fitted_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
        sparse_threshold=0,
    )
    preprocessor.fit(X)
    fn_out = getattr(preprocessor, "get_feature_names_out", None)
    feature_names = fn_out().tolist() if callable(fn_out) else (num_cols + cat_cols)
    return preprocessor, feature_names


class MixedModelGenerator(BaseDataGenerator):
    """BGM + random classifier."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = None,
        **kwargs,
    ):
        super().__init__(seed=seed, n_samples=n_samples, **kwargs)
        self.seed = seed
        self.n_samples = n_samples
        self._bgm = None
        self._clf = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MixedModelGenerator":
        self._X_fit = X.reset_index(drop=True)
        self._preprocessor, self._feature_names = _get_fitted_preprocessor(X)
        self._X_proc = pd.DataFrame(
            self._preprocessor.transform(X),
            columns=self._feature_names,
        )
        self._y = y.reset_index(drop=True)
        rng = np.random.RandomState(self.seed)
        bgm_params = _sample_bgm_params(rng)
        self._clf = _sample_classifier(rng)
        self._bgm = BayesianGaussianMixture(**bgm_params, random_state=self.seed)
        self._bgm.fit(self._X_proc)
        self._clf.fit(self._X_proc, self._y)
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted or self._bgm is None or self._clf is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n = n_samples or kwargs.get("n_samples") or self.n_samples or len(self._X_proc)
        seed = self.params.get("seed", self.seed)
        np.random.seed(seed)
        X_syn_np, _ = self._bgm.sample(n_samples=n)
        X_syn_df = pd.DataFrame(X_syn_np, columns=self._feature_names)
        y_syn = self._clf.predict(X_syn_df)
        if len(np.unique(y_syn)) < 2:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self._X_proc), n, replace=True)
            return self._X_proc.iloc[idx].reset_index(drop=True), self._y.iloc[idx].values
        return X_syn_df, y_syn

    def conditional_sampling(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return super().conditional_sampling(
            n_samples,
            target_value=target_value,
            feature_conditions=feature_conditions,
            **kwargs,
        )

    def _generate_conditional(
        self,
        n_samples: int,
        target_value: Optional[int] = None,
        feature_conditions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        fc = dict(feature_conditions or {})
        if not fc and target_value is None:
            return self.generate(n_samples, **kwargs)
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        train_df = self._X_fit.copy()
        train_df[LABEL_COL] = self._y.values
        sub = train_subset_from_conditions(
            train_df,
            target_value=target_value,
            feature_conditions=fc,
            label_col=LABEL_COL,
        )
        X_class = self._X_proc.loc[sub.index]
        if len(X_class) == 0:
            raise ValueError(
                "No training rows match target_value=%r and feature_conditions=%r"
                % (target_value, fc)
            )

        seed = self.params.get("seed", self.seed)
        rng = np.random.RandomState(seed)
        bgm_params = _sample_bgm_params(rng)
        class_bgm = BayesianGaussianMixture(**bgm_params, random_state=seed)
        class_bgm.fit(X_class)
        X_syn_np, _ = class_bgm.sample(n_samples=n_samples)

        X_syn_df = pd.DataFrame(X_syn_np, columns=self._feature_names)
        if target_value is not None:
            y_syn = pd.Series([target_value] * n_samples)
        else:
            y_syn = self._clf.predict(X_syn_df)
        return X_syn_df, y_syn

    def get_preprocessor(self):
        """Return (preprocessor, feature_names) for evaluation; None if not fitted."""
        if getattr(self, "_preprocessor", None) is None:
            return None
        return self._preprocessor, self._feature_names
