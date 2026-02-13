from typing import Tuple
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

from src.generators.base import BaseDataGenerator


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
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = num_cols + cat_cols
    return preprocessor, feature_names


class MixedModelGenerator(BaseDataGenerator):
    """Генератор BGM + случайный классификатор"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MixedModelGenerator":
        self._preprocessor, self._feature_names = _get_fitted_preprocessor(X)
        self._X_proc = pd.DataFrame(
            self._preprocessor.transform(X),
            columns=self._feature_names,
        )
        self._y = y
        self.is_fitted = True
        return self

    def generate(self, n_samples: int = None, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        n = n_samples or kwargs.get("n_samples") or self.params.get("n_samples") or len(self._X_proc)
        seed = self.params.get("seed", 42)
        rng = np.random.RandomState(seed)
        bgm_params = _sample_bgm_params(rng)
        clf = _sample_classifier(rng)
        try:
            bgm = BayesianGaussianMixture(**bgm_params, random_state=seed)
            bgm.fit(self._X_proc)
        except Exception:
            idx = rng.choice(len(self._X_proc), n, replace=True)
            return self._X_proc.iloc[idx].reset_index(drop=True), self._y.iloc[idx].values
        try:
            clf.fit(self._X_proc, self._y)
        except Exception:
            idx = rng.choice(len(self._X_proc), n, replace=True)
            return self._X_proc.iloc[idx].reset_index(drop=True), self._y.iloc[idx].values
        X_syn_np, _ = bgm.sample(n_samples=n)
        X_syn_df = pd.DataFrame(X_syn_np, columns=self._feature_names)
        y_syn = clf.predict(X_syn_df)
        if len(np.unique(y_syn)) < 2:
            idx = rng.choice(len(self._X_proc), n, replace=True)
            return self._X_proc.iloc[idx].reset_index(drop=True), self._y.iloc[idx].values
        return X_syn_df, y_syn

    def get_preprocessor(self):
        """Для оценки на тесте — тот же препроцессинг. Возвращает (preprocessor, feature_names) или None."""
        if getattr(self, "_preprocessor", None) is None:
            return None
        return self._preprocessor, self._feature_names
