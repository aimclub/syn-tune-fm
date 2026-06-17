import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class DataBalancer:
    """
    Manages training sample balancing strategies.
    Supported strategies: 'none', 'random_over', 'random_under', 'synthetic'.
    """

    def __init__(
        self,
        strategy: str = "none",
        random_state: int = 42,
        task_type: str = "classification",
        synthetic_regression_augment_ratio: float = 1.0,
        synthetic_regression_mode: str = "categorical_balance",
        categorical_balance_cols: list | None = None,
        regression_random_balance_bins: int = 5,
    ):
        self.strategy = strategy
        self.random_state = random_state
        self.task_type = (task_type or "classification").strip().lower()
        self.synthetic_regression_augment_ratio = float(synthetic_regression_augment_ratio)
        self.synthetic_regression_mode = (synthetic_regression_mode or "categorical_balance").strip().lower()
        self.categorical_balance_cols = list(categorical_balance_cols or [])
        self.regression_random_balance_bins = max(2, int(regression_random_balance_bins))

    def balance(self, X_train: pd.DataFrame, y_train: pd.Series, generator=None, target_col: str = None):
        print(f"      Applying balancing strategy: {self.strategy}")

        if self.strategy == "none":
            return X_train.copy(), y_train.copy()

        if self.task_type == "regression" and self.strategy in ("random_over", "random_under"):
            return self._balance_regression_random(X_train, y_train, oversample=self.strategy == "random_over")

        elif self.strategy == "random_over":
            ros = RandomOverSampler(random_state=self.random_state)
            return ros.fit_resample(X_train, y_train)

        elif self.strategy == "random_under":
            rus = RandomUnderSampler(random_state=self.random_state)
            return rus.fit_resample(X_train, y_train)

        elif self.strategy == "synthetic":
            if generator is None:
                raise ValueError("Generator is required for 'synthetic' balancing.")

            if self.task_type == "regression":
                if self.synthetic_regression_mode == "categorical_balance":
                    return self._balance_regression_categorical(X_train, y_train, generator)
                if self.synthetic_regression_mode == "augment":
                    return self._balance_regression_augment(X_train, y_train, generator)
                raise ValueError(
                    f"Unknown synthetic_regression_mode={self.synthetic_regression_mode!r}. "
                    "Use 'categorical_balance' or 'augment'."
                )

            counts = y_train.value_counts()
            majority_count = counts.max()

            X_syn_list, y_syn_list = [X_train], [y_train]

            for cls, count in counts.items():
                deficit = majority_count - count
                if deficit > 0:
                    X_syn, y_syn = generator.conditional_sampling(
                        n_samples=deficit,
                        target_value=int(cls),
                    )
                    if y_syn.name != y_train.name:
                        y_syn = y_syn.rename(y_train.name)
                    X_syn_list.append(X_syn)
                    y_syn_list.append(y_syn)

            X_balanced = pd.concat(X_syn_list, ignore_index=True)
            y_balanced = pd.Series(np.concatenate([y.values for y in y_syn_list]))
            y_balanced.name = y_train.name

            idx = np.random.permutation(len(X_balanced))
            return X_balanced.iloc[idx].reset_index(drop=True), y_balanced.iloc[idx].reset_index(drop=True)

        else:
            raise ValueError(f"Unknown balancing strategy: {self.strategy}")

    def _balance_regression_random(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        oversample: bool,
    ):
        """ROS/RUS on quantile bins of continuous y; duplicated rows keep original y values."""
        y_name = y_train.name or "target"
        packed = pd.concat(
            [X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(y_name)],
            axis=1,
        )
        n_unique = int(packed[y_name].nunique(dropna=True))
        n_bins = min(self.regression_random_balance_bins, max(2, n_unique))
        try:
            y_bin = pd.qcut(packed[y_name], q=n_bins, duplicates="drop", labels=False)
        except ValueError:
            print(
                "      [Regression random balance] qcut failed; returning unbalanced train set."
            )
            return X_train.copy(), y_train.copy()

        y_bin = pd.Series(y_bin, name="_y_bin_")
        sampler = (
            RandomOverSampler(random_state=self.random_state)
            if oversample
            else RandomUnderSampler(random_state=self.random_state)
        )
        X_res, _ = sampler.fit_resample(packed, y_bin)
        X_out = X_res.drop(columns=[y_name])
        y_out = X_res[y_name].reset_index(drop=True)
        y_out.name = y_name
        mode = "random_over" if oversample else "random_under"
        print(
            f"      [Regression random balance] {mode} via {n_bins} target quantile bins: "
            f"{len(X_train)} -> {len(X_out)} rows"
        )
        return X_out.reset_index(drop=True), y_out

    def _balance_regression_augment(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        generator,
    ):
        n_extra = max(
            1,
            int(len(X_train) * self.synthetic_regression_augment_ratio),
        )
        X_syn, y_syn = generator.generate(n_samples=n_extra)
        if y_syn.name != y_train.name:
            y_syn = y_syn.rename(y_train.name)
        X_balanced = pd.concat([X_train.reset_index(drop=True), X_syn.reset_index(drop=True)], ignore_index=True)
        y_balanced = pd.concat(
            [y_train.reset_index(drop=True), y_syn.reset_index(drop=True)],
            ignore_index=True,
        )
        idx = np.random.RandomState(self.random_state).permutation(len(X_balanced))
        return X_balanced.iloc[idx].reset_index(drop=True), y_balanced.iloc[idx].reset_index(drop=True)

    def _balance_regression_categorical(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        generator,
    ):
        cols = [c for c in self.categorical_balance_cols if c in X_train.columns]
        if not cols:
            print(
                "      [Regression synthetic] no categorical/discrete columns to balance; "
                "falling back to joint augment."
            )
            return self._balance_regression_augment(X_train, y_train, generator)

        X_syn_list = [X_train.reset_index(drop=True)]
        y_syn_list = [y_train.reset_index(drop=True)]
        total_added = 0

        for col in cols:
            counts = X_train[col].value_counts(dropna=False)
            majority_count = int(counts.max())
            print(
                f"      [Regression synthetic] column {col!r}: majority={majority_count}, "
                f"n_categories={len(counts)}"
            )

            for val, count in counts.items():
                deficit = majority_count - int(count)
                if deficit <= 0:
                    continue
                try:
                    X_syn, y_syn = generator.conditional_sampling(
                        n_samples=deficit,
                        feature_conditions={col: val},
                    )
                except (ValueError, RuntimeError, NotImplementedError) as exc:
                    print(
                        f"      [Regression synthetic] skip {col}={val!r} "
                        f"(deficit={deficit}): {exc}"
                    )
                    continue

                if y_syn.name != y_train.name:
                    y_syn = y_syn.rename(y_train.name)
                X_syn_list.append(X_syn.reset_index(drop=True))
                y_syn_list.append(y_syn.reset_index(drop=True))
                total_added += len(X_syn)

        print(
            f"      [Regression synthetic] categorical balance added {total_added} rows "
            f"({len(X_train)} -> {sum(len(x) for x in X_syn_list)})"
        )

        X_balanced = pd.concat(X_syn_list, ignore_index=True)
        y_balanced = pd.concat(y_syn_list, ignore_index=True)
        idx = np.random.RandomState(self.random_state).permutation(len(X_balanced))
        return X_balanced.iloc[idx].reset_index(drop=True), y_balanced.iloc[idx].reset_index(drop=True)
