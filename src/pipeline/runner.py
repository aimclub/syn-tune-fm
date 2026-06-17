import hydra
from omegaconf import DictConfig
import logging
import pandas as pd
import numpy as np
import os
import sys
import optuna
import scipy.stats
import time
import traceback
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from src.data_processor.splits import SplitConfigHoldout, SplitConfigKFold, apply_imbalance
from collections import defaultdict
from omegaconf import OmegaConf

# --- Imports: Metrics ---
from src.metrics.factory import MetricFactory

# --- Imports: Models ---
from src.models.tabpfn_v1_wrp import TabPFNModelV1

# --- Imports: Training ---
from src.training.loops import TrainingLoop

# --- Imports: Data Loaders ---
try:
    from src.data_loader.openml_loader import OpenMLDataLoader
except ImportError:
    print("Warning: OpenMLDataLoader not found. Please implement it in src/data_loader/")
    OpenMLDataLoader = None

try:
    from src.data_loader.csv_loader import CSVDataLoader
except ImportError:
    print("Warning: CSVDataLoader not found. Please implement it in src/data_loader/")
    CSVDataLoader = None

# --- Imports: Generators ---
from src.generators import (
    GaussianCopulaGenerator,
    CTGANGenerator,
    TVAEGenerator,
    GMMGenerator,
    MixedModelGenerator,
    TableAugmentationGenerator,
    TabularDiffusionGenerator,
    TabDDPMGenerator,
)

from src.training.balancing import DataBalancer
from collections import defaultdict
from src.utils.experiment_labels import dataset_label_from_cfg

LOGGER = logging.getLogger(__name__)

_OPENML_LOADER_PARAM_KEYS = frozenset(
    {
        "dataset_id",
        "target_column",
        "test_size",
        "random_state",
        "balance",
        "target_quantile_bins",
        "hpo_js_single_column_default",
        "task_type",
        "max_rows",
    }
)

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        OmegaConf.set_struct(cfg, False)
        self._dataset_task_type = str(
            OmegaConf.select(cfg, "dataset.params.task_type", default="classification")
            or "classification"
        ).lower()

        em = OmegaConf.select(cfg, "dataset.params.evaluation_metrics")
        if em is not None:
            cfg.evaluation.metrics = list(em)

        if self._dataset_task_type == "regression":
            if cfg.model.get("params") is None:
                cfg.model.params = OmegaConf.create({})
            cfg.model.params.task_type = "regression"

        self.cfg = cfg

        if "metrics" in self.cfg.evaluation:
            self.metrics = MetricFactory.get_metrics(self.cfg.evaluation.metrics)
        else:
            self.metrics = []
            print("Warning: No metrics defined in config.")

    def _filter_experiment_variants(self, variants):
        """Optionally restrict which C0–C4 variants run (see experiment_c0_c4_generators.yaml)."""
        exp_setup = self.cfg.get("experiment_setup", {})
        only_syn = exp_setup.get("run_only_synthetic_balancing_variants", False)
        only_non_syn = exp_setup.get("run_only_non_synthetic_balancing_variants", False)
        if only_syn and only_non_syn:
            raise ValueError(
                "Cannot set both run_only_synthetic_balancing_variants and "
                "run_only_non_synthetic_balancing_variants to true."
            )
        if only_syn:
            filtered = {
                k: v
                for k, v in variants.items()
                if (v or {}).get("balancing") == "synthetic"
            }
            if not filtered:
                raise ValueError(
                    "experiment_setup.run_only_synthetic_balancing_variants=true but no variant has "
                    "balancing: synthetic. Remove the flag or add such a variant."
                )
            print(
                f"      [Config] run_only_synthetic_balancing_variants=true → variants: {list(filtered.keys())} "
                f"(skipping modes that do not use the generative model)"
            )
            return filtered
        if only_non_syn:
            filtered = {
                k: v
                for k, v in variants.items()
                if (v or {}).get("balancing") != "synthetic"
            }
            if not filtered:
                raise ValueError(
                    "experiment_setup.run_only_non_synthetic_balancing_variants=true but every variant "
                    "uses synthetic balancing. Remove the flag or adjust variants."
                )
            print(
                f"      [Config] run_only_non_synthetic_balancing_variants=true → variants: {list(filtered.keys())} "
                f"(skipping C2 / synthetic balancing)"
            )
            return filtered
        return variants

    def _get_data_loader(self):
        name = self.cfg.dataset.name
        params = self.cfg.dataset.params
        
        if name == 'openml' or 'dataset_id' in params:
            if OpenMLDataLoader is None:
                raise ImportError("OpenMLDataLoader is not implemented or imported.")
            p = OmegaConf.to_container(params, resolve=True) or {}
            loader_params = {k: v for k, v in p.items() if k in _OPENML_LOADER_PARAM_KEYS}
            return OpenMLDataLoader(**loader_params)
        
        if name == 'csv':
            if CSVDataLoader is None:
                raise ImportError("CSVDataLoader is not implemented or imported.")
            return CSVDataLoader(**params)
        
        raise ValueError(f"Unknown dataset loader: {name}")

    def _get_generator(self):
        name = self.cfg.generator.name
        params = self.cfg.generator.params
        if name == "gaussian":
            return GaussianCopulaGenerator(**params)
        elif name == "ctgan":
            return CTGANGenerator(**params)
        elif name == "tvae":
            return TVAEGenerator(**params)
        elif name == "gmm":
            return GMMGenerator(**params)
        elif name == "mixed_model":
            return MixedModelGenerator(**params)
        elif name == "tableaugmentation":
            return TableAugmentationGenerator(**params)
        elif name == "diffusion":
            return TabularDiffusionGenerator(**params)
        elif name == "tabddpm":
            return TabDDPMGenerator(**params)
        raise ValueError(
            f"Generator '{name}' is not implemented. Choose: gaussian, ctgan, tvae, gmm, mixed_model, tableaugmentation, diffusion, tabddpm."
        )

    def _get_model(self):
        model_config = self.cfg.get('model', {})
        model_name = model_config.get('name', '')
        params = model_config.get('params', {})
        
        if model_name == 'tabpfn_v1':
            from src.models.tabpfn_v1_wrp import TabPFNModelV1
            return TabPFNModelV1(params)
            
        elif model_name == 'tabpfn_v2':
            from src.models.tabpfn_v2_wrp import TabPFNModelV2
            return TabPFNModelV2(params)
            
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

    def run(self):
        ds_label = dataset_label_from_cfg(self.cfg.dataset)
        LOGGER.info(
            f"--- Starting Experiment: dataset={ds_label} (config name={self.cfg.dataset.name}) | "
            f"generator={self.cfg.generator.name} ---"
        )

        # 1. Load Data
        LOGGER.info("[1/4] Loading Data & Inferring Schema...")
        loader = self._get_data_loader()
        self.datamodule = loader.load()
        self.target_col = self.datamodule.schema.target_col

        # 2. Optional: Generator HPO 
        best_params = self._run_optuna_tuning()
        if best_params:
            self.cfg.generator.params.update(best_params)

        # 3. K-Fold Cross Validation
        LOGGER.info("[2/4] Starting K-Fold Cross Validation...")
        exp_setup = self.cfg.get("experiment_setup", {})
        n_folds = exp_setup.get("n_folds", 5)
        cv_seed = exp_setup.get("cv_random_state", 42)
        kfold_cfg = SplitConfigKFold(n_splits=n_folds, random_seed=cv_seed)
        self.datamodule.prepare_kfold(kfold_cfg)
        
        variants = self.cfg.get("variants", {"default_run": {"finetune": True, "balancing": "none"}})
        variants = self._filter_experiment_variants(variants)
        fold_results = {v_name: defaultdict(list) for v_name in variants.keys()}

        # Read imbalance setting from config (default is None, i.e., no change)
        minority_fraction = self.cfg.get("minority_fraction", None)

        for fold_id in range(kfold_cfg.n_splits):
            LOGGER.info("--- Processing Fold %s/%s ---", fold_id + 1, kfold_cfg.n_splits)
            
            fold_data = self.datamodule.get_fold(fold_id)
            X_train = fold_data.train.drop(columns=[self.target_col])
            y_train = fold_data.train[self.target_col]
            X_test = fold_data.test.drop(columns=[self.target_col])
            y_test = fold_data.test[self.target_col]

            # --- NEW: APPLY IMBALANCE TO TRAIN ONLY (ID 68) ---
            if minority_fraction is not None and minority_fraction < 0.5:
                if self._dataset_task_type == "regression":
                    LOGGER.warning(
                        "minority_fraction is set but dataset is regression; skipping artificial class imbalance."
                    )
                else:
                    LOGGER.info("Applying artificial imbalance (minority_fraction=%s)...", minority_fraction)
                    X_train, y_train = apply_imbalance(X_train, y_train, minority_fraction)

            # Fit generator on (possibly truncated) X_train
            generator = None
            needs_synthetic = any(v.get('balancing') == 'synthetic' for v in variants.values())
            if needs_synthetic:
                LOGGER.info("Fitting generator for synthetic balancing...")
                generator = self._get_generator()
                generator.set_task_type(self._dataset_task_type)
                generator.fit(X_train, y_train)

            # Run each experiment variant
            for variant_name, variant_cfg in variants.items():
                LOGGER.info("Executing variant: %s", variant_name)
                
                strategy = variant_cfg.get('balancing', 'none')
                exp_setup = self.cfg.get("experiment_setup", {})
                aug_ratio = float(exp_setup.get("synthetic_regression_augment_ratio", 1.0))
                reg_mode = str(
                    exp_setup.get("synthetic_regression_mode", "categorical_balance")
                ).lower()
                schema = self.datamodule.schema
                cat_balance_cols = [
                    c
                    for c in (schema.categorical_cols + schema.discrete_cols)
                    if c in X_train.columns
                ]
                reg_bins = int(exp_setup.get("regression_random_balance_bins", 5))
                balancer = DataBalancer(
                    strategy=strategy,
                    random_state=42,
                    task_type=self._dataset_task_type,
                    synthetic_regression_augment_ratio=aug_ratio,
                    synthetic_regression_mode=reg_mode,
                    categorical_balance_cols=cat_balance_cols,
                    regression_random_balance_bins=reg_bins,
                )
                
                # Your balancer should now internally use generator.sample_conditional
                X_train_bal, y_train_bal = balancer.balance(
                    X_train, y_train, generator=generator, target_col=self.target_col
                )
                max_context_size = exp_setup.get("max_train_context_size")
                if max_context_size is not None and int(max_context_size) > 0:
                    max_context_size = int(max_context_size)
                    if len(X_train_bal) > max_context_size:
                        LOGGER.warning(
                            "Reducing context from %s to %s samples "
                            "(experiment_setup.max_train_context_size=%s)...",
                            len(X_train_bal),
                            max_context_size,
                            max_context_size,
                        )
                        vc_ctx = y_train_bal.value_counts()
                        strat_ctx = (
                            y_train_bal
                            if len(vc_ctx) >= 2 and vc_ctx.min() >= 2
                            else None
                        )
                        X_train_bal, _, y_train_bal, _ = train_test_split(
                            X_train_bal,
                            y_train_bal,
                            train_size=max_context_size,
                            stratify=strat_ctx,
                            random_state=42,
                        )

                model = self._get_model()
                do_finetune = variant_cfg.get('finetune', True)
                
                if do_finetune:
                    training_cfg = dict(self.cfg.model.get('params', {}))
                    trainer = TrainingLoop(model=model, config=training_cfg)
                    model = trainer.run(
                        X_train=X_train_bal, y_train=y_train_bal, 
                        X_real=X_train, y_real=y_train
                    )
                else:
                    LOGGER.info("Skipping Fine-Tuning. Using strictly In-Context Learning (Frozen Backbone).")
                    model.fit_context(X_train_bal, y_train_bal)

                y_pred = model.predict(X_test)
                y_probs = None
                if self._dataset_task_type != "regression":
                    try:
                        y_probs = model.predict_proba(X_test)
                    except (NotImplementedError, ValueError):
                        y_probs = None

                for metric in self.metrics:
                    val = metric.calculate(y_test, y_pred, y_probs)
                    fold_results[variant_name][metric.name].append(val)
                    LOGGER.info("%s: %.4f", metric.name, val)

        # 4. Aggregate results with STANDARD DEVIATION
        LOGGER.info("[4/4] Aggregating Cross-Validation Results...")
        final_results = {}
        for variant_name, metrics_dict in fold_results.items():
            final_results[variant_name] = {}
            LOGGER.info("%s Metrics:", variant_name)
            for m_name, vals in metrics_dict.items():
                # Calculate mean and standard deviation over 5 folds
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                
                # Record with _mean and _std suffixes
                final_results[variant_name][f"{m_name}_mean"] = mean_val
                final_results[variant_name][f"{m_name}_std"] = std_val
                
                LOGGER.info("%s: %.4f ± %.4f", m_name, mean_val, std_val)

        return final_results
    
    def _run_optuna_tuning(self):
        # --- 1. ADDING "SWITCH" FOR SPEED ---
        do_hpo = self.cfg.get('experiment_setup', {}).get('run_hpo', True) # Default is True
        if not do_hpo:
            LOGGER.info("[HPO] Tuning is disabled (run_hpo=false). Using default parameters.")
            return None # Return None to use parameters from config

        # Disable OmegaConf strict mode
        OmegaConf.set_struct(self.cfg, False)
        
        exp_setup = self.cfg.get("experiment_setup", {})
        tuning_split_size = exp_setup.get("tuning_split_size", 0.2)
        tuning_seed = exp_setup.get("tuning_random_state", 5)
        holdout_cfg = SplitConfigHoldout(val_size=tuning_split_size, random_seed=tuning_seed)
        self.datamodule.prepare_holdout(holdout_cfg)
        holdout_data = self.datamodule.get_holdout()
        
        X_train_full = holdout_data.train.drop(columns=[self.target_col])
        y_train_full = holdout_data.train[self.target_col]
        X_val = holdout_data.val.drop(columns=[self.target_col])
        schema = self.datamodule.schema

        continuous_cols = [c for c in schema.continuous_cols if c in X_val.columns]
        discrete_cols = [
            c for c in (schema.discrete_cols + schema.categorical_cols)
            if c in X_val.columns and c not in continuous_cols
        ]

        hpo_js_mode = str(exp_setup.get("hpo_js_mode", "all") or "all").lower()
        if hpo_js_mode in ("multi", "all"):
            hpo_js_mode = "all"
        if hpo_js_mode not in ("all", "single"):
            raise ValueError(
                "experiment_setup.hpo_js_mode must be one of: all, multi, single "
                f"(got {exp_setup.get('hpo_js_mode')!r})."
            )
        ds_params = OmegaConf.to_container(self.cfg.dataset.params, resolve=True) or {}
        single_default = ds_params.get("hpo_js_single_column_default")
        single_override = exp_setup.get("hpo_js_single_column")
        single_col = single_override if single_override not in (None, "", "null") else single_default

        if hpo_js_mode == "single":
            if not single_col:
                raise ValueError(
                    "experiment_setup.hpo_js_mode=single requires hpo_js_single_column "
                    "(CLI) or dataset.params.hpo_js_single_column_default in the dataset yaml."
                )
            if single_col not in discrete_cols:
                raise ValueError(
                    f"hpo_js single column {single_col!r} is not among discrete/categorical "
                    f"columns in the holdout val split: {discrete_cols[:20]}..."
                )
            discrete_cols_for_js = [single_col]
        else:
            discrete_cols_for_js = list(discrete_cols)

        LOGGER.info(
            "      [HPO] Distribution objective columns: "
            f"{len(continuous_cols)} continuous (Wasserstein), "
            f"{len(discrete_cols_for_js)} discrete/categorical for JS-div "
            f"(mode={hpo_js_mode!r}, schema has {len(discrete_cols)} discrete/cat cols)"
        )

        # --- 2. SUBSAMPLING FOR OPTUNA ACCELERATION ---
        # Train the generator only on 30% of the data, this is enough to search for hyperparameters
        sample_size = int(len(X_train_full) * 0.3)
        # If the data is very small (less than 500 rows), take all
        if sample_size < 500:
            sample_size = len(X_train_full)
            
        X_train_hpo = X_train_full.sample(n=sample_size, random_state=42)
        y_train_hpo = y_train_full.loc[X_train_hpo.index]

        def _js_divergence_for_column(
            real_series: pd.Series,
            syn_series: pd.Series,
            eps: float = 1e-12,
        ) -> float:
            real_clean = real_series.dropna().astype(str)
            syn_clean = syn_series.dropna().astype(str)
            if real_clean.empty or syn_clean.empty:
                return float("nan")

            categories = pd.Index(pd.concat([real_clean, syn_clean], ignore_index=True).unique())
            if categories.empty:
                return float("nan")

            real_probs = (
                real_clean.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy(dtype=float)
            )
            syn_probs = (
                syn_clean.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy(dtype=float)
            )

            real_probs = real_probs + eps
            syn_probs = syn_probs + eps
            real_probs = real_probs / real_probs.sum()
            syn_probs = syn_probs / syn_probs.sum()

            midpoint = 0.5 * (real_probs + syn_probs)
            js_div = 0.5 * scipy.stats.entropy(real_probs, midpoint) + 0.5 * scipy.stats.entropy(syn_probs, midpoint)
            return float(js_div)

        def objective(trial):
            gen_name = self.cfg.generator.name
            params = {}
            
            if gen_name == "ctgan":
                # Paper-aligned mapping for this implementation:
                # n_iter -> epochs, discriminator_n_iter -> discriminator_steps,
                # generator/discriminator_{n_layers_hidden,n_units_hidden} -> *_dim tuples.
                params['epochs'] = trial.suggest_int('epochs', 100, 1000, step=100)
                params['batch_size'] = trial.suggest_categorical('batch_size', [100, 200, 500])
                lr_choice = trial.suggest_categorical('lr', [1e-4, 2e-4, 1e-3])
                params['generator_lr'] = lr_choice
                params['discriminator_lr'] = lr_choice
                params['discriminator_steps'] = trial.suggest_int('discriminator_steps', 1, 5)

                gen_layers = trial.suggest_int('generator_n_layers_hidden', 1, 4)
                gen_units = trial.suggest_int('generator_n_units_hidden', 50, 150, step=50)
                dis_layers = trial.suggest_int('discriminator_n_layers_hidden', 1, 4)
                dis_units = trial.suggest_int('discriminator_n_units_hidden', 50, 150, step=50)
                params['generator_dim'] = tuple([gen_units] * gen_layers)
                params['discriminator_dim'] = tuple([dis_units] * dis_layers)

                # Keep pac fixed and compatible with all chosen batch sizes.
                params['pac'] = 10
            elif gen_name == "tvae":
                # Paper-aligned mapping for this implementation:
                # n_iter -> epochs, n_units_embedding -> embedding_dim,
                # encoder/decoder hidden layers+units -> compress/decompress dims.
                params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300, 400, 500])
                params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
                params['embedding_dim'] = trial.suggest_int('embedding_dim', 50, 500, step=50)

                enc_layers = trial.suggest_int('encoder_n_layers_hidden', 1, 5)
                enc_units = trial.suggest_int('encoder_n_units_hidden', 50, 500, step=50)
                dec_layers = trial.suggest_int('decoder_n_layers_hidden', 1, 5)
                dec_units = trial.suggest_int('decoder_n_units_hidden', 50, 500, step=50)
                params['compress_dims'] = tuple([enc_units] * enc_layers)
                params['decompress_dims'] = tuple([dec_units] * dec_layers)

                # Nearest available regularization analog in this implementation.
                params['l2scale'] = trial.suggest_categorical('l2scale', [1e-4, 1e-3])
                params['loss_factor'] = 2
            elif gen_name == "tabddpm":
                # Paper-aligned mapping for this implementation:
                # n_iter -> epochs
                # Keep search space conservative to reduce NaNs during sampling.
                params['epochs'] = trial.suggest_int('epochs', 1000, 5000, log=True)
                params['batch_size'] = trial.suggest_int('batch_size', 256, 1024, log=True)
                params['lr'] = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
                # Avoid longer diffusion chains which are more prone to instability.
                params['num_timesteps'] = trial.suggest_categorical('num_timesteps', [50, 100])
            elif gen_name == "gmm":
                params['n_components'] = trial.suggest_int('n_components', 2, 20)
            
            original_params = OmegaConf.to_container(self.cfg.generator.params, resolve=True) if self.cfg.generator.params else {}
            
            if not self.cfg.generator.params:
                self.cfg.generator.params = {}
            for k, v in params.items():
                self.cfg.generator.params[k] = v

            LOGGER.info(
                "[HPO] Trial %s/%s started with params: %s",
                trial.number + 1,
                n_trials,
                params,
            )
            
            try:
                gen = self._get_generator()
                gen.set_task_type(self._dataset_task_type)
                gen.fit(X_train_hpo, y_train_hpo) # Train on truncated (fast) dataset
                
                # Generate test set
                X_syn, _ = gen.generate(n_samples=len(X_val))
                
                distance_scores = []

                for col in continuous_cols:
                    if col not in X_syn.columns:
                        continue
                    val_col = X_val[col].dropna().values
                    syn_col = X_syn[col].dropna().values
                    if len(val_col) == 0 or len(syn_col) == 0:
                        continue
                    dist = scipy.stats.wasserstein_distance(val_col, syn_col)
                    if np.isfinite(dist):
                        distance_scores.append(float(dist))

                for col in discrete_cols_for_js:
                    if col not in X_syn.columns:
                        continue
                    js_div = _js_divergence_for_column(X_val[col], X_syn[col])
                    if np.isfinite(js_div):
                        distance_scores.append(float(js_div))

                # If no valid column score can be computed, return a penalty.
                if not distance_scores:
                    return float('inf')

                return float(np.mean(distance_scores))
                
            except BaseException as e:
                # tab_ddpm raises FoundNANsError inheriting from BaseException,
                # so catch it here and mark trial as pruned instead of aborting run.
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                LOGGER.warning("[Trial failed] %s: %r", type(e).__name__, e)
                LOGGER.warning("[Trial failed] Traceback:\n%s", traceback.format_exc().rstrip())
                raise optuna.exceptions.TrialPruned()
            finally:
                self.cfg.generator.params = original_params

        n_trials = self.cfg.get('experiment_setup', {}).get('optuna_trials', 10)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        hpo_started_at = time.time()

        def _trial_progress_callback(study_obj, trial):
            elapsed = time.time() - hpo_started_at
            status = trial.state.name
            value_str = (
                f"{float(trial.value):.6f}"
                if trial.value is not None and np.isfinite(trial.value)
                else str(trial.value)
            )
            LOGGER.info(
                "[HPO] Trial %s/%s finished | status=%s | value=%s | elapsed=%.1fs",
                trial.number + 1,
                n_trials,
                status,
                value_str,
                elapsed,
            )
            if status == "COMPLETE":
                best_trial = study_obj.best_trial
                best_value = (
                    f"{float(best_trial.value):.6f}"
                    if best_trial.value is not None and np.isfinite(best_trial.value)
                    else str(best_trial.value)
                )
                LOGGER.info(
                    "[HPO] Current best after trial %s: trial=%s, value=%s",
                    trial.number + 1,
                    best_trial.number + 1,
                    best_value,
                )

        study.optimize(objective, n_trials=n_trials, callbacks=[_trial_progress_callback])

        completed_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and np.isfinite(t.value)
        ]
        if not completed_trials:
            LOGGER.warning("[HPO] No successful completed trials. Using default generator parameters.")
            return None

        best_params = self._normalize_best_hpo_params(self.cfg.generator.name, study.best_params)
        LOGGER.info("Best HPO parameters found: %s", best_params)
        return best_params

    def _normalize_best_hpo_params(self, gen_name: str, best_params: dict) -> dict:
        """Convert Optuna categorical aliases back to real generator params."""
        params = dict(best_params)

        if gen_name == "ctgan":
            dim_map = {
                "128x128": (128, 128),
                "256x256": (256, 256),
                "256x256x256": (256, 256, 256),
                "512x512": (512, 512),
            }
            if "generator_dim" in params and isinstance(params["generator_dim"], str):
                params["generator_dim"] = dim_map[params["generator_dim"]]
            if "discriminator_dim" in params and isinstance(params["discriminator_dim"], str):
                params["discriminator_dim"] = dim_map[params["discriminator_dim"]]

        elif gen_name == "tvae":
            dim_map = {
                "64x64": (64, 64),
                "128x128": (128, 128),
                "256x256": (256, 256),
            }
            if "compress_dims" in params and isinstance(params["compress_dims"], str):
                params["compress_dims"] = dim_map[params["compress_dims"]]
            if "decompress_dims" in params and isinstance(params["decompress_dims"], str):
                params["decompress_dims"] = dim_map[params["decompress_dims"]]

        elif gen_name == "tabddpm":
            layer_map = {
                "128x128": [128, 128],
                "256x256": [256, 256],
                "512x256": [512, 256],
                "512x512": [512, 512],
            }
            if "rtdl_d_layers" in params and isinstance(params["rtdl_d_layers"], str):
                params["rtdl_d_layers"] = layer_map[params["rtdl_d_layers"]]

        return params
