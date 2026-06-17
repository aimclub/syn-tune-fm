## 🔬 Research Framework: Synthetic Data for TabPFN Fine-Tuning

This repository studies how different synthetic data generation methods affect TabPFN fine-tuning on **tabular classification and regression** tasks.

The pipeline supports **conditional sampling** from fitted generators (Gaussian Copula, CTGAN, TVAE, TabDDPM, GMM, and others) and compares five experimental regimes **C0–C4** with optional **Optuna HPO** over generator hyperparameters.

### 📂 Project structure

The project is modular and uses Hydra for configs, so you can switch generators, datasets, and experiment variants without code changes.

```plaintext
.
├── configs/                      # Hydra experiment setup
│   ├── dataset/                  # OpenML / CSV datasets (classification & regression)
│   ├── generator/                # Generator hyperparameters
│   ├── model/                    # TabPFN v1 / v2 wrappers
│   ├── experiment.yaml           # Default classification experiment
│   ├── experiment_c0_c4.yaml     # C0–C4 grid (Gaussian default)
│   └── experiment_c0_c4_generators.yaml  # C0–C4 + generator sweep / HPO
│
├── src/
│   ├── data_loader/              # OpenML and local CSV loaders
│   ├── data_processor/           # Schema, preprocessing, CV splits
│   ├── generators/               # Synthetic generators (fit → generate / conditional_sampling)
│   ├── models/                   # TabPFN wrappers
│   ├── metrics/                  # Classification + regression metrics
│   ├── training/                 # Fine-tuning loops, objectives, DataBalancer
│   ├── pipeline/                 # ExperimentRunner (Load → Gen → Balance → Train → Eval)
│   └── utils/                    # Plotting, TabPFN auth helpers
│
├── examples/                     # Generator usage examples
├── packages/
│   └── yandex-tab-ddpm/          # Vendored `tab_ddpm` for TabDDPMGenerator
├── tests/                        # Unit tests (generators, conditional sampling, balancing)
├── outputs/                      # Hydra single-run logs (gitignored)
├── multirun/                     # Hydra multirun logs (gitignored)
├── run_experiment.py             # Entry point
├── check_env.py                  # Quick import sanity check
└── requirements.txt
```

### 🐍 Environment

From the repo root, in a virtualenv:

```bash
pip install -r requirements.txt
pip install -e packages/yandex-tab-ddpm
```

Then verify imports:

```bash
python check_env.py
```

**TabPFN access:** set a Prior Labs API key before running experiments (accept the license in your account):

```bash
export TABPFN_TOKEN=your_key_here
# or: export PRIORLABS_API_KEY=...
# or: ~/.config/tabpfn/token  (single line)
# or: .env.local in repo root with TABPFN_TOKEN=...
```

See `src/utils/tabpfn_env.py` for the full lookup order.

To refresh vendored TabDDPM sources from upstream: `python scripts/refresh_tab_ddpm_vendor.py`, then re-run `pip install -e packages/yandex-tab-ddpm`.

### 🧪 Experiment variants (C0–C4)

| Variant | Fine-tune | Balancing |
|---------|-----------|-----------|
| **C0** `C0_No_FT` | no | none |
| **C1** `C1_FT_Imbal` | yes | none (imbalanced train) |
| **C2** `C2_FT_SynBal` | yes | **synthetic** (generator-based) |
| **C3** `C3_FT_RandBal` | yes | random oversampling |
| **C4** `C4_FT_DownBal` | yes | random undersampling |

On **regression** datasets:

- **C2** can balance rare categorical values via `conditional_sampling`, or augment with joint `generate()`.
- **C3/C4** use quantile bins of the continuous target for ROS/RUS.

Configure modes in `experiment_setup` (`synthetic_regression_mode`, `synthetic_regression_augment_ratio`, `regression_random_balance_bins`).

### 🛠 Running experiments

#### Default classification run

```bash
python run_experiment.py
```

#### C0–C4 on a dataset

```bash
python run_experiment.py --config-name experiment_c0_c4 dataset=adult experiment_setup.run_hpo=false
```

#### Compare generators (CTGAN / TVAE / TabDDPM / Gaussian)

Only **C2** depends on the generator choice. To avoid re-running C0, C1, C3, C4 for every generator:

```bash
python run_experiment.py --config-name experiment_c0_c4_generators -m \
  generator=ctgan,tvae,tabddpm \
  dataset=house_prices \
  experiment_setup.run_only_synthetic_balancing_variants=true \
  experiment_setup.run_hpo=false
```

#### Regression example (Ames Housing)

```bash
python run_experiment.py --config-name experiment_c0_c4_generators \
  dataset=house_prices generator=gaussian \
  experiment_setup.run_hpo=false
```

#### Multirun over datasets

```bash
python run_experiment.py --multirun --config-name experiment_c0_c4_generators \
  dataset=house_prices,brazilian_houses \
  experiment_setup.run_only_non_synthetic_balancing_variants=true \
  experiment_setup.run_hpo=false
```

Hydra writes results under `outputs/` (single run) or `multirun/` (sweeps). `run_experiment.py` also saves bar plots and appends a row to `run_results.csv` in the run folder.

### 📊 Datasets

**Classification** (OpenML): `adult`, `bank_marketing`, `blood_transfusion`, `credit_g`, `diabetes`, `ionosphere`, `iris`, `mc1`, `mushroom`, `parkinsons`, `pc4`, `telco_churn`, and others in `configs/dataset/`.

**Regression**: `house_prices` (Ames), `brazilian_houses`, `boston_housing`, `diamonds`, `mercedes_benz`, `abalone`.

Each dataset YAML sets `task_type`, target column, and default evaluation metrics (`macro_f1` / `rmse`, etc.).

### ⚙️ Generators

| Generator | Config key | Conditional sampling |
|-----------|------------|----------------------|
| Gaussian Copula | `gaussian` | yes |
| CTGAN | `ctgan` | yes |
| TVAE | `tvae` | yes |
| TabDDPM | `tabddpm` | yes (incl. joint X+y diffusion for regression) |
| GMM | `gmm` | yes |
| Mixed model / table augmentation | `mixed_model`, `tableaugmentation` | yes |

Switch via CLI: `python run_experiment.py generator=ctgan`.

**Example usage of generator classes:**  
`examples/example_generators_usage.py` — run: `python examples/example_generators_usage.py`

### 🔧 HPO (Optuna)

Enable with `experiment_setup.run_hpo=true`. The objective combines distributional distances (Wasserstein for numeric features, Jensen–Shannon for categorical/discrete) between real and synthetic data.

- `experiment_setup.hpo_js_mode=all` — all categorical/discrete columns contribute to JS.
- `experiment_setup.hpo_js_mode=single` — one column (`hpo_js_single_column` or dataset default).

### ➕ Adding a new generator

1. Create `src/generators/<name>/model.py` with a class inheriting `BaseDataGenerator` (`src/generators/base.py`). Implement `fit(X, y)`, `generate(n_samples)`, and optionally `_generate_conditional` for `conditional_sampling`.
2. Export the class in `src/generators/__init__.py` and register it in `runner._get_generator()`.
3. Add `configs/generator/<name>.yaml`.

### ✅ Tests

```bash
pytest tests/
```