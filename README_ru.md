## 🔬 Research Framework: Synthetic Data for TabPFN Fine-Tuning

Репозиторий для исследования влияния синтетических данных на дообучение TabPFN на **табличных задачах классификации и регрессии**.

Поддерживается **условная генерация** (`conditional_sampling`) из обученных генераторов (Gaussian Copula, CTGAN, TVAE, TabDDPM, GMM и др.) и сравнение пяти режимов **C0–C4** с опциональным **Optuna HPO** по гиперпараметрам генератора.

### 📂 Структура проекта

Проект модульный, конфигурация через Hydra — генератор, датасет и варианты эксперимента меняются без правок кода.

```plaintext
.
├── configs/                      # Hydra-конфиги
│   ├── dataset/                  # OpenML / CSV (классификация и регрессия)
│   ├── generator/                # Параметры генераторов
│   ├── model/                    # Обёртки TabPFN v1 / v2
│   ├── experiment.yaml           # Базовый классификационный эксперимент
│   ├── experiment_c0_c4.yaml     # Сетка C0–C4 (Gaussian по умолчанию)
│   └── experiment_c0_c4_generators.yaml  # C0–C4 + sweep по генераторам / HPO
│
├── src/
│   ├── data_loader/              # Загрузка OpenML и локальных CSV
│   ├── data_processor/           # Схема, препроцессинг, CV-сплиты
│   ├── generators/               # Генераторы (fit → generate / conditional_sampling)
│   ├── models/                   # Обёртки TabPFN
│   ├── metrics/                  # Метрики классификации и регрессии
│   ├── training/                 # Циклы fine-tune, loss, DataBalancer
│   ├── pipeline/                 # ExperimentRunner (Load → Gen → Balance → Train → Eval)
│   └── utils/                    # Графики, авторизация TabPFN
│
├── examples/                     # Примеры использования генераторов
├── packages/yandex-tab-ddpm/     # Vendored `tab_ddpm` для TabDDPM
├── tests/                        # Юнит-тесты
├── outputs/                      # Логи Hydra (gitignored)
├── multirun/                     # Логи multirun (gitignored)
├── run_experiment.py             # Точка входа
├── check_env.py                  # Проверка импортов
└── requirements.txt
```

### 🐍 Окружение

Из корня репозитория, в virtualenv:

```bash
pip install -r requirements.txt
pip install -e packages/yandex-tab-ddpm
```

Проверка:

```bash
python check_env.py
```

**Доступ к TabPFN:** перед запуском задайте API-ключ Prior Labs (лицензию нужно принять в аккаунте):

```bash
export TABPFN_TOKEN=ваш_ключ
# или: export PRIORLABS_API_KEY=...
# или: ~/.config/tabpfn/token  (одна строка)
# или: .env.local в корне с TABPFN_TOKEN=...
```

Порядок поиска ключа — в `src/utils/tabpfn_env.py`.

Обновление vendored TabDDPM: `python scripts/refresh_tab_ddpm_vendor.py`, затем снова `pip install -e packages/yandex-tab-ddpm`.

### 🧪 Варианты эксперимента (C0–C4)

| Вариант | Fine-tune | Балансировка |
|---------|-----------|--------------|
| **C0** `C0_No_FT` | нет | нет |
| **C1** `C1_FT_Imbal` | да | нет (имбаланс в train) |
| **C2** `C2_FT_SynBal` | да | **синтетическая** (через генератор) |
| **C3** `C3_FT_RandBal` | да | random oversampling |
| **C4** `C4_FT_DownBal` | да | random undersampling |

На **регрессии**:

- **C2** — балансировка редких категорий через `conditional_sampling` или аугментация joint `generate()`.
- **C3/C4** — ROS/RUS по квантильным бинам непрерывного таргета.

Режимы задаются в `experiment_setup` (`synthetic_regression_mode`, `synthetic_regression_augment_ratio`, `regression_random_balance_bins`).

### 🛠 Запуск экспериментов

#### Базовый классификационный прогон

```bash
python run_experiment.py
```

#### C0–C4 на датасете

```bash
python run_experiment.py --config-name experiment_c0_c4 dataset=adult experiment_setup.run_hpo=false
```

#### Сравнение генераторов (CTGAN / TVAE / TabDDPM / Gaussian)

От выбора генератора зависит в основном **C2**. Чтобы не пересчитывать C0, C1, C3, C4 для каждого генератора:

```bash
python run_experiment.py --config-name experiment_c0_c4_generators -m \
  generator=ctgan,tvae,tabddpm \
  dataset=house_prices \
  experiment_setup.run_only_synthetic_balancing_variants=true \
  experiment_setup.run_hpo=false
```

#### Пример регрессии (Ames Housing)

```bash
python run_experiment.py --config-name experiment_c0_c4_generators \
  dataset=house_prices generator=gaussian \
  experiment_setup.run_hpo=false
```

#### Multirun по датасетам

```bash
python run_experiment.py --multirun --config-name experiment_c0_c4_generators \
  dataset=house_prices,brazilian_houses \
  experiment_setup.run_only_non_synthetic_balancing_variants=true \
  experiment_setup.run_hpo=false
```

Результаты — в `outputs/` или `multirun/`. `run_experiment.py` строит bar-графики и дописывает строку в `run_results.csv` в папке прогона.

### 📊 Датасеты

**Классификация** (OpenML): `adult`, `bank_marketing`, `blood_transfusion`, `credit_g`, `diabetes`, `ionosphere`, `iris`, `mc1`, `mushroom`, `parkinsons`, `pc4`, `telco_churn` и др. в `configs/dataset/`.

**Регрессия**: `house_prices` (Ames), `brazilian_houses`, `boston_housing`, `diamonds`, `mercedes_benz`, `abalone`.

В YAML датасета задаются `task_type`, таргет и метрики (`macro_f1` / `rmse` и т.д.).

### ⚙️ Генераторы

| Генератор | Ключ конфига | Условная генерация |
|-----------|--------------|-------------------|
| Gaussian Copula | `gaussian` | да |
| CTGAN | `ctgan` | да |
| TVAE | `tvae` | да |
| TabDDPM | `tabddpm` | да (в т.ч. joint X+y для регрессии) |
| GMM | `gmm` | да |
| Mixed model / table augmentation | `mixed_model`, `tableaugmentation` | да |

Переключение: `python run_experiment.py generator=ctgan`.

**Пример API генераторов:**  
`examples/example_generators_usage.py` — `python examples/example_generators_usage.py`

### 🔧 HPO (Optuna)

Включение: `experiment_setup.run_hpo=true`. Целевая функция — расстояния между реальными и синтетическими данными (Wasserstein для числовых, Jensen–Shannon для категориальных/discrete).

- `experiment_setup.hpo_js_mode=all` — все категориальные/discrete колонки в JS.
- `experiment_setup.hpo_js_mode=single` — одна колонка (`hpo_js_single_column` или дефолт датасета).

### ➕ Добавление нового генератора

1. Создать `src/generators/<name>/model.py` с классом-наследником `BaseDataGenerator`. Реализовать `fit(X, y)`, `generate(n_samples)` и при необходимости `_generate_conditional`.
2. Экспорт в `src/generators/__init__.py` и регистрация в `runner._get_generator()`.
3. Добавить `configs/generator/<name>.yaml`.

### ✅ Тесты

```bash
pytest tests/
```

Основные: `test_generators.py`, `test_conditional_sampling.py`, `test_preprocessor_conditional_sampling.py`, `test_balancing.py`.

### 📦 Таксономия методов генерации

* *Traditional* — Gaussian, GMM
* *VAE* — TVAE
* *GAN* — CTGAN
* *Diffusion* — TabDDPM
* *LLM* — GReaT (опциональная зависимость)
