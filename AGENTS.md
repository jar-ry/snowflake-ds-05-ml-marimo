# Marimo ML Pipeline — Agent Guide

## What This Repo Is

A reactive ML pipeline built with marimo notebooks for predicting `MONTHLY_CUSTOMER_VALUE`. Hybrid architecture: 6 marimo notebooks (plain `.py` files) + 4 shared Python modules + 1 config dataclass.

## Repo Structure

```
├── notebooks/
│   ├── orchestrator.py            # Chains all 5 stages with gated run buttons
│   ├── 01_feature_pipeline.py     # Entity, FeatureView, Dataset generation
│   ├── 02_training_pipeline.py    # HPO config, @remote to SPCS
│   ├── 03_promotion_pipeline.py   # Default/alias/tag, SHAP explain
│   ├── 04_serving_pipeline.py     # SPCS inference service, batch predict
│   └── 05_monitoring_pipeline.py  # Baseline table, ModelMonitor
├── lib/
│   ├── session.py                 # create_session(), create_model_registry(), create_feature_store()
│   ├── features.py                # load_data(), preprocess(), get_spine_df()
│   ├── modelling.py               # build_pipeline(), evaluate_model(), train(), generate_train_val_set()
│   └── versioning.py              # next_dataset_version(), next_model_version()
├── conf/
│   └── defaults.py                # PipelineConfig dataclass (all config defaults)
├── connection.json.example        # Snowflake credentials template
└── pyproject.toml
```

## Environment

Python >= 3.10. Key packages: `marimo>=0.10.0`, `snowflake-ml-python>=1.7.0`, `xgboost`, `scikit-learn`.

marimo is installed at `/opt/miniconda3/bin/marimo` (v0.23.2). Snowflake ML packages are in the same conda base env.

```bash
pip install marimo snowflake-ml-python xgboost scikit-learn
```

On macOS, xgboost requires `brew install libomp`.

## How to Run

```bash
# Interactive edit mode (buttons gate each stage)
marimo edit notebooks/orchestrator.py

# App mode (read-only UI)
marimo run notebooks/orchestrator.py

# Script mode / CI/CD (all gates bypassed)
python notebooks/orchestrator.py

# Run specific stages
python notebooks/orchestrator.py -- --stages 1,2

# Override config
python notebooks/orchestrator.py -- --connection PROD_CONN --database PROD_DB
```

Each sub-notebook is independently runnable:
```bash
marimo edit notebooks/01_feature_pipeline.py
python notebooks/01_feature_pipeline.py
```

## Lint / Validate

```bash
marimo check notebooks/orchestrator.py
marimo check notebooks/01_feature_pipeline.py
```

`marimo check` validates cell dependencies, variable naming, and notebook structure. 0 errors required. Cosmetic `markdown-indentation` warnings (marimo wants pure-md cells dedented) are harmless and do not affect functionality.

## Snowflake Connection

Uses Snowflake named connections (not `connection.json`). The session is created via:
```python
snowflake.connector.connect(connection_name="JCHEN_AWS1")
```

Default connection name and all other config live in `conf/defaults.py` as a `PipelineConfig` dataclass.

## Key Snowflake Objects

- **Database:** `RETAIL_REGRESSION_DEMO`
- **Schemas:** `DS` (raw data), `FEATURE_STORE`, `MODELLING` (Model Registry), `PROD_SCHEMA`
- **Compute Pool:** `CUSTOMER_VALUE_MODEL_POOL_CPU` (CPU_X64_L, min 1, max 10)
- **Warehouse:** `RETAIL_REGRESSION_DEMO_WH`
- **Model:** `MODELLING.UC01_SNOWFLAKEML_RF_REGRESSOR_MODEL`
- **Stage:** `DS.payload_stage`
- **Source Tables:** `DS.CUSTOMERS`, `DS.PURCHASE_BEHAVIOR`
- **Target Column:** `MONTHLY_CUSTOMER_VALUE`
- **14 Feature Columns:** AGE, GENDER, LOYALTY_TIER, TENURE_MONTHS, AVG_ORDER_VALUE, PURCHASE_FREQUENCY, RETURN_RATE, TOTAL_ORDERS, ANNUAL_INCOME, AVERAGE_ORDER_PER_MONTH, DAYS_SINCE_LAST_PURCHASE, DAYS_SINCE_SIGNUP, EXPECTED_DAYS_BETWEEN_PURCHASES, DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE

## Marimo Patterns

### Notebook File Format
Marimo notebooks are plain `.py` files:
```python
import marimo
__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    # shared imports

@app.cell
def _():
    # cell code
    return (variable,)

if __name__ == "__main__":
    app.run()
```

### Gating Pattern
Expensive operations are gated behind run buttons to prevent eager execution:
```python
run_button = mo.ui.run_button(label="Run")
mo.stop(not run_button.value)
# expensive code here
```

### CI/CD Mode Detection
The orchestrator detects script mode and bypasses interactive gates:
```python
is_interactive = mo.app_meta().mode != "script"
if is_interactive:
    mo.stop(not run_button.value)
elif stage_number not in run_stages:
    mo.stop(True)
```

### Variable Naming
Variables starting with `_` are private to a cell and NOT exported via marimo's reactive DAG. Use non-underscore names (`is_interactive`, `run_stages`) for variables that need to flow between cells.

### Config Override
`mo.cli_args()` returns CLI args passed after `--` in script mode. The config cell reads:
```python
_args = mo.cli_args()
sf_database = mo.ui.text(value=_args.get("database", _defaults.snowflake.database))
```

### `@remote` Training Pattern
`tune`, `RandomSearch`, `Registry` are imported **inside** the `@remote` function because they only exist in the Snowflake container runtime, not locally:
```python
@remote(pool_name, stage_name=stage, target_instances=3)
def train_remote(..., session):
    from snowflake.ml.modeling import tune
    from snowflake.ml.modeling.tune.search import RandomSearch
    # ... HPO logic
```

## Architecture Notes

- `lib/` modules have zero marimo dependency and are serializable by `@remote`
- `conf/defaults.py` is a `@dataclass` with nested configs: SnowflakeConfig, FeatureStoreConfig, ModelRegistryConfig, HPOConfig, ComputeConfig, ServingConfig, MonitoringConfig, FeatureColumns
- The `train()` function in `lib/modelling.py` is the per-trial HPO function executed by Ray workers on the compute pool
- Models are logged with `target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]` and `enable_explainability=True`
- Before HPO, the code pre-creates the model in the Registry with a dummy version to avoid race conditions from parallel trials

## Common Modifications

- **Change model type:** Edit `build_pipeline()` in `lib/modelling.py` and update HPO search space in `orchestrator.py` Stage 2
- **Change features:** Edit `lib/features.py` (`load_data`, `preprocess`) or the inlined versions in `01_feature_pipeline.py`
- **Change HPO:** Modify the multiselect options in the Stage 2 HPO cell, or update `conf/defaults.py` HPOConfig
- **Change compute:** Adjust `target_instances` in `conf/defaults.py` ComputeConfig or the orchestrator UI slider
- **Add a stage:** Create a new `notebooks/0N_*.py`, add a gated cell block in `orchestrator.py`

## Dual Repo Setup

This code is tracked in two git repos:
- **Blog repo:** `/Users/jarrychen/Code/blog/` (tracks `marimo-ml-jobs/code/` as a regular tree)
- **Standalone repo:** `https://github.com/jar-ry/snowflake-ds-05-ml-marimo.git`

After making changes, push to both:
```bash
# Standalone
cd /Users/jarrychen/Code/blog/marimo-ml-jobs/code
git add -A && git commit -m "msg" && git push

# Blog
cd /Users/jarrychen/Code/blog
git add marimo-ml-jobs/ && git commit -m "msg" && git push
```
