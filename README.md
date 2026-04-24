# 05 - Reactive ML Pipeline with Marimo

**Maturity Level:** ⭐⭐⭐ Advanced | **Runs In:** Local IDE / CI/CD → Snowflake Container Runtime

## Overview

A hybrid multi-notebook ML pipeline built with [marimo](https://marimo.io), combining the simplicity of a single notebook with the modularity of a production framework. Each pipeline stage is an independent marimo notebook backed by shared Python modules, with a single orchestrator that chains all five stages.

The same file works three ways: interactive development (`marimo edit`), read-only app (`marimo run`), and headless CI/CD (`python file.py`).

## When to Use

- Interactive development with reactive UI and gated execution
- CI/CD pipelines that run the same notebook code headlessly
- Teams that want each pipeline stage independently runnable and debuggable
- Git-friendly notebooks (marimo notebooks are plain `.py` files)
- Modular domain logic testable with pytest

## What the Pipeline Does

The pipeline predicts `MONTHLY_CUSTOMER_VALUE` using XGBoost, end-to-end on Snowflake:

- **Stage 1 (Features)** registers a CUSTOMER entity, creates a FeatureView backed by a Dynamic Table, and generates a versioned Snowflake Dataset
- **Stage 2 (Training)** submits an HPO job via `@remote` to a SPCS compute pool, running 10 RandomSearch trials across 3 nodes
- **Stage 3 (Promotion)** sets the best model as default, applies alias/tag, copies to PROD_SCHEMA, and generates SHAP explanations
- **Stage 4 (Serving)** deploys the model as a SPCS inference service and runs batch prediction
- **Stage 5 (Monitoring)** creates a baseline table and configures a ModelMonitor for drift detection

## Contents

```
marimo-ml-jobs/code/
├── notebooks/
│   ├── orchestrator.py            # Chains all 5 stages, full config UI
│   ├── 01_feature_pipeline.py     # Feature Store, FeatureView, Dataset
│   ├── 02_training_pipeline.py    # HPO config, @remote job submission
│   ├── 03_promotion_pipeline.py   # Promote, tag, SHAP explain
│   ├── 04_serving_pipeline.py     # SPCS service, batch predict
│   └── 05_monitoring_pipeline.py  # Baseline, ModelMonitor
├── lib/
│   ├── session.py                 # Snowflake session from named connection
│   ├── features.py                # load_data(), preprocess(), get_spine_df()
│   ├── modelling.py               # build_pipeline(), evaluate_model(), train()
│   └── versioning.py              # Auto-increment dataset/model versions
├── conf/
│   └── defaults.py                # PipelineConfig dataclass (all defaults)
├── connection.json.example        # Snowflake credentials template
└── pyproject.toml
```

## How It Works

```
┌──────────────────────────────┐        ┌──────────────────────────────────────┐
│        Local Machine         │        │            Snowflake                 │
│                              │        │                                      │
│  orchestrator.py (marimo)    │        │  ┌──────────────────────────────┐    │
│  ┌─ Stage 1: Features ──────┼───────► │  │  Feature Store + Dataset     │    │
│  ├─ Stage 2: Training ──────┼───────► │  │  Compute Pool (SPCS)         │    │
│  ├─ Stage 3: Promotion ─────┼───────► │  │  Model Registry              │    │
│  ├─ Stage 4: Serving ───────┼───────► │  │  SPCS Inference Service      │    │
│  └─ Stage 5: Monitoring ────┼───────► │  │  ModelMonitor                │    │
│                              │        │  └──────────────────────────────┘    │
│  lib/ (pure Python modules)  │        │                                      │
│  conf/defaults.py            │        │                                      │
└──────────────────────────────┘        └──────────────────────────────────────┘
```

## Three Execution Modes

**Interactive (edit mode)** — develop and debug with reactive UI, each stage gated by a Run button:
```bash
marimo edit notebooks/orchestrator.py
```

**App mode** — same buttons, read-only presentation:
```bash
marimo run notebooks/orchestrator.py
```

**CI/CD (script mode)** — all gates bypassed, runs end-to-end using defaults:
```bash
python notebooks/orchestrator.py
```

Select specific stages:
```bash
python notebooks/orchestrator.py -- --stages 1,2,3
```

Override config from CLI:
```bash
python notebooks/orchestrator.py -- \
  --stages all \
  --connection PROD_CONN \
  --database PROD_DB \
  --warehouse PROD_WH \
  --num_trials 20
```

Each sub-notebook is also independently runnable:
```bash
marimo edit notebooks/01_feature_pipeline.py
python notebooks/01_feature_pipeline.py
```

## Config Cascade

```
conf/defaults.py (PipelineConfig dataclass)
    ↓ overridden by
mo.cli_args() (--key value in script mode)
    ↓ displayed via
mo.ui elements (editable in interactive mode)
```

In script mode, `mo.ui` elements return their initial values (set from CLI args or defaults). In interactive mode, the user can edit them before clicking Run.

## Prerequisites

- Python >= 3.10 with `marimo`, `snowflake-ml-python`, `xgboost`, `scikit-learn`
- A Snowflake named connection (e.g. `JCHEN_AWS1`) configured in `~/.snowflake/connections.toml`
- Database `RETAIL_REGRESSION_DEMO` with schemas `DS`, `FEATURE_STORE`, `MODELLING`
- Compute pool `CUSTOMER_VALUE_MODEL_POOL_CPU` and stage `payload_stage`
- Source tables: `DS.CUSTOMERS`, `DS.PURCHASE_BEHAVIOR`

## Quick Start

```bash
pip install marimo snowflake-ml-python xgboost scikit-learn
marimo edit notebooks/orchestrator.py
```

Click "Connect to Snowflake", then click each stage's Run button sequentially.

## Snowflake Services Used

- Feature Store (Entity, FeatureView, Dynamic Tables)
- Datasets and DataConnectors
- ML Jobs (`@remote` decorator to SPCS compute pool)
- Tuner / HPO (`tune.Tuner`, `RandomSearch`)
- Experiment Tracking
- Model Registry (versioning, aliases, tags, cross-schema copy)
- Model Explainability (SHAP via `explain` function)
- SPCS Model Service (container-based inference)
- Model Monitoring (`ModelMonitor` for drift detection)

## Related Repos

| Repo | Description |
|------|-------------|
| [snowflake-ds-setup](https://github.com/jar-ry/snowflake-ds-setup) | Environment setup, data generation, helper utilities (run first) |
| [snowflake-ds-02-ml-jobs-notebook](https://github.com/jar-ry/snowflake-ds-02-ml-jobs-notebook) | Same pipeline as a single Jupyter notebook with `@remote` |
| [snowflake-ds-03-ml-jobs-framework](https://github.com/jar-ry/snowflake-ds-03-ml-jobs-framework) | Production framework with CLI dispatcher and `submit_directory` |
