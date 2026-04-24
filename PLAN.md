# Marimo ML Jobs: Customer Value Model

## Decision: Hybrid Multi-Notebook Architecture

We take the best ideas from all three prior blogs and combine them with marimo-native patterns:

- **From Blog 3 (ml-jobs-notebook)**: The pipeline stages, `@remote` for SPCS compute, Feature Store / Model Registry / ML Jobs APIs
- **From Blog 4 (ml-jobs-framework)**: Modular domain logic in plain Python modules, config separation, clean file boundaries
- **From marimo**: Reactive DAG per notebook, `@app.function` for cross-notebook imports, `mo.ui` for interactive config, `mo.stop`/`mo.ui.run_button` for gating expensive ops, `mo.cache` for caching, lazy runtime mode

### Why This Scales to 50+ Cells

The monolithic single-notebook approach (original plan) breaks down at scale because:
- Too many global variables crowd marimo's one-definition-per-variable namespace
- A 50+ cell `.py` file is hard to navigate, even with marimo's DAG visualization
- You can't debug or re-run just the training stage without scrolling past feature engineering
- `@remote` serialization is cleaner when training logic lives in plain `.py` modules

The framework approach (Blog 4) is overkill because:
- ~20 files of boilerplate for one pipeline
- CLI dispatcher is unnecessary when each notebook is independently runnable
- `submit_directory` payload assembly is fragile

The hybrid solves both: **multiple small marimo notebooks** (each 8-12 cells, independently runnable) + **plain Python modules** (testable, serializable by `@remote`) + **one orchestrator notebook** (the single-pane-of-glass experience).

### Why marimo uniquely enables this

marimo notebooks are `.py` files. A function decorated with `@app.function` in one notebook can be imported by another notebook or script via normal `from notebook import func`. This means:

- Each sub-notebook is a first-class Python module AND a reactive notebook
- The orchestrator imports stages from sub-notebooks without needing a CLI dispatcher
- Any notebook can be run standalone (`marimo edit`, `python notebook.py`, or `marimo run`)

---

## Architecture

```
marimo-ml-jobs/code/
├── notebooks/
│   ├── 01_feature_pipeline.py         # marimo: FE, FeatureView, Dataset (~10 cells)
│   ├── 02_training_pipeline.py        # marimo: HPO config, @remote, results (~10 cells)
│   ├── 03_promotion_pipeline.py       # marimo: promote, tag, SHAP explain (~8 cells)
│   ├── 04_serving_pipeline.py         # marimo: SPCS service, batch predict (~8 cells)
│   ├── 05_monitoring_pipeline.py      # marimo: baseline, ModelMonitor (~6 cells)
│   └── orchestrator.py               # marimo: imports & chains all stages (~12 cells)
├── lib/
│   ├── __init__.py
│   ├── session.py                     # create_session() via named connection
│   ├── features.py                    # load_data(), preprocess() (Snowpark)
│   ├── modelling.py                   # build_pipeline(), evaluate(), train()
│   └── versioning.py                 # dataset/model version helpers
├── conf/
│   ├── __init__.py
│   └── defaults.py                    # PipelineConfig dataclass with all defaults
├── pyproject.toml
└── PLAN.md
```

**~14 files** (vs Blog 3's 4 files vs Blog 4's ~20 files). The sweet spot.

### Layer Responsibilities

**`conf/defaults.py`** - Single source of truth for all config. A `@dataclass` with sections for Snowflake connection, Feature Store, Model Registry, HPO, compute, serving, monitoring. Each notebook can override via `mo.ui` elements or accept defaults for headless script execution.

**`lib/`** - Pure Python modules with zero marimo dependency. Testable with pytest. Serializable by `@remote`. Four focused modules instead of Blog 3's kitchen-sink `useful_fns.py`:
- `session.py`: Session creation from named connection (not connection.json)
- `features.py`: Snowpark DataFrames for load + preprocess
- `modelling.py`: sklearn Pipeline construction, evaluation, training loop with Tuner
- `versioning.py`: Auto-increment dataset/model versions

**`notebooks/`** - Each notebook is a reactive pipeline stage. Handles orchestration, visualization, and interactive config for its domain. Delegates all logic to `lib/`. Each notebook follows the same pattern:
1. `import marimo as mo` + setup cell
2. Config cell with `mo.ui` elements (pre-filled from `conf/defaults.py`)
3. Logic cells that call `lib/` functions
4. Display cells with `mo.md()`, `mo.ui.table()`, plots
5. Gate cell with `mo.ui.run_button` for side-effect operations

**`notebooks/orchestrator.py`** - The "Blog 3 experience" rebuilt for scale. Imports and chains all 5 stages. Uses `mo.ui.tabs` to organize stages visually. Can run the full pipeline headlessly via `python notebooks/orchestrator.py` or interactively via `marimo edit notebooks/orchestrator.py`.

---

## Notebook Details

### 01_feature_pipeline.py (~10 cells)

Cells:
1. Setup: `import marimo as mo`
2. Config: database, schema, warehouse (mo.ui.text, pre-filled from defaults)
3. Session: create Snowflake session from config
4. Registry + Feature Store: create/reference MR and FS
5. Entity: create/get CUSTOMER entity
6. Load data: import `load_data` from lib/features.py, display with mo.ui.table
7. Preprocess: import `preprocess` from lib/features.py, display derived features
8. FeatureView: register FV as Dynamic Table (gated with mo.ui.run_button)
9. Dataset: generate versioned training dataset (gated with mo.ui.run_button)
10. Summary: mo.md with dataset version, row count, feature list

Exports (via @app.function): `get_session`, `get_feature_store`, `get_training_dataset`

### 02_training_pipeline.py (~10 cells)

Cells:
1. Setup: `import marimo as mo`
2. Config: HPO params (num_trials, target_instances, search space ranges as mo.ui.slider)
3. Session + dataset: either from orchestrator args or standalone (re-create session)
4. Define training functions: import from lib/modelling.py
5. Define @remote train_remote: wraps lib functions, uses config values
6. HPO search space: built reactively from slider values
7. Submit training: gated with mo.ui.run_button + mo.stop
8. Wait + logs: results.wait(), show_logs()
9. Results table: mo.ui.table with all trial metrics (sortable)
10. Best model summary: mo.md with champion metrics

Exports: `get_training_results`, `get_best_run_name`

### 03_promotion_pipeline.py (~8 cells)

Cells:
1. Setup
2. Config: model name, alias, tag name
3. Session + best model: from args or standalone
4. Set DEFAULT version
5. Set PROD alias + create tag
6. Copy to PROD_SCHEMA
7. SHAP explanations + violin plot
8. Promotion summary: mo.md

Exports: `get_promoted_model`

### 04_serving_pipeline.py (~8 cells)

Cells:
1. Setup
2. Config: service name, compute pool
3. Session + model
4. Create SPCS inference service (gated)
5. Test inference on sample data
6. Batch predict on FeatureView data
7. Save predictions to prediction_table
8. Summary: row counts, service endpoint

Exports: `get_predictions_table`

### 05_monitoring_pipeline.py (~6 cells)

Cells:
1. Setup
2. Config: refresh interval, aggregation window
3. Session + predictions table
4. Create baseline table (first 50% by timestamp)
5. Create ModelMonitor (gated)
6. Summary: monitor status

### orchestrator.py (~12 cells)

Cells:
1. Setup: `import marimo as mo`
2. Config: full pipeline config (mo.ui.dictionary wrapping all params)
3. Session: shared session for all stages
4. Feature pipeline tab: calls 01's exported functions
5. Training pipeline tab: calls 02's exported functions
6. Promotion pipeline tab: calls 03's exported functions
7. Serving pipeline tab: calls 04's exported functions
8. Monitoring pipeline tab: calls 05's exported functions
9. Full pipeline run button: runs all stages sequentially
10. Pipeline status dashboard: mo.ui.tabs with stage summaries
11. Run history: mo.md with timestamps, versions, metrics
12. Cleanup / audit

---

## Key Design Patterns

### Config Cascade
```
conf/defaults.py (dataclass defaults)
    ↓ overridden by
mo.ui elements in each notebook (interactive mode)
    ↓ overridden by
mo.cli_args() (script/CI mode)
```
In interactive mode, UI elements are pre-filled from defaults. In script mode, defaults are used directly (UI elements fall back to their initial values).

### Gating Expensive Operations
Every cell with Snowflake side-effects (FeatureView registration, @remote training, service deployment, monitor creation) is gated:
```python
run_button = mo.ui.run_button(label="Submit Training Job")
mo.stop(not run_button.value)
results = train_remote(...)
```

### Caching with mo.cache
Expensive but deterministic operations (loading data, preprocessing) use `@mo.cache`:
```python
@mo.cache
def load_and_preprocess(session, config):
    raw = load_data(session, config.database, config.schema)
    return preprocess(raw)
```
If config hasn't changed, the cached result is returned instantly.

### Cross-Notebook Imports
The orchestrator imports from sub-notebooks:
```python
from notebooks.01_feature_pipeline import get_training_dataset
from notebooks.02_training_pipeline import get_best_run_name
```
Each sub-notebook defines reusable functions via `@app.function` in a setup cell. These functions only reference setup-cell imports (no cell-level globals), making them portable.

---

## Implementation Phases

### Phase 1: Scaffold & Config
- [ ] Create directory structure
- [ ] Create conf/defaults.py with PipelineConfig dataclass
- [ ] Create pyproject.toml

### Phase 2: Library Modules
- [ ] Create lib/session.py
- [ ] Create lib/features.py (port from Blog 3 feature_engineering_fns.py)
- [ ] Create lib/modelling.py (port training logic from Blog 3 cells 28-29)
- [ ] Create lib/versioning.py (port from Blog 3 useful_fns.py)

### Phase 3: Notebooks (Stage by Stage)
- [ ] Create 01_feature_pipeline.py
- [ ] Create 02_training_pipeline.py
- [ ] Create 03_promotion_pipeline.py
- [ ] Create 04_serving_pipeline.py
- [ ] Create 05_monitoring_pipeline.py
- [ ] Create orchestrator.py

### Phase 4: Polish
- [ ] Test each notebook standalone: `marimo edit notebooks/0X_*.py`
- [ ] Test orchestrator: `marimo edit notebooks/orchestrator.py`
- [ ] Test script mode: `python notebooks/orchestrator.py`
- [ ] Test app mode: `marimo run notebooks/orchestrator.py`

---

## Comparison: All Four Approaches

| Aspect | Blog 3 (Jupyter) | Blog 4 (Framework) | Original Plan (Marimo Single) | This (Marimo Hybrid) |
|--------|------------------|--------------------|-----------------------------|---------------------|
| Files | 4 | ~20 | 4 | ~14 |
| Cells per file | 55 | N/A | 18 | 6-12 |
| Config | Hardcoded | parameters.yml | mo.ui in one file | dataclass + mo.ui per stage |
| Compute | @remote | submit_directory | @remote | @remote |
| Stages runnable independently | No | Yes (CLI) | No | Yes (each notebook) |
| Testable domain logic | No | Yes (pytest) | No | Yes (lib/ modules) |
| Git diffs | Painful | Clean | Clean | Clean |
| Interactive | Jupyter widgets | No | Full (mo.ui) | Full (mo.ui per stage) |
| Deployable as app | No | No | Yes | Yes (per stage or full) |
| Scalability ceiling | ~30 cells | Unlimited | ~40 cells | Unlimited |
