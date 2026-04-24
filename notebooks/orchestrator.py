import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import json
    import sys
    from pathlib import Path
    from datetime import datetime
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conf.defaults import PipelineConfig
    from lib.session import create_session, create_model_registry, create_feature_store
    from lib.features import load_data, preprocess, get_spine_df
    from lib.versioning import next_dataset_version
    from lib.modelling import build_pipeline, evaluate_model, train, generate_train_val_set, FEATURE_COLS, TARGET_COL
    from snowflake.ml.feature_store import FeatureView, Entity
    import snowflake.snowpark.functions as F
    from snowflake.ml.data.data_connector import DataConnector
    from snowflake.ml.dataset import Dataset, load_dataset
    from snowflake.ml.monitoring.entities.model_monitor_config import (
        ModelMonitorConfig,
        ModelMonitorSourceConfig,
    )



@app.cell
def _():
    is_interactive = mo.app_meta().mode != "script"
    _cli = mo.cli_args()
    _stages_arg = _cli.get("stages", "all")
    run_stages = set(range(1, 6)) if _stages_arg == "all" else {int(s) for s in _stages_arg.split(",")}
    mo.md("""
    # Customer Value Model — End-to-End Pipeline

    Five gated stages. Click each **Run** button to execute that stage.
    """)
    return is_interactive, run_stages


@app.cell
def _():
    _defaults = PipelineConfig()
    _args = mo.cli_args()

    sf_connection = mo.ui.text(value=_args.get("connection", _defaults.snowflake.connection_name), label="Connection")
    sf_database = mo.ui.text(value=_args.get("database", _defaults.snowflake.database), label="Database")
    sf_schema = mo.ui.text(value=_args.get("schema", _defaults.snowflake.schema), label="Schema")
    sf_warehouse = mo.ui.text(value=_args.get("warehouse", _defaults.snowflake.warehouse), label="Warehouse")

    fs_schema = mo.ui.text(value=_args.get("fs_schema", _defaults.feature_store.schema), label="FS Schema")
    fv_name = mo.ui.text(value=_args.get("fv_name", _defaults.feature_store.fv_name), label="FeatureView")
    fv_version = mo.ui.text(value=_args.get("fv_version", _defaults.feature_store.fv_version), label="FV Version")
    refresh_freq = mo.ui.text(value=_args.get("refresh_freq", _defaults.feature_store.refresh_freq), label="Refresh Freq")
    dataset_name = mo.ui.text(value=_args.get("dataset_name", _defaults.feature_store.dataset_name), label="Dataset")

    mr_schema = mo.ui.text(value=_args.get("mr_schema", _defaults.model_registry.schema), label="Registry Schema")
    model_name = mo.ui.text(value=_args.get("model_name", _defaults.model_registry.model_name), label="Model Name")
    experiment_name = mo.ui.text(value=_args.get("experiment_name", _defaults.model_registry.experiment_name), label="Experiment")
    prod_alias = mo.ui.text(value=_args.get("prod_alias", _defaults.model_registry.prod_alias), label="Prod Alias")
    prod_schema = mo.ui.text(value=_args.get("prod_schema", _defaults.model_registry.prod_schema), label="Prod Schema")

    pool_name = mo.ui.text(value=_args.get("pool_name", _defaults.compute.pool_name), label="Compute Pool")
    stage_name = mo.ui.text(value=_args.get("stage_name", _defaults.compute.stage_name), label="Stage")
    target_instances = mo.ui.slider(1, 10, value=int(_args.get("target_instances", _defaults.compute.target_instances)), label="Instances")
    num_trials = mo.ui.slider(1, 30, value=int(_args.get("num_trials", _defaults.hpo.num_trials)), label="HPO Trials")

    service_name = mo.ui.text(value=_args.get("service_name", _defaults.serving.service_name), label="Service Name")
    prediction_table = mo.ui.text(value=_args.get("prediction_table", _defaults.monitoring.prediction_table), label="Prediction Table")
    baseline_table = mo.ui.text(value=_args.get("baseline_table", _defaults.monitoring.baseline_table), label="Baseline Table")
    refresh_interval = mo.ui.text(value=_args.get("refresh_interval", _defaults.monitoring.refresh_interval), label="Monitor Refresh")
    aggregation_window = mo.ui.text(value=_args.get("aggregation_window", _defaults.monitoring.aggregation_window), label="Agg Window")
    bg_warehouse = mo.ui.text(value=_args.get("bg_warehouse", _defaults.monitoring.background_warehouse), label="Background WH")

    mo.md(f"""
    ## Configuration
    {mo.hstack([sf_connection, sf_database, sf_schema, sf_warehouse])}
    {mo.hstack([fs_schema, fv_name, fv_version, refresh_freq, dataset_name])}
    {mo.hstack([mr_schema, model_name, experiment_name, prod_alias, prod_schema])}
    {mo.hstack([pool_name, stage_name, target_instances, num_trials])}
    {mo.hstack([service_name, prediction_table, baseline_table])}
    {mo.hstack([refresh_interval, aggregation_window, bg_warehouse])}
    """)
    return (
        aggregation_window,
        baseline_table,
        bg_warehouse,
        dataset_name,
        experiment_name,
        fs_schema,
        fv_name,
        fv_version,
        model_name,
        mr_schema,
        num_trials,
        pool_name,
        prediction_table,
        prod_alias,
        prod_schema,
        refresh_freq,
        refresh_interval,
        service_name,
        sf_connection,
        sf_database,
        sf_schema,
        sf_warehouse,
        stage_name,
        target_instances,
    )


@app.cell
def _(is_interactive):
    connect_button = mo.ui.run_button(label="Connect to Snowflake")
    connect_button if is_interactive else None
    return (connect_button,)


@app.cell
def _(is_interactive, connect_button, sf_connection, sf_database, sf_schema, sf_warehouse):
    if is_interactive:
        mo.stop(not connect_button.value)
    session = create_session(
        connection_name=sf_connection.value,
        database=sf_database.value,
        schema=sf_schema.value,
        warehouse=sf_warehouse.value,
    )
    _env = session.sql("SELECT current_user(), current_version()").collect()
    mo.md(f"**Session**: {_env[0][0]} | {_env[0][1]} | {session.get_current_database()}.{session.get_current_schema()}")
    return (session,)


@app.cell
def _(fs_schema, mr_schema, session, sf_database, sf_warehouse):
    mr = create_model_registry(session, sf_database.value, mr_schema.value)
    fs = create_feature_store(session, sf_database.value, fs_schema.value, sf_warehouse.value)
    mo.md("Model Registry and Feature Store ready")
    return fs, mr


@app.cell
def _():
    mo.md("""
    ---
    ## Stage 1: Feature Pipeline
    """)
    return


@app.cell
def _(is_interactive):
    run_features_button = mo.ui.run_button(label="Run Feature Pipeline")
    run_features_button if is_interactive else None
    return (run_features_button,)


@app.cell
def _(
    is_interactive,
    run_stages,
    dataset_name,
    fs,
    fv_name,
    fv_version,
    refresh_freq,
    run_features_button,
    session,
    sf_database,
    sf_schema,
):
    if is_interactive:
        mo.stop(not run_features_button.value)
    elif 1 not in run_stages:
        mo.stop(True)

    _entities = json.loads(
        fs.list_entities().select(F.to_json(F.array_agg("NAME", True))).collect()[0][0]
    )
    if "CUSTOMER" not in _entities:
        _customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"], desc="Primary Key for CUSTOMER ORDER")
        fs.register_entity(_customer_entity)
    else:
        _customer_entity = fs.get_entity("CUSTOMER")

    _raw = load_data(session, sf_database.value, sf_schema.value)
    _preprocessed = preprocess(_raw)

    _preprocess_features_desc = {
        "AVERAGE_ORDER_PER_MONTH": "Average number of orders per month",
        "DAYS_SINCE_LAST_PURCHASE": "Days since last purchase",
        "DAYS_SINCE_SIGNUP": "Days since signup",
        "EXPECTED_DAYS_BETWEEN_PURCHASES": "Expected days between purchases",
        "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE": "Days since expected last purchase date",
    }

    _fv_instance = FeatureView(
        name=fv_name.value,
        entities=[_customer_entity],
        feature_df=_preprocessed,
        timestamp_col="BEHAVIOR_UPDATED_AT",
        refresh_freq=refresh_freq.value,
        desc="Customer Modelling Features",
    ).attach_feature_desc(_preprocess_features_desc)

    fv_registered = fs.register_feature_view(
        feature_view=_fv_instance,
        version=fv_version.value,
        block=True,
        overwrite=True,
    )

    _spine_sdf = get_spine_df(fv_registered)
    _fv_schema = fs.list_feature_views().to_pandas()["SCHEMA_NAME"][0]
    _ds_version = next_dataset_version(session, dataset_name.value, schema_name=_fv_schema)

    training_dataset = fs.generate_dataset(
        name=dataset_name.value,
        version=_ds_version,
        spine_df=_spine_sdf,
        features=[fv_registered],
        spine_timestamp_col="ASOF_DATE",
    )
    _count = training_dataset.read.to_snowpark_dataframe().count()

    mo.md(f"""
    ### Stage 1 Complete
    - Entity: CUSTOMER
    - FeatureView: `{fv_name.value}` v`{fv_version.value}`
    - Dataset: `{dataset_name.value}` v`{_ds_version}` ({_count} rows)
    """)
    return


@app.cell
def _():
    mo.md("""
    ---
    ## Stage 2: Training Pipeline (HPO via @remote)
    """)
    return


@app.cell
def _():
    _defaults = PipelineConfig()
    max_depth = mo.ui.multiselect(options=[1, 4, 6, 10], value=_defaults.hpo.max_depth_choices, label="max_depth")
    eta = mo.ui.multiselect(options=[0.01, 0.1, 0.8], value=_defaults.hpo.eta_choices, label="eta")
    n_estimators = mo.ui.multiselect(options=[10, 150, 500], value=_defaults.hpo.n_estimators_choices, label="n_estimators")
    subsample = mo.ui.multiselect(options=[0.5, 0.7, 1.0], value=_defaults.hpo.subsample_choices, label="subsample")
    reg_lambda = mo.ui.multiselect(options=[0.1, 1, 10], value=_defaults.hpo.reg_lambda_choices, label="reg_lambda")

    mo.md(f"""
    ### HPO Search Space
    {mo.hstack([max_depth, eta, n_estimators, subsample, reg_lambda])}
    """)
    return eta, max_depth, n_estimators, reg_lambda, subsample


@app.cell
def _(is_interactive):
    run_training_button = mo.ui.run_button(label="Submit Training Job")
    run_training_button if is_interactive else None
    return (run_training_button,)


@app.cell
def _(
    is_interactive,
    run_stages,
    dataset_name,
    eta,
    experiment_name,
    fs_schema,
    max_depth,
    model_name,
    mr_schema,
    n_estimators,
    num_trials,
    pool_name,
    reg_lambda,
    run_training_button,
    session,
    sf_database,
    stage_name,
    subsample,
    target_instances,
):
    if is_interactive:
        mo.stop(not run_training_button.value)
    elif 2 not in run_stages:
        mo.stop(True)

    from snowflake.ml.jobs import remote

    _source_dataset = f"{sf_database.value}.{fs_schema.value}.{dataset_name.value}"
    _fqn_model = f"{mr_schema.value}.{model_name.value}"
    _max_depth_choices = max_depth.value
    _eta_choices = eta.value
    _n_estimators_choices = n_estimators.value
    _subsample_choices = subsample.value
    _reg_lambda_choices = reg_lambda.value
    _num_trials_val = num_trials.value

    @remote(pool_name.value, stage_name=stage_name.value, target_instances=target_instances.value)
    def train_remote(source_dataset, model_name_arg, mr_schema_name, experiment_name_arg, session):
        from snowflake.ml.modeling import tune
        from snowflake.ml.modeling.tune.search import RandomSearch
        from snowflake.ml.registry import Registry
        from snowflake.ml.data.data_connector import DataConnector
        from snowflake.ml.dataset import Dataset, load_dataset

        _ds = Dataset.load(session=session, name=source_dataset)
        _ds_version = str(_ds.list_versions()[-1])
        _ds_df = load_dataset(session, source_dataset, _ds_version)
        _dc = DataConnector.from_dataset(_ds_df)
        _df = _dc.to_pandas()
        _train_df, _val_df = generate_train_val_set(_df)

        _dataset_map = {
            "train": DataConnector.from_dataframe(session.create_dataframe(_train_df)),
            "val": DataConnector.from_dataframe(session.create_dataframe(_val_df)),
        }

        _search_space = {
            "mr_schema_name": mr_schema_name,
            "model_name": model_name_arg,
            "experiment_name": experiment_name_arg,
            "max_depth": tune.choice(_max_depth_choices),
            "eta": tune.choice(_eta_choices),
            "n_estimators": tune.choice(_n_estimators_choices),
            "subsample": tune.choice(_subsample_choices),
            "reg_lambda": tune.choice(_reg_lambda_choices),
            "random_state": tune.choice([42]),
        }

        _tuner_config = tune.TunerConfig(
            metric="mean_absolute_percentage_error",
            mode="min",
            search_alg=RandomSearch(),
            num_trials=_num_trials_val,
        )

        _mr_schema = model_name_arg.split(".")[0] if "." in model_name_arg else mr_schema_name
        _mr_model = model_name_arg.split(".")[-1]
        _mr = Registry(session=session, database_name=session.get_current_database(), schema_name=_mr_schema)
        try:
            _mr.get_model(_mr_model)
        except Exception:
            from sklearn.linear_model import LinearRegression
            _dummy = LinearRegression().fit([[0]], [0])
            _mr.log_model(_dummy, model_name=_mr_model, version_name="dummy_version", sample_input_data=[[0]])

        _tuner = tune.Tuner(train_func=train, search_space=_search_space, tuner_config=_tuner_config)
        _results = _tuner.run(dataset_map=_dataset_map)
        return _results.results

    training_results = train_remote(
        source_dataset=_source_dataset,
        model_name_arg=_fqn_model,
        mr_schema_name=mr_schema.value,
        experiment_name_arg=experiment_name.value,
        session=session,
    )
    mo.md("### Training job submitted. Waiting for results...")
    return (training_results,)


@app.cell
def _(training_results):
    import pandas as pd
    with mo.status.spinner("Waiting for training results..."):
        training_results.wait()

    all_results = training_results.result()
    _best = all_results.sort_values(by="mean_absolute_percentage_error", ascending=True).iloc[0]
    best_run_name = _best["run_name"]

    mo.md(f"""
    ### Training Complete
    **Best Run**: `{best_run_name}`
    | MAE: {_best.get('mean_absolute_error', 'N/A'):.4f}
    | MAPE: {_best.get('mean_absolute_percentage_error', 'N/A'):.4f}
    | R2: {_best.get('r2_score', 'N/A'):.4f}
    """)
    mo.ui.table(all_results, label="All Trial Results")
    return (best_run_name,)


@app.cell
def _():
    mo.md("""
    ---
    ## Stage 3: Model Promotion
    """)
    return


@app.cell
def _(is_interactive):
    run_promotion_button = mo.ui.run_button(label="Promote Best Model")
    run_promotion_button if is_interactive else None
    return (run_promotion_button,)


@app.cell
def _(
    is_interactive,
    run_stages,
    best_run_name,
    dataset_name,
    fs_schema,
    model_name,
    mr,
    mr_schema,
    prod_alias,
    prod_schema,
    run_promotion_button,
    session,
    sf_database,
):
    if is_interactive:
        mo.stop(not run_promotion_button.value)
    elif 3 not in run_stages:
        mo.stop(True)

    _model_object = mr.get_model(model_name.value)
    _best_version = _model_object.version(best_run_name)

    _model_object.default = best_run_name

    try:
        _best_version.set_alias(prod_alias.value)
    except Exception:
        pass

    session.sql(
        f"CREATE OR REPLACE TAG {sf_database.value}.{mr_schema.value}.live_model_version;"
    ).collect()
    _model_object.set_tag(
        f"{sf_database.value}.{mr_schema.value}.live_model_version",
        best_run_name,
    )

    session.sql(f"CREATE OR REPLACE SCHEMA {sf_database.value}.{prod_schema.value};").collect()
    session.sql(f"""
    CREATE OR REPLACE MODEL {sf_database.value}.{prod_schema.value}.{model_name.value}
    WITH VERSION {best_run_name}
    FROM MODEL {sf_database.value}.{mr_schema.value}.{model_name.value}
    VERSION {best_run_name};
    """).collect()

    _source = f"{sf_database.value}.{fs_schema.value}.{dataset_name.value}"
    _ds = Dataset.load(session=session, name=_source)
    _ds_version = str(_ds.list_versions()[-1])
    _dc = DataConnector.from_dataset(load_dataset(session, _source, _ds_version))
    _df = _dc.to_pandas()
    _train_df, _ = generate_train_val_set(_df)
    _X_explain = _train_df.drop(
        columns=[TARGET_COL, "CUSTOMER_ID", "ASOF_DATE", "COL_1"], errors="ignore"
    ).head(100)
    explanations = _best_version.run(_X_explain, function_name="explain")

    mo.md(f"""
    ### Stage 3 Complete
    - Default: `{best_run_name}`
    - Alias: `{prod_alias.value}`
    - Tag: `live_model_version = {best_run_name}`
    - Copied to `{prod_schema.value}`
    - SHAP explanations generated ({len(explanations)} rows)
    """)
    return


@app.cell
def _():
    mo.md("""
    ---
    ## Stage 4: Serving (SPCS Inference + Batch Predict)
    """)
    return


@app.cell
def _(is_interactive):
    run_serving_button = mo.ui.run_button(label="Create Service & Run Batch Prediction")
    run_serving_button if is_interactive else None
    return (run_serving_button,)


@app.cell
def _(
    is_interactive,
    run_stages,
    fs_schema,
    fv_name,
    fv_version,
    model_name,
    mr,
    pool_name,
    run_serving_button,
    service_name,
    session,
    sf_database,
    sf_schema,
):
    if is_interactive:
        mo.stop(not run_serving_button.value)
    elif 4 not in run_stages:
        mo.stop(True)

    _model_object = mr.get_model(model_name.value)
    _model_object.version("DEFAULT").create_service(
        service_name=service_name.value,
        service_compute_pool=pool_name.value,
        ingress_enabled=True,
        gpu_requests=None,
    )

    _fs = create_feature_store(session, sf_database.value, fs_schema.value, session.get_current_warehouse())
    _fv_data = _fs.get_feature_view(fv_name.value, fv_version.value).feature_df

    _inference_result = _model_object.version("DEFAULT").run(
        _fv_data,
        function_name="predict",
        service_name=service_name.value,
    ).with_column_renamed('"output_feature_0"', "PREDICTION")

    session.use_schema(sf_schema.value)
    _inference_result.write.mode("overwrite").save_as_table("prediction_table")
    _count = session.table("prediction_table").count()

    mo.md(f"""
    ### Stage 4 Complete
    - Service: `{service_name.value}` on `{pool_name.value}`
    - Predictions: `prediction_table` ({_count} rows)
    """)
    return


@app.cell
def _():
    mo.md("""
    ---
    ## Stage 5: Monitoring (Baseline + ModelMonitor)
    """)
    return


@app.cell
def _(is_interactive):
    run_monitoring_button = mo.ui.run_button(label="Set Up Monitoring")
    run_monitoring_button if is_interactive else None
    return (run_monitoring_button,)


@app.cell
def _(
    is_interactive,
    run_stages,
    aggregation_window,
    baseline_table,
    bg_warehouse,
    model_name,
    mr,
    mr_schema,
    prediction_table,
    refresh_interval,
    run_monitoring_button,
    session,
    sf_database,
    sf_schema,
):
    if is_interactive:
        mo.stop(not run_monitoring_button.value)
    elif 5 not in run_stages:
        mo.stop(True)

    _pred_df = session.table(prediction_table.value)
    _total_rows = _pred_df.count()
    _baseline = _pred_df.order_by(F.col("BEHAVIOR_UPDATED_AT")).limit(_total_rows // 2)
    _baseline.write.mode("overwrite").save_as_table(baseline_table.value)
    _baseline_count = session.table(baseline_table.value).count()

    _model_object = mr.get_model(model_name.value)
    _fqn_pred = f"{sf_database.value}.{sf_schema.value}.{prediction_table.value}"
    _fqn_base = f"{sf_database.value}.{sf_schema.value}.{baseline_table.value}"

    _source_config = ModelMonitorSourceConfig(
        source=_fqn_pred,
        timestamp_column="BEHAVIOR_UPDATED_AT",
        id_columns=["CUSTOMER_ID"],
        prediction_score_columns=["PREDICTION"],
        actual_score_columns=["MONTHLY_CUSTOMER_VALUE"],
        segment_columns=["GENDER"],
        baseline=_fqn_base,
    )

    _monitor_config = ModelMonitorConfig(
        model_version=_model_object.version("DEFAULT"),
        model_function_name="predict",
        background_compute_warehouse_name=bg_warehouse.value,
        refresh_interval=refresh_interval.value,
        aggregation_window=aggregation_window.value,
    )

    session.use_schema(mr_schema.value)
    _monitor = mr.add_monitor(
        name=f"{model_name.value}_monitor",
        source_config=_source_config,
        model_monitor_config=_monitor_config,
    )

    mo.md(f"""
    ### Stage 5 Complete
    - Baseline: `{baseline_table.value}` ({_baseline_count} rows from {_total_rows})
    - Monitor: `{model_name.value}_monitor`
    """)
    return


@app.cell
def _():
    from zoneinfo import ZoneInfo
    _now = datetime.now(ZoneInfo("Australia/Melbourne"))
    mo.md(f"---\n**Last run**: {_now.strftime('%A, %B %d, %Y %I:%M:%S %p %Z')}")
    return


if __name__ == "__main__":
    app.run()
