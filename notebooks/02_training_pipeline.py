import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conf.defaults import PipelineConfig
    from lib.session import create_session, create_model_registry
    from lib.modelling import build_pipeline, evaluate_model, train, generate_train_val_set, FEATURE_COLS, TARGET_COL
    from lib.versioning import latest_dataset_version
    from snowflake.ml.data.data_connector import DataConnector
    from snowflake.ml.dataset import Dataset, load_dataset
    from snowflake.ml.registry import Registry


@app.cell
def _():
    _defaults = PipelineConfig()
    sf_connection = mo.ui.text(value=_defaults.snowflake.connection_name, label="Connection")
    sf_database = mo.ui.text(value=_defaults.snowflake.database, label="Database")
    sf_warehouse = mo.ui.text(value=_defaults.snowflake.warehouse, label="Warehouse")
    pool_name = mo.ui.text(value=_defaults.compute.pool_name, label="Compute Pool")
    stage_name = mo.ui.text(value=_defaults.compute.stage_name, label="Stage")
    target_instances = mo.ui.slider(1, 10, value=_defaults.compute.target_instances, label="Target Instances")
    num_trials = mo.ui.slider(1, 30, value=_defaults.hpo.num_trials, label="HPO Trials")
    model_name = mo.ui.text(
        value=f"{_defaults.model_registry.schema}.{_defaults.model_registry.model_name}",
        label="Model Name",
    )
    mr_schema = mo.ui.text(value=_defaults.model_registry.schema, label="Registry Schema")
    experiment_name = mo.ui.text(value=_defaults.model_registry.experiment_name, label="Experiment")
    fs_schema = mo.ui.text(value=_defaults.feature_store.schema, label="Feature Store Schema")
    dataset_name = mo.ui.text(value=_defaults.feature_store.dataset_name, label="Dataset Name")

    mo.md(f"""
    ## Training Configuration
    {mo.hstack([sf_connection, sf_database, sf_warehouse])}
    {mo.hstack([pool_name, stage_name, target_instances])}
    {mo.hstack([num_trials, model_name, mr_schema, experiment_name])}
    {mo.hstack([fs_schema, dataset_name])}
    """)
    return (
        sf_connection, sf_database, sf_warehouse, pool_name, stage_name,
        target_instances, num_trials, model_name, mr_schema, experiment_name,
        fs_schema, dataset_name,
    )


@app.cell
def _():
    _defaults = PipelineConfig()
    max_depth = mo.ui.multiselect(
        options=[1, 4, 6, 10],
        value=_defaults.hpo.max_depth_choices,
        label="max_depth",
    )
    eta = mo.ui.multiselect(
        options=[0.01, 0.1, 0.8],
        value=_defaults.hpo.eta_choices,
        label="eta",
    )
    n_estimators = mo.ui.multiselect(
        options=[10, 150, 500],
        value=_defaults.hpo.n_estimators_choices,
        label="n_estimators",
    )
    subsample = mo.ui.multiselect(
        options=[0.5, 0.7, 1.0],
        value=_defaults.hpo.subsample_choices,
        label="subsample",
    )
    reg_lambda = mo.ui.multiselect(
        options=[0.1, 1, 10],
        value=_defaults.hpo.reg_lambda_choices,
        label="reg_lambda",
    )

    mo.md(f"""
    ### HPO Search Space
    {mo.hstack([max_depth, eta, n_estimators, subsample, reg_lambda])}
    """)
    return max_depth, eta, n_estimators, subsample, reg_lambda


@app.cell
def _(sf_connection, sf_database, sf_warehouse):
    session = create_session(
        connection_name=sf_connection.value,
        database=sf_database.value,
        warehouse=sf_warehouse.value,
    )
    return (session,)


@app.cell
def _():
    submit_button = mo.ui.run_button(label="Submit Training Job")
    submit_button
    return (submit_button,)


@app.cell
def _(
    submit_button, session, sf_database, fs_schema, dataset_name,
    model_name, mr_schema, experiment_name,
    pool_name, stage_name, target_instances, num_trials,
    max_depth, eta, n_estimators, subsample, reg_lambda,
):
    mo.stop(not submit_button.value)

    from snowflake.ml.jobs import remote
    from snowflake.ml.modeling import tune
    from snowflake.ml.modeling.tune.search import RandomSearch

    _source_dataset = f"{sf_database.value}.{fs_schema.value}.{dataset_name.value}"

    def _create_data_connector(_session, _dataset_name):
        _ds = Dataset.load(session=_session, name=_dataset_name)
        _ds_version = str(_ds.list_versions()[-1])
        _ds_df = load_dataset(_session, _dataset_name, _ds_version)
        return DataConnector.from_dataset(_ds_df)

    @remote(pool_name.value, stage_name=stage_name.value, target_instances=target_instances.value)
    def _train_remote(source_dataset, model_name_arg, mr_schema_name, experiment_name_arg, session):
        _dc = _create_data_connector(session, dataset_name=source_dataset)
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
            "max_depth": tune.choice(max_depth.value),
            "eta": tune.choice(eta.value),
            "n_estimators": tune.choice(n_estimators.value),
            "subsample": tune.choice(subsample.value),
            "reg_lambda": tune.choice(reg_lambda.value),
            "random_state": tune.choice([42]),
        }

        _tuner_config = tune.TunerConfig(
            metric="mean_absolute_percentage_error",
            mode="min",
            search_alg=RandomSearch(),
            num_trials=num_trials.value,
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

    results = _train_remote(
        source_dataset=_source_dataset,
        model_name_arg=model_name.value,
        mr_schema_name=mr_schema.value,
        experiment_name_arg=experiment_name.value,
        session=session,
    )
    mo.md("### Training job submitted. Waiting for results...")
    return (results,)


@app.cell
def _(results):
    with mo.status.spinner("Waiting for training results..."):
        results.wait()
    results.show_logs()
    return


@app.cell
def _(results):
    import pandas as pd
    all_results = results.result()
    best_result = all_results.sort_values(by="mean_absolute_percentage_error", ascending=True).iloc[0]
    best_run_name = best_result["run_name"]

    mo.md(f"""
    ### Training Results
    **Best Run**: `{best_run_name}`
    | **MAE**: {best_result.get('mean_absolute_error', 'N/A'):.4f}
    | **MAPE**: {best_result.get('mean_absolute_percentage_error', 'N/A'):.4f}
    | **R2**: {best_result.get('r2_score', 'N/A'):.4f}
    """)

    mo.ui.table(all_results, label="All Trial Results")
    return all_results, best_run_name


@app.function
def get_training_results():
    return all_results


@app.function
def get_best_run_name():
    return best_run_name


if __name__ == "__main__":
    app.run()
