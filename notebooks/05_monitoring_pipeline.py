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
    import snowflake.snowpark.functions as F
    from snowflake.ml.monitoring.entities.model_monitor_config import (
        ModelMonitorConfig,
        ModelMonitorSourceConfig,
    )


@app.cell
def _():
    _defaults = PipelineConfig()
    sf_connection = mo.ui.text(value=_defaults.snowflake.connection_name, label="Connection")
    sf_database = mo.ui.text(value=_defaults.snowflake.database, label="Database")
    sf_schema = mo.ui.text(value=_defaults.snowflake.schema, label="Schema")
    sf_warehouse = mo.ui.text(value=_defaults.snowflake.warehouse, label="Warehouse")
    mr_schema = mo.ui.text(value=_defaults.model_registry.schema, label="Registry Schema")
    model_name = mo.ui.text(value=_defaults.model_registry.model_name, label="Model Name")
    prediction_table = mo.ui.text(value=_defaults.monitoring.prediction_table, label="Prediction Table")
    baseline_table = mo.ui.text(value=_defaults.monitoring.baseline_table, label="Baseline Table")
    refresh_interval = mo.ui.text(value=_defaults.monitoring.refresh_interval, label="Refresh Interval")
    aggregation_window = mo.ui.text(value=_defaults.monitoring.aggregation_window, label="Aggregation Window")
    bg_warehouse = mo.ui.text(value=_defaults.monitoring.background_warehouse, label="Background WH")

    mo.md(f"""
    ## Monitoring Configuration
    {mo.hstack([sf_connection, sf_database, sf_schema, sf_warehouse])}
    {mo.hstack([mr_schema, model_name, prediction_table, baseline_table])}
    {mo.hstack([refresh_interval, aggregation_window, bg_warehouse])}
    """)
    return (
        sf_connection, sf_database, sf_schema, sf_warehouse,
        mr_schema, model_name, prediction_table, baseline_table,
        refresh_interval, aggregation_window, bg_warehouse,
    )


@app.cell
def _(sf_connection, sf_database, sf_schema, sf_warehouse, mr_schema):
    session = create_session(
        connection_name=sf_connection.value,
        database=sf_database.value,
        schema=sf_schema.value,
        warehouse=sf_warehouse.value,
    )
    mr = create_model_registry(session, sf_database.value, mr_schema.value)
    return session, mr


@app.cell
def _():
    baseline_button = mo.ui.run_button(label="Create Baseline Table")
    baseline_button
    return (baseline_button,)


@app.cell
def _(baseline_button, session, prediction_table, baseline_table):
    mo.stop(not baseline_button.value)

    _df = session.table(prediction_table.value)
    _total_rows = _df.count()
    _baseline = _df.order_by(F.col("BEHAVIOR_UPDATED_AT")).limit(_total_rows // 2)
    _baseline.write.mode("overwrite").save_as_table(baseline_table.value)
    _baseline_count = session.table(baseline_table.value).count()

    mo.md(f"### Baseline table `{baseline_table.value}` created ({_baseline_count} rows from {_total_rows} total)")
    return


@app.cell
def _():
    monitor_button = mo.ui.run_button(label="Create Model Monitor")
    monitor_button
    return (monitor_button,)


@app.cell
def _(
    monitor_button, session, mr, model_name, sf_database, sf_schema,
    prediction_table, baseline_table, refresh_interval, aggregation_window, bg_warehouse, mr_schema,
):
    mo.stop(not monitor_button.value)

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
    mo.md(f"### ModelMonitor `{model_name.value}_monitor` created")
    return


if __name__ == "__main__":
    app.run()
