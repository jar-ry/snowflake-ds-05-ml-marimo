import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conf.defaults import PipelineConfig
    from lib.session import create_session, create_model_registry, create_feature_store
    import snowflake.snowpark.functions as F


@app.cell
def _():
    _defaults = PipelineConfig()
    sf_connection = mo.ui.text(value=_defaults.snowflake.connection_name, label="Connection")
    sf_database = mo.ui.text(value=_defaults.snowflake.database, label="Database")
    sf_schema = mo.ui.text(value=_defaults.snowflake.schema, label="Schema")
    sf_warehouse = mo.ui.text(value=_defaults.snowflake.warehouse, label="Warehouse")
    mr_schema = mo.ui.text(value=_defaults.model_registry.schema, label="Registry Schema")
    model_name = mo.ui.text(value=_defaults.model_registry.model_name, label="Model Name")
    service_name = mo.ui.text(value=_defaults.serving.service_name, label="Service Name")
    pool_name = mo.ui.text(value=_defaults.compute.pool_name, label="Compute Pool")
    fs_schema = mo.ui.text(value=_defaults.feature_store.schema, label="FS Schema")
    fv_name = mo.ui.text(value=_defaults.feature_store.fv_name, label="FeatureView")
    fv_version = mo.ui.text(value=_defaults.feature_store.fv_version, label="FV Version")

    mo.md(f"""
    ## Serving Configuration
    {mo.hstack([sf_connection, sf_database, sf_schema, sf_warehouse])}
    {mo.hstack([mr_schema, model_name, service_name, pool_name])}
    {mo.hstack([fs_schema, fv_name, fv_version])}
    """)
    return (
        sf_connection, sf_database, sf_schema, sf_warehouse,
        mr_schema, model_name, service_name, pool_name,
        fs_schema, fv_name, fv_version,
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
def _(mr, model_name):
    model_object = mr.get_model(model_name.value)
    _default_version = model_object.version("DEFAULT")
    mo.md(f"### Model `{model_name.value}` loaded (DEFAULT version)")
    return (model_object,)


@app.cell
def _():
    create_service_button = mo.ui.run_button(label="Create Inference Service")
    create_service_button
    return (create_service_button,)


@app.cell
def _(create_service_button, model_object, service_name, pool_name):
    mo.stop(not create_service_button.value)

    model_object.version("DEFAULT").create_service(
        service_name=service_name.value,
        service_compute_pool=pool_name.value,
        ingress_enabled=True,
        gpu_requests=None,
    )
    mo.md(f"### Service `{service_name.value}` created on pool `{pool_name.value}`")
    return


@app.cell
def _():
    predict_button = mo.ui.run_button(label="Run Batch Prediction")
    predict_button
    return (predict_button,)


@app.cell
def _(
    predict_button, model_object, service_name, session,
    sf_database, fs_schema, fv_name, fv_version, sf_schema,
):
    mo.stop(not predict_button.value)

    _fs = create_feature_store(session, sf_database.value, fs_schema.value, session.get_current_warehouse())
    _fv_data = _fs.get_feature_view(fv_name.value, fv_version.value).feature_df

    inference_result_sdf = model_object.version("DEFAULT").run(
        _fv_data,
        function_name="predict",
        service_name=service_name.value,
    ).with_column_renamed('"output_feature_0"', "PREDICTION")

    session.use_schema(sf_schema.value)
    inference_result_sdf.write.mode("overwrite").save_as_table("prediction_table")

    _count = session.table("prediction_table").count()
    mo.md(f"### Predictions saved to `prediction_table` ({_count} rows)")
    return (inference_result_sdf,)


@app.function
def get_predictions_table():
    return session.table("prediction_table")


if __name__ == "__main__":
    app.run()
