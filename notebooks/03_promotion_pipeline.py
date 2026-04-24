import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conf.defaults import PipelineConfig
    from lib.session import create_session, create_model_registry
    from lib.modelling import generate_train_val_set, FEATURE_COLS, TARGET_COL
    from lib.versioning import latest_dataset_version
    from snowflake.ml.data.data_connector import DataConnector
    from snowflake.ml.dataset import Dataset, load_dataset


@app.cell
def _():
    _defaults = PipelineConfig()
    sf_connection = mo.ui.text(value=_defaults.snowflake.connection_name, label="Connection")
    sf_database = mo.ui.text(value=_defaults.snowflake.database, label="Database")
    sf_warehouse = mo.ui.text(value=_defaults.snowflake.warehouse, label="Warehouse")
    mr_schema = mo.ui.text(value=_defaults.model_registry.schema, label="Registry Schema")
    model_name = mo.ui.text(value=_defaults.model_registry.model_name, label="Model Name")
    prod_alias = mo.ui.text(value=_defaults.model_registry.prod_alias, label="Production Alias")
    prod_schema = mo.ui.text(value=_defaults.model_registry.prod_schema, label="Production Schema")
    fs_schema = mo.ui.text(value=_defaults.feature_store.schema, label="FS Schema")
    dataset_name = mo.ui.text(value=_defaults.feature_store.dataset_name, label="Dataset")
    best_run_input = mo.ui.text(value="", label="Best Run Name (from training)")

    mo.md(f"""
    ## Promotion Configuration
    {mo.hstack([sf_connection, sf_database, sf_warehouse])}
    {mo.hstack([mr_schema, model_name, best_run_input])}
    {mo.hstack([prod_alias, prod_schema, fs_schema, dataset_name])}
    """)
    return (
        sf_connection, sf_database, sf_warehouse, mr_schema, model_name,
        prod_alias, prod_schema, best_run_input, fs_schema, dataset_name,
    )


@app.cell
def _(sf_connection, sf_database, sf_warehouse, mr_schema):
    session = create_session(
        connection_name=sf_connection.value,
        database=sf_database.value,
        warehouse=sf_warehouse.value,
    )
    mr = create_model_registry(session, sf_database.value, mr_schema.value)
    return session, mr


@app.cell
def _(mr, model_name, best_run_input):
    mo.stop(not best_run_input.value, mo.md("*Enter the best run name above to continue.*"))
    model_object = mr.get_model(model_name.value)
    model_versions = model_object.show_versions()
    best_version = model_object.version(best_run_input.value)
    _best_df = model_versions[model_versions["name"] == best_run_input.value]
    mo.ui.table(_best_df, label="Best Model Version")
    return model_object, best_version


@app.cell
def _():
    promote_button = mo.ui.run_button(label="Promote to Production")
    promote_button
    return (promote_button,)


@app.cell
def _(promote_button, model_object, best_version, best_run_input, mr, sf_database, prod_alias, prod_schema, model_name):
    mo.stop(not promote_button.value)

    model_object.default = best_run_input.value

    try:
        best_version.set_alias(prod_alias.value)
    except Exception:
        pass

    session.sql(
        f"CREATE OR REPLACE TAG {mr._database_name}.{mr._schema_name}.live_model_version;"
    ).collect()
    model_object.set_tag(
        f"{mr._database_name}.{mr._schema_name}.live_model_version",
        best_run_input.value,
    )

    session.sql(f"CREATE OR REPLACE SCHEMA {sf_database.value}.{prod_schema.value};")
    session.sql(f"""
    CREATE OR REPLACE MODEL {sf_database.value}.{prod_schema.value}.{model_name.value}
    WITH VERSION {best_run_input.value}
    FROM MODEL {mr._database_name}.{mr._schema_name}.{model_name.value}
    VERSION {best_run_input.value};
    """)

    mo.md(f"""
    ### Promotion Complete
    - Default version set to `{best_run_input.value}`
    - Alias `{prod_alias.value}` applied
    - Tag `live_model_version` set
    - Model copied to `{prod_schema.value}`
    """)
    return


@app.cell
def _():
    explain_button = mo.ui.run_button(label="Generate SHAP Explanations")
    explain_button
    return (explain_button,)


@app.cell
def _(explain_button, session, sf_database, fs_schema, dataset_name, best_version):
    mo.stop(not explain_button.value)

    _source = f"{sf_database.value}.{fs_schema.value}.{dataset_name.value}"
    _ds = Dataset.load(session=session, name=_source)
    _ds_version = str(_ds.list_versions()[-1])
    _dc = DataConnector.from_dataset(load_dataset(session, _source, _ds_version))
    _df = _dc.to_pandas()
    _train_df, _ = generate_train_val_set(_df)
    _X_explain = _train_df.drop(
        columns=[TARGET_COL, "CUSTOMER_ID", "ASOF_DATE", "COL_1"], errors="ignore"
    ).head(100)

    explanations = best_version.run(_X_explain, function_name="explain")

    from snowflake.ml.monitoring.explain_visualize import plot_violin
    _X_encoded = pd.get_dummies(_X_explain, columns=["GENDER"])
    _fig = plot_violin(shap_df=explanations, feature_df=_X_encoded)

    mo.md("### SHAP Explanations")
    return (explanations,)


@app.function
def get_promoted_model():
    return model_object


if __name__ == "__main__":
    app.run()
