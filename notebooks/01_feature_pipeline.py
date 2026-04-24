import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conf.defaults import PipelineConfig
    from lib.session import create_session, create_model_registry, create_feature_store
    from datetime import datetime
    from lib.versioning import next_dataset_version
    from snowflake.ml.feature_store import FeatureView, Entity
    import snowflake.snowpark.functions as F


@app.cell
def _():
    _defaults = PipelineConfig()
    sf_database = mo.ui.text(value=_defaults.snowflake.database, label="Database")
    sf_schema = mo.ui.text(value=_defaults.snowflake.schema, label="Schema")
    sf_warehouse = mo.ui.text(value=_defaults.snowflake.warehouse, label="Warehouse")
    sf_connection = mo.ui.text(value=_defaults.snowflake.connection_name, label="Connection")
    fs_schema = mo.ui.text(value=_defaults.feature_store.schema, label="Feature Store Schema")
    fv_name = mo.ui.text(value=_defaults.feature_store.fv_name, label="FeatureView Name")
    fv_version = mo.ui.text(value=_defaults.feature_store.fv_version, label="FeatureView Version")
    refresh_freq = mo.ui.text(value=_defaults.feature_store.refresh_freq, label="Refresh Frequency")
    dataset_name = mo.ui.text(value=_defaults.feature_store.dataset_name, label="Dataset Name")

    mo.md(f"""
    ## Configuration
    {mo.hstack([sf_connection, sf_database, sf_schema, sf_warehouse])}
    {mo.hstack([fs_schema, fv_name, fv_version, refresh_freq, dataset_name])}
    """)
    return (
        dataset_name,
        fs_schema,
        fv_name,
        fv_version,
        refresh_freq,
        sf_connection,
        sf_database,
        sf_schema,
        sf_warehouse,
    )


@app.cell
def _(sf_connection, sf_database, sf_schema, sf_warehouse):
    session = create_session(
        connection_name=sf_connection.value,
        database=sf_database.value,
        schema=sf_schema.value,
        warehouse=sf_warehouse.value,
    )
    _env = session.sql("SELECT current_user(), current_version()").collect()
    mo.md(f"""
    ### Snowflake Session
    **User**: {_env[0][0]} | **Version**: {_env[0][1]}
    | **Database**: {session.get_current_database()}
    | **Warehouse**: {session.get_current_warehouse()}
    """)
    return (session,)


@app.cell
def _(fs_schema, session, sf_database, sf_warehouse):
    mr = create_model_registry(session, sf_database.value, "MODELLING")
    fs = create_feature_store(session, sf_database.value, fs_schema.value, sf_warehouse.value)
    mo.md("### Model Registry and Feature Store ready")
    return (fs,)


@app.cell
def _(fs):
    _entities = json.loads(
        fs.list_entities().select(F.to_json(F.array_agg("NAME", True))).collect()[0][0]
    )
    if "CUSTOMER" not in _entities:
        customer_entity = Entity(name="CUSTOMER", join_keys=["CUSTOMER_ID"], desc="Primary Key for CUSTOMER ORDER")
        fs.register_entity(customer_entity)
    else:
        customer_entity = fs.get_entity("CUSTOMER")
    mo.md(f"### Entity: CUSTOMER registered")
    return (customer_entity,)


@app.cell
def _(session, sf_database, sf_schema):
    _cust_tbl = f"{sf_database.value}.{sf_schema.value}.CUSTOMERS"
    _behavior_tbl = f"{sf_database.value}.{sf_schema.value}.PURCHASE_BEHAVIOR"

    _cust_sdf = session.table(_cust_tbl)
    _behavior_sdf = session.table(_behavior_tbl)

    _joined = _cust_sdf.join(
        _behavior_sdf,
        _cust_sdf["CUSTOMER_ID"] == _behavior_sdf["CUSTOMER_ID"],
        "left",
    ).rename({
        _cust_sdf["UPDATED_AT"]: "CUSTOMER_UPDATED_AT",
        _cust_sdf["CUSTOMER_ID"]: "CUSTOMER_ID",
        _behavior_sdf["UPDATED_AT"]: "BEHAVIOR_UPDATED_AT",
    })

    raw_data = _joined[[
        "CUSTOMER_ID", "AGE", "GENDER", "STATE", "ANNUAL_INCOME",
        "LOYALTY_TIER", "TENURE_MONTHS", "SIGNUP_DATE",
        "CUSTOMER_UPDATED_AT", "AVG_ORDER_VALUE", "PURCHASE_FREQUENCY",
        "RETURN_RATE", "MONTHLY_CUSTOMER_VALUE", "LAST_PURCHASE_DATE",
        "TOTAL_ORDERS", "BEHAVIOR_UPDATED_AT",
    ]]
    mo.ui.table(raw_data.limit(20).to_pandas(), label="Raw Data Preview")
    return (raw_data,)


@app.cell
def _(raw_data):
    _data = raw_data.with_column(
        "ANNUAL_INCOME", F.round(F.col("ANNUAL_INCOME"), 0)
    )
    preprocessed_data = _data.with_columns(
        [
            "AVERAGE_ORDER_PER_MONTH",
            "DAYS_SINCE_LAST_PURCHASE",
            "DAYS_SINCE_SIGNUP",
            "EXPECTED_DAYS_BETWEEN_PURCHASES",
            "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE",
        ],
        [
            F.col("TOTAL_ORDERS") / F.col("TENURE_MONTHS"),
            F.datediff("day", F.col("LAST_PURCHASE_DATE"), F.col("BEHAVIOR_UPDATED_AT")),
            F.datediff("day", F.col("SIGNUP_DATE"), F.col("BEHAVIOR_UPDATED_AT")),
            F.lit(30) / F.col("PURCHASE_FREQUENCY"),
            F.round(
                F.datediff("day", F.col("LAST_PURCHASE_DATE"), F.col("BEHAVIOR_UPDATED_AT"))
                - (F.lit(30) / F.col("PURCHASE_FREQUENCY")),
                0,
            ),
        ],
    )
    mo.ui.table(preprocessed_data.limit(20).to_pandas(), label="Preprocessed Data Preview")
    return (preprocessed_data,)


@app.cell
def _():
    register_fv_button = mo.ui.run_button(label="Register FeatureView")
    register_fv_button
    return (register_fv_button,)


@app.cell
def _(
    customer_entity,
    fs,
    fv_name,
    fv_version,
    preprocessed_data,
    refresh_freq,
    register_fv_button,
):
    mo.stop(not register_fv_button.value)

    _preprocess_features_desc = {
        "AVERAGE_ORDER_PER_MONTH": "Average number of orders per month",
        "DAYS_SINCE_LAST_PURCHASE": "Days since last purchase",
        "DAYS_SINCE_SIGNUP": "Days since signup",
        "EXPECTED_DAYS_BETWEEN_PURCHASES": "Expected days between purchases",
        "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE": "Days since expected last purchase date",
    }

    _fv_instance = FeatureView(
        name=fv_name.value,
        entities=[customer_entity],
        feature_df=preprocessed_data,
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
    mo.md(f"### FeatureView `{fv_name.value}` version `{fv_version.value}` registered")
    return (fv_registered,)


@app.cell
def _():
    generate_dataset_button = mo.ui.run_button(label="Generate Training Dataset")
    generate_dataset_button
    return (generate_dataset_button,)


@app.cell
def _(dataset_name, fs, fv_registered, generate_dataset_button, session):
    mo.stop(not generate_dataset_button.value)

    _asof_date = datetime.now()
    _spine_sdf = fv_registered.feature_df.group_by("CUSTOMER_ID").agg(
        F.lit(_asof_date.strftime("%Y-%m-%d %H:%M:%S")).as_("ASOF_DATE")
    )
    _spine_sdf = _spine_sdf.with_column("col_1", F.lit("values1"))
    _fv_schema = fs.list_feature_views().to_pandas()["SCHEMA_NAME"][0]
    _ds_version = next_dataset_version(session, dataset_name.value, schema_name=_fv_schema)

    training_dataset = fs.generate_dataset(
        name=dataset_name.value,
        version=_ds_version,
        spine_df=_spine_sdf,
        features=[fv_registered],
        spine_timestamp_col="ASOF_DATE",
    )
    training_dataset_sdf = training_dataset.read.to_snowpark_dataframe()
    _count = training_dataset_sdf.count()

    mo.md(f"""
    ### Training Dataset
    **Name**: `{dataset_name.value}` | **Version**: `{_ds_version}` | **Rows**: {_count}
    """)
    return (training_dataset_sdf,)


@app.cell
def _(session):
    def get_session():
        return session

    return


@app.cell
def _(fs):
    def get_feature_store():
        return fs

    return


@app.cell
def _(training_dataset_sdf):
    def get_training_dataset():
        return training_dataset_sdf

    return


if __name__ == "__main__":
    app.run()
