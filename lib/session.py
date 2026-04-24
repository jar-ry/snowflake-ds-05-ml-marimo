import os

from snowflake.snowpark import Session
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore, CreationMode


def create_session(
    connection_name: str = "JCHEN_AWS1",
    database: str = "RETAIL_REGRESSION_DEMO",
    schema: str = "DS",
    warehouse: str = "RETAIL_REGRESSION_DEMO_WH",
) -> Session:
    name = os.getenv("SNOWFLAKE_CONNECTION_NAME") or connection_name
    session = Session.builder.configs({"connection_name": name}).create()
    session.sql_simplifier_enabled = True
    session.sql(f"USE DATABASE {database}").collect()
    session.sql(f"USE SCHEMA {schema}").collect()
    session.sql(f"USE WAREHOUSE {warehouse}").collect()
    return session


def create_model_registry(
    session: Session,
    database: str,
    schema: str = "MODELLING",
) -> Registry:
    cs = session.get_current_schema()
    try:
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}").collect()
    except Exception:
        pass
    mr = Registry(session=session, database_name=database, schema_name=schema)
    session.sql(f"USE SCHEMA {cs}").collect()
    return mr


def create_feature_store(
    session: Session,
    database: str,
    schema: str = "FEATURE_STORE",
    warehouse: str = "RETAIL_REGRESSION_DEMO_WH",
) -> FeatureStore:
    try:
        fs = FeatureStore(
            session, database, schema, warehouse,
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )
    except Exception:
        fs = FeatureStore(
            session, database, schema, warehouse,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
    return fs
