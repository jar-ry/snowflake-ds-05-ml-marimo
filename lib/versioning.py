import ast

from snowflake.snowpark import Session
from snowflake.ml.dataset import Dataset
from snowflake.ml._internal.exceptions import dataset_errors


def next_dataset_version(
    session: Session,
    dataset_name: str,
    schema_name: str | None = None,
) -> str:
    if schema_name is None:
        schema_name = session.get_current_schema()
    full_name = f"{session.get_current_database()}.{schema_name}.{dataset_name}"

    try:
        ds = Dataset.load(session=session, name=full_name)
        versions = ds.list_versions()
    except dataset_errors.DatasetNotExistError:
        return "V_1"

    if len(versions) == 0:
        return "V_1"

    nums = sorted(int(v.rsplit("_", 1)[-1]) for v in versions)
    return f"V_{nums[-1] + 1}"


def next_model_version(df, model_name: str) -> str:
    if "." in model_name:
        model_name = model_name.split(".")[-1]
    if df.empty or df[df["name"] == model_name].empty:
        return "V_1"

    list_of_lists = df["versions"].apply(ast.literal_eval)
    all_versions = [v for sublist in list_of_lists for v in sublist]
    nums = sorted(int(v.rsplit("_", 1)[-1]) for v in all_versions)
    return f"V_{nums[-1] + 1}"


def latest_dataset_version(
    session: Session,
    dataset_name: str,
) -> str:
    ds = Dataset.load(session=session, name=dataset_name)
    return str(ds.list_versions()[-1])
