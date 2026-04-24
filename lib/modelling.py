import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


FEATURE_COLS = [
    "AGE", "GENDER", "LOYALTY_TIER", "TENURE_MONTHS", "AVG_ORDER_VALUE",
    "PURCHASE_FREQUENCY", "RETURN_RATE", "TOTAL_ORDERS", "ANNUAL_INCOME",
    "AVERAGE_ORDER_PER_MONTH", "DAYS_SINCE_LAST_PURCHASE", "DAYS_SINCE_SIGNUP",
    "EXPECTED_DAYS_BETWEEN_PURCHASES", "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE",
]

TARGET_COL = "MONTHLY_CUSTOMER_VALUE"

NUMERICAL_COLS = [
    "AGE", "TENURE_MONTHS", "AVG_ORDER_VALUE", "PURCHASE_FREQUENCY",
    "RETURN_RATE", "TOTAL_ORDERS", "ANNUAL_INCOME", "AVERAGE_ORDER_PER_MONTH",
    "DAYS_SINCE_LAST_PURCHASE", "DAYS_SINCE_SIGNUP",
    "EXPECTED_DAYS_BETWEEN_PURCHASES", "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE",
]

CATEGORICAL_COLS = ["GENDER"]
ORDINAL_COLS = ["LOYALTY_TIER"]
TIER_ORDER = ["low", "medium", "high"]


def generate_train_val_set(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = dataframe[FEATURE_COLS]
    y = dataframe[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_test, y_test], axis=1)
    return train_df, val_df


def build_pipeline(model_params: dict) -> Pipeline:
    ordinal_encoder = OrdinalEncoder(
        categories=[TIER_ORDER], dtype=int,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("NUM", MinMaxScaler(), NUMERICAL_COLS),
            ("CAT", OneHotEncoder(), CATEGORICAL_COLS),
            ("ORD", ordinal_encoder, ORDINAL_COLS),
        ],
        remainder="passthrough",
    )
    model = xgb.XGBRegressor(**model_params)
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = model.predict(X_test)
    return {
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }


def train():
    from snowflake.ml.modeling import tune
    from snowflake.ml.experiment import ExperimentTracking
    from snowflake.snowpark.context import get_active_session

    session = get_active_session()
    tuner_context = tune.get_tuner_context()
    params = tuner_context.get_hyper_params()
    dm = tuner_context.get_dataset_map()
    model_name = params.pop("model_name")
    mr_schema_name = params.pop("mr_schema_name")
    experiment_name = params.pop("experiment_name")

    exp = ExperimentTracking(session=session, schema_name=mr_schema_name)
    exp.set_experiment(experiment_name)

    with exp.start_run() as run:
        train_data = dm["train"].to_pandas()
        val_data = dm["val"].to_pandas()

        X_train = train_data.drop(TARGET_COL, axis=1)
        y_train = train_data[TARGET_COL]
        X_val = val_data.drop(TARGET_COL, axis=1)
        y_val = val_data[TARGET_COL]

        model = build_pipeline(model_params=params)
        exp.log_params(params)

        print("Training model...", end="")
        model.fit(X_train, y_train)

        print("Evaluating model...", end="")
        metrics = evaluate_model(model, X_val, y_val)

        print("Log metrics...", end="")
        exp.log_metrics(metrics)
        metrics["run_name"] = run.name

        exp.log_model(
            model=model,
            model_name=model_name,
            version_name=run.name,
            sample_input_data=X_train,
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
            options={"enable_explainability": True},
        )

        tuner_context.report(metrics=metrics, model="model")
