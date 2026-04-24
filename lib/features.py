from datetime import datetime

from snowflake.snowpark import DataFrame, Session
import snowflake.snowpark.functions as F


def load_data(
    session: Session,
    database: str,
    schema: str,
) -> DataFrame:
    cust_tbl = f"{database}.{schema}.CUSTOMERS"
    behavior_tbl = f"{database}.{schema}.PURCHASE_BEHAVIOR"

    cust_sdf = session.table(cust_tbl)
    behavior_sdf = session.table(behavior_tbl)

    raw_data = cust_sdf.join(
        behavior_sdf,
        cust_sdf["CUSTOMER_ID"] == behavior_sdf["CUSTOMER_ID"],
        "left",
    ).rename({
        cust_sdf["UPDATED_AT"]: "CUSTOMER_UPDATED_AT",
        cust_sdf["CUSTOMER_ID"]: "CUSTOMER_ID",
        behavior_sdf["UPDATED_AT"]: "BEHAVIOR_UPDATED_AT",
    })

    return raw_data[[
        "CUSTOMER_ID", "AGE", "GENDER", "STATE", "ANNUAL_INCOME",
        "LOYALTY_TIER", "TENURE_MONTHS", "SIGNUP_DATE",
        "CUSTOMER_UPDATED_AT", "AVG_ORDER_VALUE", "PURCHASE_FREQUENCY",
        "RETURN_RATE", "MONTHLY_CUSTOMER_VALUE", "LAST_PURCHASE_DATE",
        "TOTAL_ORDERS", "BEHAVIOR_UPDATED_AT",
    ]]


def preprocess(data: DataFrame) -> DataFrame:
    data = data.with_column(
        "ANNUAL_INCOME", F.round(F.col("ANNUAL_INCOME"), 0)
    )
    data = data.with_columns(
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
    return data


def get_spine_df(feature_view) -> DataFrame:
    asof_date = datetime.now()
    spine_sdf = feature_view.feature_df.group_by("CUSTOMER_ID").agg(
        F.lit(asof_date.strftime("%Y-%m-%d %H:%M:%S")).as_("ASOF_DATE")
    )
    spine_sdf = spine_sdf.with_column("col_1", F.lit("values1"))
    return spine_sdf
