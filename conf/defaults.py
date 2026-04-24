from dataclasses import dataclass, field


@dataclass
class SnowflakeConfig:
    connection_name: str = "JCHEN_AWS1"
    database: str = "RETAIL_REGRESSION_DEMO"
    schema: str = "DS"
    warehouse: str = "RETAIL_REGRESSION_DEMO_WH"
    warehouse_size: str = "MEDIUM"


@dataclass
class FeatureStoreConfig:
    schema: str = "FEATURE_STORE"
    entity_name: str = "CUSTOMER"
    join_keys: list[str] = field(default_factory=lambda: ["CUSTOMER_ID"])
    fv_name: str = "FV_PREPROCESS"
    fv_version: str = "V_1"
    refresh_freq: str = "60 minute"
    timestamp_col: str = "BEHAVIOR_UPDATED_AT"
    dataset_name: str = "TRAINING_DATASET"


@dataclass
class ModelRegistryConfig:
    schema: str = "MODELLING"
    model_name: str = "UC01_SNOWFLAKEML_RF_REGRESSOR_MODEL"
    experiment_name: str = "MY_EXPERIMENT"
    prod_schema: str = "PROD_SCHEMA"
    prod_alias: str = "PROD"


@dataclass
class HPOConfig:
    num_trials: int = 10
    metric: str = "mean_absolute_percentage_error"
    mode: str = "min"
    max_depth_choices: list[int] = field(default_factory=lambda: [1, 4, 6, 10])
    eta_choices: list[float] = field(default_factory=lambda: [0.01, 0.1, 0.8])
    n_estimators_choices: list[int] = field(default_factory=lambda: [10, 150, 500])
    subsample_choices: list[float] = field(default_factory=lambda: [0.5, 0.7, 1.0])
    reg_lambda_choices: list[float] = field(default_factory=lambda: [0.1, 1, 10])
    random_state: int = 42


@dataclass
class ComputeConfig:
    pool_name: str = "CUSTOMER_VALUE_MODEL_POOL_CPU"
    stage_name: str = "payload_stage"
    target_instances: int = 3


@dataclass
class ServingConfig:
    service_name: str = "customer_value_service_v4"
    ingress_enabled: bool = True
    gpu_requests: str | None = None


@dataclass
class MonitoringConfig:
    refresh_interval: str = "1 hour"
    aggregation_window: str = "1 day"
    background_warehouse: str = "COMPUTE_WH"
    prediction_table: str = "prediction_table"
    baseline_table: str = "baseline"
    segment_columns: list[str] = field(default_factory=lambda: ["GENDER"])


@dataclass
class FeatureColumns:
    target: str = "MONTHLY_CUSTOMER_VALUE"
    numerical: list[str] = field(default_factory=lambda: [
        "AGE", "TENURE_MONTHS", "AVG_ORDER_VALUE", "PURCHASE_FREQUENCY",
        "RETURN_RATE", "TOTAL_ORDERS", "ANNUAL_INCOME",
        "AVERAGE_ORDER_PER_MONTH", "DAYS_SINCE_LAST_PURCHASE",
        "DAYS_SINCE_SIGNUP", "EXPECTED_DAYS_BETWEEN_PURCHASES",
        "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE",
    ])
    categorical: list[str] = field(default_factory=lambda: ["GENDER"])
    ordinal: list[str] = field(default_factory=lambda: ["LOYALTY_TIER"])
    ordinal_order: list[str] = field(default_factory=lambda: ["low", "medium", "high"])
    id_col: str = "CUSTOMER_ID"

    @property
    def all_features(self) -> list[str]:
        return self.numerical + self.categorical + self.ordinal


@dataclass
class PipelineConfig:
    snowflake: SnowflakeConfig = field(default_factory=SnowflakeConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    model_registry: ModelRegistryConfig = field(default_factory=ModelRegistryConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    columns: FeatureColumns = field(default_factory=FeatureColumns)
