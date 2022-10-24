"""Defines Column Variables for "Synthetic data from a financial payment system" Dataset"""

DROP: list[str] = ["zipMerchant", "zipcodeOri"]

CATEGORICAL: list[str] = ["category", "gender", "age"]

TRANSACTION_NUMBERS: list[str] = [
    "customer_transaction_number",
    "merchant_transaction_number",
]

STEP: str = "step"
AMOUNT: str = "amount"
TARGET: str = 'fraud'

CUSTOMER_AMOUNT_MAV: list[str] = [
    "customer_amount_ma_total",
    "customer_amount_ma_10",
    "customer_amount_ma_5",
]
CUSTOMER_AMOUNT_MSTD: list[str] = [
    "customer_amount_mstd_total",
    "customer_amount_mstd_10",
    "customer_amount_mstd_5",
]

MERCHANT_AMOUNT_MAV: list[str] = [
    "merchant_amount_ma_total",
    "merchant_amount_ma_50",
    "merchant_amount_ma_10",
    "merchant_amount_ma_5",
    "merchant_amount_ma_3",
]

MERCHANT_AMOUNT_MSTD: list[str] = [
    "merchant_amount_mstd_total",
    "merchant_amount_mstd_50",
    "merchant_amount_mstd_10",
    "merchant_amount_mstd_5",
    "merchant_amount_mstd_3",
]

CATEGORY_AMOUNT_MAV: list[str] = [
    "category_amount_ma_total",
    "category_amount_ma_100",
    "category_amount_ma_10",
]

CATEGORY_AMOUNT_MSTD: list[str] = [
    "category_amount_mstd_total",
    "category_amount_mstd_100",
    "category_amount_mstd_10",
]

AMOUNT_MOVING_MAX: list[str] = [
    "merchant_amount_moving_max",
    "customer_amount_moving_max",
    "category_amount_moving_max",
]

MEAN_CATEGORY_AMOUNT_PREVIOUS_STEP: str = "mean_category_amount_previous_step"

FRAUD_COMMITED_MEAN: list[str] = [
    "customer_fraud_commited_mean",
    "merchant_fraud_commited_mean",
    "category_fraud_commited_mean",
]

NUMERICAL: list[str] = [
    STEP,
    AMOUNT,
    MEAN_CATEGORY_AMOUNT_PREVIOUS_STEP,
    *CUSTOMER_AMOUNT_MAV,
    *CUSTOMER_AMOUNT_MSTD,
    *MERCHANT_AMOUNT_MAV,
    *MERCHANT_AMOUNT_MSTD,
    *CATEGORY_AMOUNT_MAV,
    *CATEGORY_AMOUNT_MSTD,
    *AMOUNT_MOVING_MAX,
    *FRAUD_COMMITED_MEAN,
]

MODEL = [*NUMERICAL, *CATEGORICAL]
