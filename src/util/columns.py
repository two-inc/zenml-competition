"""Defines Column Variables for "Synthetic data from a financial payment system" Dataset"""

DROP: list[str] = ["zipMerchant", "zipcodeOri"]

CATEGORICAL: list[str] = ["category", "gender", "age"]

TRANSACTION_NUMBERS: list[str] = [
    "customer_transaction_number",
    "merchant_transaction_number",
]

STEP: str = "step"
AMOUNT: str = "amount"
TARGET: str = "fraud"

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
    "merchant_fraud_commited_mean",
    "category_fraud_commited_mean",
]

NUMERICAL: list[str] = [
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


USER_FRIENDLY_COLUMN_NAMES: dict[str, str] = {
    "customer_amount_mstd_total": "Customer's Moving Transaction Amount Standard Deviation (Total)",
    "customer_amount_mstd_10": "Customer's Moving Transaction Amount Standard Deviation (Last 10 Transactions)",
    "customer_amount_mstd_5": "Customer's Moving Transaction Amount Standard Deviation (Last 5 Transactions)",
    "step": "Simulated Day",
    "amount": "Transaction Amount",
    "customer_fraud_commited_mean": "Customer's Moving Proportion of Fraudulent Transactions",
    "merchant_fraud_commited_mean": "Merchant's Moving Proportion of Fraudulent Transactions",
    "category_fraud_commited_mean": "Industry's Moving Proportion of Fraudulent Transactions",
    "merchant_amount_moving_max": "Merchant's Moving Maximum Transaction Amount",
    "customer_amount_moving_max": "Customer's Moving Maximum Transaction Amount",
    "category_amount_moving_max": "Industry's Moving Maximum Transaction Amount",
    "mean_category_amount_previous_step": "Industry's Mean Transaction Amount the Previous Day",
    "category_amount_mstd_total": "Industry's Moving Transaction Amount Standard Deviation (Total)",
    "category_amount_mstd_100": "Industry's Moving Transaction Amount Standard Deviation (Last 100 Transactions)",
    "category_amount_mstd_10": "Industry's Moving Transaction Amount Standard Deviation (Last 10 Transactions)",
    "category_amount_ma_total": "Industry's Moving Transaction Amount Average (Total)",
    "category_amount_ma_100": "Industry's Moving Transaction Amount Average (Last 100 Transactions)",
    "category_amount_ma_10": "Industry's Moving Transaction Amount Average (Last 10 Transactions)",
    "merchant_amount_mstd_total": "Merchants's Moving Transaction Amount Standard Deviation (Total)",
    "merchant_amount_mstd_50": "Merchants's Moving Transaction Amount Standard Deviation (Last 50 Transactions)",
    "merchant_amount_mstd_10": "Merchants's Moving Transaction Amount Standard Deviation (Last 10 Transactions)",
    "merchant_amount_mstd_5": "Merchants's Moving Transaction Amount Standard Deviation (Last 5 Transactions)",
    "merchant_amount_mstd_3": "Merchants's Moving Transaction Amount Standard Deviation (Last 3 Transactions)",
    "merchant_amount_ma_total": "Merchant's Moving Transaction Amount Average (Total)",
    "merchant_amount_ma_50": "Merchant's Moving Transaction Amount Average (Last 50 Transactions)",
    "merchant_amount_ma_10": "Merchant's Moving Transaction Amount Average (Last 10 Transactions)",
    "merchant_amount_ma_5": "Merchant's Moving Transaction Amount Average (Last 5 Transactions)",
    "merchant_amount_ma_3": "Merchant's Moving Transaction Amount Average (Last 3 Transactions)",
    "customer_amount_ma_10": "Customer's Moving Transaction Amount Average (Last 10 Transactions)",
    "customer_amount_ma_5": "Customer's Moving Transaction Amount Average (Last 5 Transactions)",
    "customer_amount_ma_total": "Customer's Moving Transaction Amount Average (total)",
}
