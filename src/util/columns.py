DROP: list[str] = ["zipMerchant", "zipcodeOri"]

CATEGORICAL: list[str] = ["category", "gender", "age"]
NUMERICAL: list[str] = [
    "customer_transaction_number",
    "merchant_transaction_number",
    "step",
    "amount",
    "merchant_amount_ma_total",
    "merchant_amount_ma_50",
    "merchant_amount_ma_10",
    "merchant_amount_ma_5",
    "merchant_amount_ma_3",
    "merchant_amount_moving_max",
    "customer_amount_ma_total",
    "customer_amount_ma_10",
    "customer_amount_ma_5",
    "customer_amount_moving_max",
    "category_amount_ma_total",
    "category_amount_ma_100",
    "category_amount_ma_10",
    "category_amount_moving_max",
    "mean_category_amount_previous_step",
    "customer_fraud_comitted_mean",
    "merchant_fraud_comitted_mean",
    "category_fraud_comitted_mean",
]

MODEL = [*NUMERICAL, *CATEGORICAL]
