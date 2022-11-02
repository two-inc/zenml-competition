"""streamlit app"""

import numpy as np
import pandas as pd
import streamlit as st
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.logger import get_logger

from src.util.data_access import load_data

logger = get_logger(__name__)


def main():
    """Main function for streamlit"""
    data = load_data()

    with st.sidebar:
        step = st.number_input("Step", step=1, min_value=1, max_value=365)
        amount = st.slider("amount")
        mean_category_amount_previous_step = st.slider(
            "mean_category_amount_previous_step"
        )
        customer_amount_ma_total = st.slider("customer_amount_ma_total")
        customer_amount_ma_10 = st.slider("customer_amount_ma_10")
        customer_amount_ma_5 = st.slider("customer_amount_ma_5")
        customer_amount_mstd_total = st.slider("customer_amount_mstd_total")
        customer_amount_mstd_10 = st.slider("customer_amount_mstd_10")
        customer_amount_mstd_5 = st.slider("customer_amount_mstd_5")
        merchant_amount_ma_total = st.slider("merchant_amount_ma_total")
        merchant_amount_ma_50 = st.slider("merchant_amount_ma_50")
        merchant_amount_ma_10 = st.slider("merchant_amount_ma_10")
        merchant_amount_ma_5 = st.slider("merchant_amount_ma_5")
        merchant_amount_ma_3 = st.slider("merchant_amount_ma_3")
        merchant_amount_mstd_total = st.slider("merchant_amount_mstd_total")
        merchant_amount_mstd_50 = st.slider("merchant_amount_mstd_50")
        merchant_amount_mstd_10 = st.slider("merchant_amount_mstd_10")
        merchant_amount_mstd_5 = st.slider("merchant_amount_mstd_5")
        merchant_amount_mstd_3 = st.slider("merchant_amount_mstd_3")
        category_amount_ma_total = st.slider("category_amount_ma_total")
        category_amount_ma_100 = st.slider("category_amount_ma_100")
        category_amount_ma_10 = st.slider("category_amount_ma_10")
        category_amount_mstd_total = st.slider("category_amount_mstd_total")
        category_amount_mstd_100 = st.slider("category_amount_mstd_100")
        category_amount_mstd_10 = st.slider("category_amount_mstd_10")
        merchant_amount_moving_max = st.slider("merchant_amount_moving_max")
        customer_amount_moving_max = st.slider("customer_amount_moving_max")
        category_amount_moving_max = st.slider("category_amount_moving_max")
        customer_fraud_commited_mean = st.slider(
            "customer_fraud_commited_mean"
        )
        merchant_fraud_commited_mean = st.slider(
            "merchant_fraud_commited_mean"
        )
        category_fraud_commited_mean = st.slider(
            "category_fraud_commited_mean"
        )

    st.title("Detecting Fraudulent Financial Transactions")

    st.markdown("<Problem statement>")

    st.markdown(
        """
    - Order Amount:
    - Category:
    - Gender: Male, Female, Enterprise, Unknown
    - Step: TBC
    """
    )

    if st.button("Predict"):
        model_deployer = SeldonModelDeployer.get_active_model_deployer()

        services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline_3",
            pipeline_step_name="seldon_model_deployer_step",
            model_name="model",
        )
        print(services)
        if services:
            service = cast(SeldonDeploymentService, services[0])
            if service.is_running:
                print(
                    f"The Seldon prediction server is running remotely as a Kubernetes "
                    f"service and accepts inference requests at:\n"
                    f"    {service.prediction_url}\n"
                    f"To stop the service, run "
                    f"[italic green]`zenml model-deployer models delete "
                    f"{str(service.uuid)}`[/italic green]."
                )
            elif service.is_failed:
                print(
                    f"The Seldon prediction server is in a failed state:\n"
                    f" Last state: '{service.status.state.value}'\n"
                    f" Last error: '{service.status.last_error}'"
                )

        else:
            print(
                "No Seldon prediction server is currently running. The deployment "
                "pipeline must run first to train a model and deploy it. Execute "
                "the same command with the `--deploy` argument to deploy a model."
            )

        pred = [
            [
                step,
                amount,
                mean_category_amount_previous_step,
                customer_amount_ma_total,
                customer_amount_ma_10,
                customer_amount_ma_5,
                customer_amount_mstd_total,
                customer_amount_mstd_10,
                customer_amount_mstd_5,
                merchant_amount_ma_total,
                merchant_amount_ma_50,
                merchant_amount_ma_10,
                merchant_amount_ma_5,
                merchant_amount_ma_3,
                merchant_amount_mstd_total,
                merchant_amount_mstd_50,
                merchant_amount_mstd_10,
                merchant_amount_mstd_5,
                merchant_amount_mstd_3,
                category_amount_ma_total,
                category_amount_ma_100,
                category_amount_ma_10,
                category_amount_mstd_total,
                category_amount_mstd_100,
                category_amount_mstd_10,
                merchant_amount_moving_max,
                customer_amount_moving_max,
                category_amount_moving_max,
                customer_fraud_commited_mean,
                merchant_fraud_commited_mean,
                category_fraud_commited_mean,
            ]
        ]

        data = pd.Series(pred)

        # # prediction = service.predict()
        try:
            predict_format = data.to_json(orient="split")
            prediction = service.predict(predict_format)
            st.success(
                f"Given the customer's historical data, model says LEGITIMATE {prediction}"
            )
        except Exception as e:
            logger.error(e)
            st.error("An unknown error occurred, please try again later")


if __name__ == "__main__":
    main()
