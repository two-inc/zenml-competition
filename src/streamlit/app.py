"""streamlit app"""
import json
from functools import lru_cache
from typing import cast
from typing import Optional

import streamlit as st
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.logger import get_logger

from src.util import columns
from src.util.data_access import load_data
from src.util.preprocess import get_preprocessed_data

logger = get_logger(__name__)


def main():
    """Main function for streamlit"""
    data = get_data()
    mean_fraud_rate = get_mean_value(data.fraud)
    service = get_service()

    with st.sidebar:
        st.title("Select Model Parameter Values")
        transaction_number = st.number_input(
            "Transaction Number",
            min_value=get_min_value(data.index),
            max_value=get_max_value(data.index),
        )
        if st.checkbox("Replicate Transaction"):
            transaction = data.loc[transaction_number, :]
        else:
            transaction = None
        inputs = get_inputs(data, transaction)

    st.title("Detecting Fraudulent Financial Transactions")

    st.markdown("")

    st.markdown(
        """
    - Order Amount:
    - Category:
    - Gender: Male, Female, Enterprise, Unknown
    - Step: TBC
    """
    )

    if st.button("Predict"):
        try:
            predict_format = json.dumps(inputs)
            prediction = service.predict(predict_format)
            y_pred_proba = prediction["data"]["ndarray"][-1][-1]
            if y_pred_proba > mean_fraud_rate:
                st.error(
                    f"The passed transaction has a {100*y_pred_proba:.2f}% probability of being fraudulent"
                )
            else:
                st.success(
                    f"The passed transaction has a {100*y_pred_proba:.2f}% probability of being fraudulent"
                )
        except Exception as e:
            logger.error(e)
            st.error("An unknown error occurred, please try again later")


@st.cache
def get_data():
    data = load_data()
    return get_preprocessed_data(data)


def get_inputs(data: pd.DataFrame, transaction: Optional[pd.Series]):
    def get_slider(data: pd.DataFrame, col: str):
        d = data.loc[:, col]
        min = get_min_value(d)
        max = get_max_value(d)
        if transaction is not None:
            value = get_mean_value(transaction[col])
        else:
            value = get_mean_value(d)
        t = int if d.dtype == int else float
        return st.slider(
            col, min_value=t(min), max_value=t(max), value=t(value)
        )

    return [[get_slider(data, col) for col in columns.NUMERICAL]]


@st.cache
def get_min_value(data):
    return data.min()


@st.cache
def get_max_value(data):
    return data.max()


@st.cache
def get_mean_value(data):
    return data.mean()


@st.cache
def get_service():
    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline_3",
        pipeline_step_name="seldon_model_deployer_step",
        model_name="model",
    )
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
            raise ValueError

    else:
        print(
            "No Seldon prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )
        return None

    return service


if __name__ == "__main__":
    main()
