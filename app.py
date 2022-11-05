"""streamlit app"""
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from src.util import columns
from src.util.data_access import load_data
from src.util.preprocess import get_preprocessed_data

load_dotenv()

logger = logging.getLogger(__file__)


def main():
    """Defines the Streamlit Application Interface and Control Flow"""
    data = get_data()
    mean_fraud_rate = get_mean_value(data.fraud)

    with st.sidebar:
        st.markdown("### :rewind: Recreate a Historical Transaction")
        min_transaction = get_min_value(data.index)
        max_transaction = get_max_value(data.index)

        def change_number():
            st.session_state["rn"] = np.random.randint(
                min_transaction, max_transaction
            )
            return

        if "rn" not in st.session_state:
            change_number()

        transaction_number = st.number_input(
            "Transaction Number",
            min_value=min_transaction,
            max_value=max_transaction,
        )
        replicate_transaction = st.checkbox("Replicate Transaction")
        st.markdown("----------")
        st.markdown("### :game_die: Random Historical Transaction")
        random_transaction = st.button("Randomize", on_click=change_number)

        if replicate_transaction:
            transaction = data.loc[transaction_number, :]
        elif random_transaction:
            transaction_number = st.session_state["rn"]
            transaction = data.loc[transaction_number, :]
            st.write(f"Replicating Transaction #{transaction_number}")
        else:
            transaction = None
        st.markdown("----------------")

        st.markdown("### :point_right: Input Values")

        inputs = get_inputs(data, transaction)

        st.markdown("----------------")

        st.markdown("### :crystal_ball: Generate Prediction")

        if st.button("Predict"):
            try:
                prediction = get_prediction(inputs)
                if prediction > mean_fraud_rate:
                    st.error(
                        f":oncoming_police_car: Halt! The passed transaction has a {100*prediction:.2f}% probability of being fraudulent"
                    )
                else:
                    st.success(
                        f":tada: Success! The passed transaction has a {100*prediction:.2f}% probability of being fraudulent"
                    )
            except Exception as e:
                logger.error(e)
                st.error("An unknown error occurred, please try again later")

    st.title(":mag: Detecting Fraudulent Financial Transactions with ZenML")

    st.markdown("")

    st.markdown(
        """
    This Streamlit App provides a simple interface to interact with the model developed using the
    [Synthetic data from a financial payment system](https://www.kaggle.com/datasets/ealaxi/banksim1) dataset as part of the ZenML Month of MLOps Competition.

    The model and application were developed via the ZenML framework via two pipelines, the **Training Pipeline** and **Continuous Deployment Pipeline**, which will be described briefly below
    before outlining how to interact with this application.

    ### :bullettrain_side: Training Pipeline

    The Training Pipeline defines the end-to-end process of training the machine learning model to predict whether a given transaction is fraudulent or not.
    This pipeline is particularly useful compared to an ad-hoc workflow for its reproducibility and maintainability. The artifacts of each stage are automatically
    saved to the ZenML artifact storage, so we can revisit any model knowing exactly what data it was trained on. Furthermore, thanks to ZenML's integration
    seamless integration with other MLOps tools, we have integrated our pipeline with the MLFlow Experiment Tracker, giving us visibility on the performance and metadata
    of each run of our pipeline, and are able to run the pipeline as a sequence of pods on Kubernetes at the click of a button.

    ##### :inbox_tray: Importer

    - Responsible for importing the data from a Cloud Storage Bucket

    ##### :wrench: Transformer

    - Applies simple preprocessing techniques to clean up data entries
    - Creates moving averages, standard deviations & max amounts by product category, merchant and customer
    - Adds the moving merchant and customer transaction count (i.e. transaction number) to each transaction
    - Splits the data according to the `Step` feature, which denotes the simulated day on which transactions took place

    ##### :running: Trainer

    - Trains a [Histogram-based Gradient Boosting Classification Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) against the training data

    ##### :chart_with_upwards_trend: Evaluator

    - Tests model performance against the held out validation set via the following metrics, with the results tracked in our Experiment Tracker
        - [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
        - [PR AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
        - [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)
        - [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
        - [F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
        - [Brier Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss)
        - [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)

    By orchestrating all of the steps above, we are able to build a reproducible and maintainable ML pipeline using the ZenML framework,
    with automated output artifact storage, experiment tracking & remote orchestration easily configurable, if not already baked in!


    ### :recycle: Continuous Deployment Pipeline

    The Continuous Deployment Pipeline is responsible for a more ambitious task than the training pipeline, namely to train a model on some subset of data,
    and deploy that into production, provided that particular acceptance criteria are met regarding the quality of the newly trained model.

    In particular, we extend the training pipeline described above to include two additional steps:

    ##### :white_check_mark: Deployment Trigger

    - Evaluates whether the metrics computed on the trained model meet particular acceptance criteria for eventual deployment.
    - In our Dummy example, we expect any model to have obtained an F1 Score of >0.8

    ##### :rocket: Model Deployer

    - Provided that the deployment trigger has been triggered, the model trained upstream is then deployed to a dedicated API endpoint
    - In this application, the model has been deployed on a Kubernetes Cluster using [Seldon](https://www.seldon.io/)

    ### :question: Interacting with the Model

    You can submit pseudo-transactions to the model using the sidebar interface on this app. There are three ways of defining a transaction:

    ##### :rewind: Recreate a Historical Transaction

    Replicate an example from the training data to see with what probability the model believes the transaction to be fraudulent. This can be used as a baseline configuration
    from which you can manipulate the sliders to create a new, counterfactual transaction.

    ##### :game_die: Random Historical Transaction

    Randomly select a historical transaction from the data. This can be used as a baseline configuration
    from which you can manipulate the sliders to create a new, counterfactual transaction.

    ##### :point_right: Input Values

    Define the input values to the model for the transaction yourself. You can change each slider's values even if you have previously replicated a historical transaction explicitly,
    or randomly selected one.

    ##### :crystal_ball: Predict

    After defining the specific attributes of your transaction, push the `Predict` button to fire off a request to the deployed model, and see how it responds!

    """
    )


@st.cache
def get_data() -> pd.DataFrame:
    """Retrieves the 'Synthetic data from financial payment system' data such that historical transactions can be replicated within the App"""
    data = load_data()
    return get_preprocessed_data(data)


def get_inputs(
    data: pd.DataFrame, transaction: Optional[pd.Series]
) -> list[st.slider]:
    """Retrieves the input controls, pre-configured with values of a particular transaction, if supplied

    Args:
        data (pd.DataFrame): Data
        transaction (Optional[pd.Series]): Transaction to be replicated as the default value of the slider
    """

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
    """Helper function to quickly return minimum values"""
    return data.min()


@st.cache
def get_max_value(data):
    """Helper function to quickly return maximum values"""
    return data.max()


@st.cache
def get_mean_value(data):
    """Helper function to quickly return mean values"""
    return data.mean()


def get_prediction(data: list[list[float]]) -> float:
    model_uri = os.environ.get("MODEL_ENDPOINT_URI")
    data_json = {"data": {"ndarray": data}}
    response = requests.post(model_uri, json=data_json)
    return response.json()["data"]["ndarray"][0][-1]


if __name__ == "__main__":
    main()
