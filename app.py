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
        st.title(":computer: Model Interface")
        st.markdown("### :crystal_ball: Generate Prediction")

        predict = st.button("Predict")

        st.markdown("----------------")

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

    if predict:
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
    [Synthetic data from a financial payment system](https://www.kaggle.com/datasets/ealaxi/banksim1) dataset as part of the [ZenML](https://zenml.io/home) [Month of MLOps Competition](https://blog.zenml.io/mlops-competition/).

    The model is trained on a simulated dataset representing the transactions registered with a financial payment system provider. In particular, the model input features were engineered to provide the model with the necessary context on payment patterns associated with the customer, merchant and industry involved in a given transaction.

    ## :question: Interacting with the Model

    You can submit pseudo-transactions to the model using the sidebar interface on this app. There are three ways of defining a transaction:

    ##### :rewind: Recreate a Historical Transaction

    Replicate the feature inputs for an explicitly selected historical transaction.

    ##### :game_die: Random Historical Transaction

    Replicate the feature inputs for a randomly selected historical transaction.

    ##### :point_right: Input Values

    Define the input values to the model yourself. You can change each slider's values even if you have replicated a historical transaction, either explicitly or randomly.

    ##### :crystal_ball: Predict

    After defining the specific attributes of your transaction, push the `Predict` button to fire off a request to the deployed model, and see how it responds!

    ## :memo: How & Why We Made This

    At [Two](https://www.two.inc/), we make it a priority to keep our finger on the pulse of the ongoing developments in the MLOps space, as we recognize being able to develop, deploy and maintain sophisticated
    machine learning solutions is critical for the success of our business.

    We have been impressed by the framework developed by the ZenML team, and entered the Month of MLOps competition as part of our efforts to get properly acquainted with the framework and its capabilities.

    For our competition submission, we decided to implement a fraud detection model using ZenML, as we wanted to utilize the framework for a problem similar to the ones that our Data Science organization is tasked with
    addressing.

    In particular, we made use of the *[Synthetic data from a financial payment system](https://www.kaggle.com/datasets/ealaxi/banksim1)* dataset, made available by Kaggle. In line with the requirements of the competition, we began developing an end-to-end ML solution using ZenML, which was tasked with the following responsibilities:
    - Importing the Dataset
    - Cleaning the data & engineering informative features
    - Detecting data drift of new data
    - Training a model to detect fraud on a transactional level
    - Evaluating the performance of the model
    - Deploying the model to a REST API endpoint
    - Providing an interface for users to interact with the model

    To address these requirements, we built a Training Pipeline, which we used for experimentation, and a Continuous Deployment pipeline, which extended the capabilities of the Training Pipeline to identify data drift in
    new data, train a model on all available data, and evaluate the performance of this model prior to deploying this to an API endpoint.

    To enable the aforementioned pipelines, we made use of the following ZenML Stack:
    > **Artifact Storage**: [Google Cloud Storage](https://cloud.google.com/storage)
    >
    > **Container Registry**: [Google Cloud Container Registry](https://cloud.google.com/container-registry)
    >
    > **Data Validator**: [EvidentlyAI](https://www.evidentlyai.com/)
    >
    > **Experiment Tracker**: [MLFlow](https://mlflow.org/)
    >
    > **Orchestrator**: [Google Kuberenetes Engine](https://cloud.google.com/kubernetes-engine)
    >
    > **Model Deployer**: [Seldon](https://www.seldon.io/)

    We had a lot of fun implementing this solution using ZenML, and encourage readers to give the framework a try for themselves!


    ### :bullettrain_side: Training Pipeline

    The Training Pipeline defines the end-to-end process of training our model to predict whether a given transaction is fraudulent or not.

    This pipeline is particularly useful compared to an ad-hoc training workflow on account of its reproducibility and maintainability. The artifacts produced by each stage of the pipeline are automatically saved to the ZenML artifact storage, so we can revisit any model knowing exactly what data it was trained on.

    Furthermore, thanks to ZenML's infrastructure agnostic design, it was simple to integrate our pipeline with the MLFlow Experiment Tracker, giving us visibility on the performance and metadata of each run of our pipeline, and run the pipeline as a sequence of pods on Kubernetes.

    The Training Pipeline is composed of the following steps:

    ##### :inbox_tray: Baseline Data Importer

    - Responsible for importing the baseline data from a Cloud Storage Bucket
    - Baseline Data: A subset of our toy dataset to act as the "ground-truth" for the model development phase


    ##### :wrench: Transformer

    - Applies simple preprocessing techniques to clean up data entries
    - Creates moving averages, standard deviations & max amounts by product category, merchant and customer
    - Adds the moving merchant and customer transaction count (i.e. transaction number) to each transaction
    - Splits the data according to the `Step` feature, which denotes the simulated day on which transactions took place

    ##### :running: Trainer

    - Trains a [Histogram-based Gradient Boosting Classification Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) against the training data

    ##### :chart_with_upwards_trend: Evaluator

    - Tests model performance against the held out validation set and tracks the results in our MLFlow Experiment Tracker
    - Metrics used to evaluate performance
        - [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
        - [PR AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
        - [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)
        - [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
        - [F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
        - [Brier Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss)
        - [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)

    By orchestrating all of the steps above, we were able to build a reproducible and maintainable ML pipeline with automated output artifact storage, step caching, experiment tracking & remote orchestration baked in!


    ### :recycle: Continuous Deployment Pipeline

    Our Continuous Deployment Pipeline is responsible for a more ambitious task than the training pipeline, namely to train a model on some data,
    and deploy that into production, provided that particular acceptance criteria are met regarding the quality of the newly trained model.

    In particular, we extend the training pipeline described above to include four additional steps:

    ##### :newspaper: New Data Importer
    - This step imports an as-yet unseen slice of the toy dataset

    ##### :wavy_dash: Data Drift Detector
    - Implemented using the EvidentlyAI integration
    - This step compares the baseline data to the new data and identifies whether there has been significant data drift

    ##### :heavy_plus_sign: Data Combiner
    - Combines the "baseline" and "new" data to create a unified training dataset for the new model to be trained on

    ##### :white_check_mark: Deployment Trigger

    - Evaluates whether the metrics computed on the trained model meet particular acceptance criteria for eventual deployment
    - Additionally verifies that there has been no significant data drift between the baseline data and the newest batch


    ##### :rocket: Model Deployer

    - Provided that the deployment trigger has been triggered, the model trained upstream is then deployed to a dedicated API endpoint
    - In this application, the model has been deployed on a Kubernetes Cluster using Seldon

    With this pipeline architecture, it is trivial to update our model to the API endpoint exposed by Seldon, all the while ensuring that whichever model we deploy must meet our quality requirements.


    """
    )


@st.cache
def get_data() -> pd.DataFrame:
    """Retrieves the 'Synthetic data from financial payment system' data such that historical transactions can be replicated within the App"""
    data = load_data()
    return get_preprocessed_data(data)


MAX_VALUE_LIMITS = [0, 10, 50, 100, 500, 1_000, 5_000, 10_000]


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
        max_ = get_max_value(d)
        for limit in MAX_VALUE_LIMITS:
            if limit > max_:
                max_ = limit
                break
        if transaction is not None:
            value = get_mean_value(transaction[col])
        else:
            value = get_mean_value(d)
        t = int if d.dtype == int else float
        return st.slider(
            columns.USER_FRIENDLY_COLUMN_NAMES.get(col, col),
            min_value=t(0),
            max_value=t(max_),
            value=t(value),
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
