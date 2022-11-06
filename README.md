# :shinto_shrine: Two ZenML Competition Solution

This repository contains Two's Solution to the ZenML Month of MLOps Competition.

The aim of this project is to develop a production-ready ML application for fraud detection using the ZenML MLOps framework. To train our fraud detection model, we make use of the ["Synthetic data from a financial payment system"](https://www.kaggle.com/datasets/ealaxi/banksim1) Dataset available on Kaggle.


## :memo: Solution Overview
This repository contains an end-to-end ML solution using ZenML, which covers the following responsiblities:
- Importing the Dataset
- Cleaning the data & engineering informative features
- Detecting data drift of new data
- Training a model to detect fraud on a transactional level
- Evaluating the performance of the model
- Deploying the model to a REST API endpoint
- Providing an interface for users to interact with the model

To address these requirements, we built a [Training Pipeline](src/pipelines/train_pipeline.py), which we used for experimentation, and a [Continuous Deployment Pipeline](src/pipelines/deployment_pipeline.py), which extended the capabilities of the Training Pipeline to identify data drift in new data, train a model on all available data, and evaluate the performance of this model prior to deploying this to an API endpoint.

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

## Usage

There are a number of ways of interacting with the code in this repository:
1. Executing the Training & Continuous Deployment Pipelines
2. Running the Streamlit App
3. Running the Tests

##### Executing the Training & Continuous Deployment Pipelines

1. Ensure you have Python 3.9 installed on your machine

2. Install the development requirements:
```
~ $ pip install -r test-requirements.txt
```
3. Deploy and register the ZenML stack described in the Solution Overview

4. Create an `.env` file from the `.env.example` template

5. To execute the train pipeline:
```
~ $ python src/run_train_pipeline.py
```

6. To execute the deployment pipeline:
```
~ $ python src/run_deployment_pipeline.py
```

##### Running the Streamlit App

The Streamlit application entrypoint is the `app.py` file at the root of the repository. We have deployed this app to [Streamlit Cloud](https://two-inc-zenml-competition-app-staging-banb63.streamlit.app/).

To recreate the app on your local machine, you must:

1. Ensure you have Python 3.9 installed on your machine

2. Install the Streamlit requirements:
```
~ $ pip install -r requirements.txt
```

3. Create an `.env` file according to the `.env.example` template

4. Deploy the Streamlit application
```
~ $ streamlit run app.py
```

##### Running the Tests

1. Ensure you have Python 3.9 installed on your machine

2. Install the test requirements:
```
~ $ pip install -r test-requirements.txt
```

3. Execute tests using `pytest`
```
~ $ pytest
```



## Repository Structure
```
├── .github				<- CI Pipeline Definition
├── src
│   ├── pipelines			<- Pipeline Definition
│   │   ├── ...
│   ├── steps		  		<- Step Definitons
│   │   ├── ...
│   ├── util		 		<- Utility Definitions
│   │   ├── ...
│   ├── data_exploration.ipynb		<- Data Exploration Notebook
│   ├── feature_engineering.ipynb	<- Feature Engineering Experimentation Notebook
│   ├── run_deployment_pipeline.py	<- Deployment Pipeline Execution script
│   ├── run_train_pipeline.py		<- Training Pipeline Execution Script
├── tests
│   ├── util				<- Utility Function Tests
│   │   ├── ...
├── app.py 	   			<- Streamlit App
├── docker-requirements.txt 		<- Step Container Dependencies
├── notebook-requirements.txt 		<- Notebook Dependencies
├── requirements.txt   			<- Streamlit App Dependencies
├── test-requirements.txt 		<- Development Dependencies

```

## :technologist: Competition Participants
- @dgjlindsay
- @joeolaide
- @JWorthington97
- @shelmigtwo
- @vangelis-two
