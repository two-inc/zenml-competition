"""Definition of the Deployment & Pipeline"""
import numpy as np
from evidently.model_profile import Profile
from zenml.logger import get_logger
from zenml.pipelines import pipeline
from zenml.steps import BaseParameters
from zenml.steps import Output
from zenml.steps import step

from src.util.settings import docker_settings

logger = get_logger(__name__)


class DeploymentTriggerConfig(BaseParameters):
    """Thresholds determining whether to trigger the deployment"""

    min_f1: float
    max_brier: float
    min_roc_auc: float
    min_pr_auc: float


@step(enable_cache=False)
def deployment_trigger(
    metrics: dict[str, str],
    report: Profile,
    config: DeploymentTriggerConfig,
) -> Output(deployment_trigger=np.bool):
    """Evaluates metric results and data drift reports to determine whether to deploy

    Args:
        metrics (dict[str, str]): Metrics computed on holdout set
        report (Profile): Evidently Profile
        config (DeploymentTriggerConfig): Threshold configuration

    Returns:
        np.bool: Deployment Decision
    """
    report_object = report.object()
    data_drift = report_object["data_drift"]["data"]["metrics"][
        "dataset_drift"
    ]
    logger.info(f"Data Drift Detected: {data_drift}")
    target_drift = report_object["data_drift"]["data"]["metrics"]["fraud"][
        "drift_detected"
    ]
    logger.info(f"Target Drift Detected: {data_drift}")
    f1_score = metrics.get("F1_Score", config.min_f1)
    brier_score = metrics.get("Brier Score", config.max_brier)
    roc_auc = metrics.get("ROC AUC", config.min_roc_auc)
    pr_auc = metrics.get("PR AUC", config.min_pr_auc)

    deployment_decision = all(
        (
            not data_drift,
            not target_drift,
            f1_score >= config.min_f1,
            roc_auc >= config.min_roc_auc,
            pr_auc >= config.min_pr_auc,
            brier_score <= config.max_brier,
        )
    )
    logger.info(f"Deployment Decision: {deployment_decision}")
    return deployment_decision


@pipeline(
    name="continuous_deployment_pipeline_5",
    enable_cache=True,
    settings={"docker": docker_settings},
)
def continuous_deployment_pipeline(
    baseline_data_importer,
    new_data_importer,
    data_combiner,
    transformer,
    trainer,
    evaluator,
    drift_detector,
    deployment_trigger,
    model_deployer,
):
    """Trains and deploys a model provided that conditions established by the deployment trigger are met

    Process:
        - Load the 'baseline' data, which is assumed to be the state of truth for the previous model
        - Load the 'new' data, which represents the most up-to-date data available for training.
        - Check for Data Drift with Evidently
        - Combine the new and baseline datasets
        - Preprocess the data and train a new model on the total data
        - Compute performance metrics on the holdout set for the model predictions
        - Evaluate the performance metric results and the results of the drift detection component
        - Deploy the model if the deployment trigger step is successful
    """
    df_baseline = baseline_data_importer()
    df_new = new_data_importer()
    drift_report, _ = drift_detector(
        reference_dataset=df_baseline, comparison_dataset=df_new
    )
    df = data_combiner(df_baseline, df_new)
    X_train, X_test, y_train, y_test = transformer(df)
    model = trainer(X_train, y_train)
    metrics = evaluator(X_test, y_test, model)
    deployment_decision = deployment_trigger(metrics, drift_report)
    model_deployer(deployment_decision, model)
