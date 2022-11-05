"""CLI Entrypoint for the execution of the Seldon Deployment Pipeline"""
from typing import cast

import click
from rich import print
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentConfig
from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.integrations.seldon.steps import seldon_model_deployer_step
from zenml.integrations.seldon.steps import SeldonDeployerStepParameters
from zenml.logger import get_logger
from zenml.steps import BaseStep

from src.pipelines.deployment_pipeline import continuous_deployment_pipeline
from src.pipelines.deployment_pipeline import deployment_trigger
from src.pipelines.deployment_pipeline import DeploymentTriggerConfig
from src.steps.evaluator import evaluator
from src.steps.importer import baseline_data_importer
from src.steps.importer import new_data_importer
from src.steps.trainer import trainer
from src.steps.transformer import baseline_and_new_data_combiner
from src.steps.transformer import transformer
from src.steps.validation import drift_detector

logger = get_logger(__name__)


@click.command()
@click.option(
    "--min-f1",
    default=0.8,
    help="Minimum F1 Score required to deploy the model (default: 0.8)",
)
@click.option(
    "--min-roc-auc",
    default=0.8,
    help="Minimum ROC AUC required to deploy the model (default: 0.8)",
)
@click.option(
    "--min-pr-auc",
    default=0.8,
    help="Minimum F1 Score required to deploy the model (default: 0.8)",
)
@click.option(
    "--max-brier",
    default=0.05,
    help="Maximum Brier Score required to deploy the model (default: 0.8)",
)
def main(
    min_f1: float,
    max_brier: float,
    min_roc_auc: float,
    min_pr_auc: float,
):
    """Entrypoint to Continuous Deployment (CD) Pipeline execution.

    Retrieves the active model deployer and instantiates the configuration
    required to run the CD pipeline

    Example usage:
        python src/run_deployment_pipeline.py --deploy --predict --min-f1 0.8

    Args:
        min_f1 (float): Minimum F1 Score Allowed for Model Deployment
        max_brier (float): Maximum Brier Score Allowed for Model Deployment
        min_roc_auc (float): Minimum ROC AUC Score Allowed for Model Deployment
        min_pr_auc (float): Minimum PR AUC Score Allowed for Model Deployment
    """
    model_name = "model"
    deployment_pipeline_name = "continuous_deployment_pipeline_4"
    deployer_step_name = "seldon_model_deployer_step"

    seldon_implementation = "SKLEARN_SERVER"

    model_deployer = SeldonModelDeployer.get_active_model_deployer()
    logger.info(f"Active Model Deployer is: {model_deployer}")

    deployment_trigger_ = deployment_trigger(
        config=DeploymentTriggerConfig(
            min_f1=min_f1,
            max_brier=max_brier,
            min_roc_auc=min_roc_auc,
            min_pr_auc=min_pr_auc,
        )
    )
    model_deployer_step = seldon_model_deployer_step(
        params=SeldonDeployerStepParameters(
            service_config=SeldonDeploymentConfig(
                model_name=model_name,
                replicas=1,
                implementation=seldon_implementation,
            ),
            timeout=120,
        )
    )
    run_deployment_pipeline(deployment_trigger_, model_deployer_step)

    services = model_deployer.find_model_server(
        pipeline_name=deployment_pipeline_name,
        pipeline_step_name=deployer_step_name,
        model_name=model_name,
    )
    if not services:
        print(
            "No Seldon prediction server is currently running."
            "This implies there was an issue with deploying your"
            "model to Seldon."
        )
        return

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


def run_deployment_pipeline(
    deployment_trigger: BaseStep, model_deployer: BaseStep
) -> None:
    """Executes a Continuous Deployment (CD) Pipeline

    In addition to training a model as in the training pipeline,
    the CD pipeline also:
    - Compares baseline and new data, checking for data and target drift
    - Evaluates the performance of the trained model on the holdout set
    - Deploys the model to the model deployer if Deployment Trigger succeeds

    Args:
        deployment_trigger (BaseStep): Step deciding whether to deploy the model
        model_deployer (BaseStep): Step responsible for deploying the model
    """
    deployment = continuous_deployment_pipeline(
        baseline_data_importer=baseline_data_importer(),
        new_data_importer=new_data_importer(),
        data_combiner=baseline_and_new_data_combiner(),
        transformer=transformer(),
        trainer=trainer(),
        evaluator=evaluator(),
        drift_detector=drift_detector,
        deployment_trigger=deployment_trigger,
        model_deployer=model_deployer,
    )

    deployment.run()


if __name__ == "__main__":
    main()
