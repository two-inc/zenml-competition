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
from src.pipelines.deployment_pipeline import dynamic_importer
from src.pipelines.deployment_pipeline import SeldonDeploymentLoaderStepConfig
from src.pipelines.inference_pipeline import inference_pipeline
from src.run_training_pipeline import ge_validator_step
from src.steps.evaluator import evaluator
from src.steps.importer import importer
from src.steps.prediction_service_loader import prediction_service_loader
from src.steps.predictor import predictor
from src.steps.trainer import trainer
from src.steps.transformer import transformer

logger = get_logger(__name__)


@click.command()
@click.option(
    "--deploy",
    "-d",
    is_flag=True,
    default=False,
    help="Run the deployment pipeline to train and deploy a model",
)
@click.option(
    "--predict",
    "-p",
    is_flag=True,
    default=False,
    help="Run the inference pipeline to send a prediction request "
    "to the deployed model",
)
@click.option(
    "--min-f1",
    default=0.8,
    help="Minimum F1 Score required to deploy the model (default: 0.8)",
)
def main(
    deploy: bool,
    predict: bool,
    min_f1: float,
):
    """Run the Seldon example continuous deployment or inference pipeline
    Example usage:
        python src/run_deployment_pipeline.py --deploy --predict --min-f1 0.8
    """
    model_name = "model"
    deployment_pipeline_name = "continuous_deployment_pipeline"
    deployer_step_name = "seldon_model_deployer_step"

    seldon_implementation = "SKLEARN_SERVER"

    model_deployer = SeldonModelDeployer.get_active_model_deployer()
    logger.info(f"Active Model Deployer is: {model_deployer}")

    if deploy:
        deployment_trigger_ = deployment_trigger(
            config=DeploymentTriggerConfig(
                min_f1_score=min_f1,
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

    if predict:
        prediction_service_loader_ = prediction_service_loader(
            SeldonDeploymentLoaderStepConfig(
                pipeline_name=deployment_pipeline_name,
                step_name=deployer_step_name,
                model_name=model_name,
            )
        )
        predict(prediction_service_loader_)

    services = model_deployer.find_model_server(
        pipeline_name=deployment_pipeline_name,
        pipeline_step_name=deployer_step_name,
        model_name=model_name,
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

    else:
        print(
            "No Seldon prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


def run_deployment_pipeline(
    deployment_trigger: BaseStep, model_deployer: BaseStep
) -> None:
    """Initializes a continuous deployment pipeline run"""
    deployment = continuous_deployment_pipeline(
        importer=importer(),
        validator=ge_validator_step,
        transformer=transformer(),
        trainer=trainer(),
        evaluator=evaluator(),
        deployment_trigger=deployment_trigger,
        model_deployer=model_deployer,
    )

    deployment.run()


def predict(prediction_service_loader: BaseStep) -> None:
    """Initialize an inference pipeline run"""
    inference = inference_pipeline(
        dynamic_importer=dynamic_importer(),
        prediction_service_loader=prediction_service_loader,
        predictor=predictor(),
    )
    inference.run()


if __name__ == "__main__":
    main()
