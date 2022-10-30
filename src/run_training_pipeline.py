from google.cloud import storage
from zenml.integrations.great_expectations.steps import (
    great_expectations_profiler_step,
)
from zenml.integrations.great_expectations.steps import (
    great_expectations_validator_step,
)
from zenml.integrations.great_expectations.steps import (
    GreatExpectationsProfilerParameters,
)
from zenml.integrations.great_expectations.steps import (
    GreatExpectationsValidatorParameters,
)
from zenml.logger import get_logger

from src.materializer.materializer import CompetitionMaterializer
from src.pipelines.train_pipeline import train_pipeline
from src.steps.evaluator import evaluator
from src.steps.importer import importer
from src.steps.trainer import trainer
from src.steps.transformer import transformer

logger = get_logger(__name__)

# instantiate a builtin Great Expectations data profiling step
ge_profiler_params = GreatExpectationsProfilerParameters(
    expectation_suite_name="synthetic-financial-payment-data",
    data_asset_name="synthetic-financial-payment-ref-df",
)
ge_profiler_step = great_expectations_profiler_step(
    step_name="ge_profiler_step",
    params=ge_profiler_params,
)

# instantiate a builtin Great Expectations data validation step
ge_validator_params = GreatExpectationsValidatorParameters(
    expectation_suite_name="synthetic-financial-payment-data",
    data_asset_name="synthetic-financial-payment-test-df",
)

ge_validator_step = great_expectations_validator_step(
    step_name="ge_validator_step",
    params=ge_validator_params,
)


def run_training_pipeline() -> None:
    """Executes the ZenML train_pipeline"""
    pipeline = train_pipeline(
        importer().configure(output_materializers=CompetitionMaterializer),
        transformer(),
        ge_profiler_step,
        ge_validator_step,
        trainer(),
        evaluator(),
    )
    pipeline.run()


if __name__ == "__main__":
    run_training_pipeline()
