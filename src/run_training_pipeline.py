"""Run training pipeline"""
from google.cloud import storage
from zenml.logger import get_logger

from src.materializer.materializer import CompetitionMaterializer
from src.pipelines.train_pipeline import train_pipeline
from src.steps.evaluator import evaluator
from src.steps.importer import importer
from src.steps.trainer import trainer
from src.steps.transformer import transformer
from src.steps.drift_detector import drift_detector

logger = get_logger(__name__)


def run_training_pipeline() -> None:
    """Executes the ZenML train_pipeline"""
    pipeline = train_pipeline(
        importer().configure(output_materializers=CompetitionMaterializer),
        transformer(),
        drift_detector,
        trainer(),
        evaluator(),
    )
    pipeline.run()


if __name__ == "__main__":
    run_training_pipeline()
