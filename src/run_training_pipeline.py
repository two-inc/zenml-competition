from src.pipelines.train_pipeline import train_pipeline
from src.steps.evaluator import evaluator
from src.steps.importer import importer
from src.steps.trainer import trainer
from src.steps.transformer import transformer


def run_training_pipeline() -> None:
    """Executes the ZenML train_pipeline"""
    pipeline = train_pipeline(
        importer(),
        transformer(),
        trainer(),
        evaluator(),
    )
    pipeline.run()


if __name__ == "__main__":
    run_training_pipeline()
