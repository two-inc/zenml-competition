from pipelines.train_pipeline import train_pipeline
from steps.importer import importer
from steps.transformer import transformer
from steps.trainer import trainer
from steps.evaluator import evaluator


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
