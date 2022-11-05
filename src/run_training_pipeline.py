"""Run training pipeline"""
from src.pipelines.train_pipeline import train_pipeline
from src.steps.evaluator import evaluator
from src.steps.importer import baseline_data_importer
from src.steps.trainer import trainer
from src.steps.transformer import transformer


def run_training_pipeline() -> None:
    """Executes the Training Pipeline"""
    pipeline = train_pipeline(
        baseline_data_importer(),
        transformer(),
        trainer(),
        evaluator(),
    )
    pipeline.run()


if __name__ == "__main__":
    run_training_pipeline()
