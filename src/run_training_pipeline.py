from src.pipelines.train_pipeline   import train_pipeline
from src.steps   import importer
from src.steps   import transformer
from src.steps   import trainer
from src.steps   import evaluator


def run_training_pipeline():
    pipeline = train_pipeline(
        importer.importer(),
        transformer.transformer(),
        trainer.trainer(),
        evaluator.evaluator(),
    )
    pipeline.run()

if __name__ == "__main__":
    run_training_pipeline()
