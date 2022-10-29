from zenml.integrations.great_expectations.visualizers.ge_visualizer import (
    GreatExpectationsVisualizer,
)
from zenml.post_execution import get_pipeline


def visualize_results(pipeline_name: str, step_name: str) -> None:
    pipeline = get_pipeline(pipeline_name)
    last_run = pipeline.runs[-1]
    validation_step = last_run.get_step(step=step_name)
    GreatExpectationsVisualizer().visualize(validation_step)


if __name__ == "__main__":
    visualize_results("validation_pipeline", "profiler")
    visualize_results("validation_pipeline", "train_validator")
    visualize_results("validation_pipeline", "test_validator")
