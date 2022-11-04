from zenml.integrations.evidently.steps import (
    EvidentlyProfileParameters,
    evidently_profile_step,
)
from evidently.model_profile import Profile


drift_detector = evidently_profile_step(
    step_name="drift_detector",
    params=EvidentlyProfileParameters(
        profile_sections=[
            "datadrift",
        ],
        verbose_level=1,
    ),
)

