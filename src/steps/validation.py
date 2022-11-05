from evidently.model_profile import Profile
from zenml.integrations.evidently.steps import evidently_profile_step
from zenml.integrations.evidently.steps import EvidentlyColumnMapping
from zenml.integrations.evidently.steps import EvidentlyProfileParameters
from zenml.steps import step

from src.util import columns


drift_detector = evidently_profile_step(
    step_name="drift_detector",
    params=EvidentlyProfileParameters(
        column_mapping=EvidentlyColumnMapping(target=columns.TARGET),
        profile_sections=["datadrift", "categoricaltargetdrift"],
        verbose_level=1,
    ),
)
