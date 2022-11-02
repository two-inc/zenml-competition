from zenml.config import DockerSettings
from zenml.integrations.constants import GCP
from zenml.integrations.constants import KUBEFLOW
from zenml.integrations.constants import KUBERNETES
from zenml.integrations.constants import MLFLOW
from zenml.integrations.constants import SCIPY
from zenml.integrations.constants import SELDON
from zenml.integrations.constants import SKLEARN
from zenml.integrations.constants import TENSORFLOW

REQUIRED_INTEGRATIONS = [
    SELDON,
    SKLEARN,
    MLFLOW,
    KUBEFLOW,
    KUBERNETES,
    SCIPY,
    GCP,
    TENSORFLOW,
]


docker_settings = DockerSettings(
    requirements="docker-requirements.txt",
    required_integrations=REQUIRED_INTEGRATIONS,
    dockerignore=".dockerignore",
    apt_packages=["cmake", "libgomp1", "gcc", "git"],
)
