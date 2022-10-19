from zenml.config import DockerSettings
from zenml.integrations.constants import SELDON
from zenml.integrations.constants import SKLEARN

docker_settings = DockerSettings(
    requirements="requirements.txt", required_integrations=[SELDON, SKLEARN]
)
