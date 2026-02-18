"""Infrastructure utilities for starting required services."""
import time

import docker
import requests
from docker.errors import DockerException, NotFound

from chatbot_backend.config import (
    QDRANT_CONTAINER_NAME,
    QDRANT_GRPC_PORT,
    QDRANT_HOST_PORT,
    QDRANT_IMAGE,
    QDRANT_STARTUP_TIMEOUT,
    QDRANT_STORAGE_PATH,
    QDRANT_URL,
)


class QdrantStartupError(Exception):
    """Raised when Qdrant cannot be started or reached."""

    pass


def _is_qdrant_healthy() -> bool:
    """Check if Qdrant is responding to health checks."""
    try:
        response = requests.get(f"{QDRANT_URL}/healthz", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_qdrant_container() -> None:
    """Start or create the Qdrant Docker container."""
    try:
        client = docker.from_env()
    except DockerException as e:
        raise QdrantStartupError(
            f"Cannot connect to Docker daemon. Is Docker running? Error: {e}"
        ) from e

    # Ensure storage directory exists
    QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    try:
        container = client.containers.get(QDRANT_CONTAINER_NAME)
        if container.status != "running":
            print(f"Starting existing Qdrant container '{QDRANT_CONTAINER_NAME}'...")
            container.start()
        else:
            print(f"Qdrant container '{QDRANT_CONTAINER_NAME}' is already running.")
    except NotFound:
        print(f"Creating new Qdrant container '{QDRANT_CONTAINER_NAME}'...")
        try:
            client.images.pull(QDRANT_IMAGE)
        except DockerException as e:
            raise QdrantStartupError(f"Failed to pull Qdrant image: {e}") from e

        try:
            client.containers.run(
                QDRANT_IMAGE,
                name=QDRANT_CONTAINER_NAME,
                ports={
                    "6333/tcp": QDRANT_HOST_PORT,
                    "6334/tcp": QDRANT_GRPC_PORT,
                },
                volumes={
                    str(QDRANT_STORAGE_PATH): {"bind": "/qdrant/storage", "mode": "rw"}
                },
                detach=True,
            )
        except DockerException as e:
            raise QdrantStartupError(f"Failed to create Qdrant container: {e}") from e


def _wait_for_qdrant() -> None:
    """Poll until Qdrant is healthy or timeout is reached."""
    start = time.time()
    while time.time() - start < QDRANT_STARTUP_TIMEOUT:
        if _is_qdrant_healthy():
            print("Qdrant is healthy and ready.")
            return
        time.sleep(0.5)

    raise QdrantStartupError(
        f"Qdrant did not become healthy within {QDRANT_STARTUP_TIMEOUT} seconds."
    )


def ensure_qdrant_running() -> None:
    """Ensure Qdrant is running and healthy, starting it if necessary.

    Raises:
        QdrantStartupError: If Qdrant cannot be started or does not become healthy.
    """
    if _is_qdrant_healthy():
        print("Qdrant is already running and healthy.")
        return

    print("Qdrant is not running. Attempting to start via Docker...")
    _start_qdrant_container()
    _wait_for_qdrant()
