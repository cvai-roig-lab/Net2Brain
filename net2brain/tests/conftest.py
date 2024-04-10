from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def root_path() -> Path:
    return Path(__file__).parent


def available_devices() -> list:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    # if hasattr(torch.backends, "mps"):
    #     if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #         devices.append("mps")
    return devices


@pytest.fixture(params=available_devices())
def device(request) -> str:
    return request.param
