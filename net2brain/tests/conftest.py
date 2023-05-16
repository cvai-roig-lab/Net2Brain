import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def root_path() -> Path:
    return Path(__file__).parent
