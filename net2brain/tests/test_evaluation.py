from pathlib import Path

import pandas as pd
import pytest

from net2brain.evaluations.rsa import RSA


@pytest.mark.parametrize(
    "case,model_name",
    [
        ("case1", "ResNet18"),
        ("case3", "RN50"),
    ],
)
def test_rsa(root_path, case, model_name):
    data_path = root_path / "test_cases"

    brain_path = root_path.parent.parent / Path(
        "notebooks", "input_data", "brain_data", "78images"
    )
    rsa = RSA(
        brain_rdms_path=brain_path,
        model_rdms_path=str(data_path / case / "rdm"),
        model_name=model_name,
    )
    df = rsa.evaluate()

    gt = pd.read_csv(data_path / case / "rsa" / "results.csv")
    pd.testing.assert_frame_equal(df, gt)
