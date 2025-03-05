from pathlib import Path

import pandas as pd
import pytest

from net2brain.evaluations.rsa import RSA


@pytest.mark.parametrize(
    "case,model_name",
    [
        ("case1", "ResNet18"),
        ("case2", "RN50"),
    ],
)
def test_rsa(root_path, case, model_name):
    data_path = root_path / "test_cases"

    brain_path = root_path / Path("data", "brain_data")
    rsa = RSA(
        brain_rdms_path=brain_path,
        model_rdms_path=str(data_path / case / "rdm"),
        model_name=model_name,
    )

    df = rsa.evaluate()
    df["ROI"] = df["ROI"].apply(lambda x: x.split(" ")[1])  # TODO: Fix this in the code
    df["Layer"] = df["Layer"].apply(lambda x: x.split(" ")[1])
    df = df.sort_values(by=["ROI", "Layer"]).reset_index(drop=True)

    gt = pd.read_csv(data_path / case / "rsa" / "results.csv")
    gt["ROI"] = gt["ROI"].apply(lambda x: x.split(" ")[1])
    gt["Layer"] = gt["Layer"].apply(lambda x: x.split(" ")[1])
    gt = gt.sort_values(by=["ROI", "Layer"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(df.drop(columns=["R2_array"]), gt.drop(columns=["R2_array"]))
    gt["R2_array"] = gt["R2_array"].apply(lambda x: list(map(float, x.strip("[]").split())) if isinstance(x, str) else x)
    assert df["R2_array"].apply(tuple).equals(gt["R2_array"].apply(tuple))
