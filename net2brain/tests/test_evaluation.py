from pathlib import Path

import pandas as pd
import pytest
import numpy as np

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
    df["ROI"] = df["ROI"].apply(lambda x: x.split(" ")[1])
    df["Layer"] = df["Layer"].apply(lambda x: x.split(" ")[1])
    df = df.sort_values(by=["ROI", "Layer"]).reset_index(drop=True)

    gt = pd.read_csv(data_path / case / "rsa" / "results.csv")
    gt["ROI"] = gt["ROI"].apply(lambda x: x.split(" ")[1])
    gt["Layer"] = gt["Layer"].apply(lambda x: x.split(" ")[1])
    gt = gt.sort_values(by=["ROI", "Layer"]).reset_index(drop=True)

    # Parse R2_array from string if needed
    gt["R2_array"] = gt["R2_array"].apply(
        lambda x: np.fromstring(x.strip("[]").replace('\n', ' '), sep=' ') if isinstance(x, str) else x
    )

    # Compare float columns with tolerance
    float_cols = ['R2', '%R2', 'Significance', 'SEM', 'LNC', 'UNC']
    for col in float_cols:
        assert np.allclose(df[col].values, gt[col].values, rtol=1e-6, atol=1e-10), f"Column {col} differs"

    # Compare string columns exactly
    string_cols = ['ROI', 'Layer', 'Model']
    for col in string_cols:
        assert df[col].equals(gt[col]), f"Column {col} differs"

    # Compare R2_array with tolerance
    for i in range(len(df)):
        assert np.allclose(df.iloc[i]['R2_array'], gt.iloc[i]['R2_array'], rtol=1e-6, atol=1e-10), \
            f"R2_array differs at row {i}"