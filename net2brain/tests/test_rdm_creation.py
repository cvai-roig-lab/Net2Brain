import numpy as np
import pytest
from net2brain.rdm_creation import RDMCreator


@pytest.fixture(params=["case1", "case2"])
def case(request):
    return request.param


def test_rdm_creator(root_path, tmp_path, case):
    data_path = root_path / "test_cases" / case

    rdm = RDMCreator(feat_path=str(data_path / "features"), save_path=str(tmp_path))
    rdm.create_rdms()

    # Compare extractions with ground truth
    gt_path = data_path / "rdm"

    for gt_file in gt_path.glob("*.npz"):
        test_file = tmp_path / gt_file.name
        gt = np.load(gt_file)["arr_0"]
        test = np.load(test_file)["arr_0"]
        assert np.allclose(gt, test)
