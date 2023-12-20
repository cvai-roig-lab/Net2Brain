import numpy as np
import pytest
from net2brain.rdm_creation import RDMCreator
from net2brain.rdm import LayerRDM


@pytest.fixture(params=["case1"])
def case(request):
    return request.param


@pytest.mark.parametrize('distance', ('euclidean', 'pearson', 'cosine'))
def test_rdm_creator(root_path, tmp_path, case, distance):
    data_path = root_path / "test_cases" / case

    rdm = RDMCreator(verbose=False, device="cpu")
    rdm.create_rdms(feature_path=data_path / "features", save_path=tmp_path / distance,
                    distance=distance, chunk_size=None, save_format="npz")

    # Compare extractions with ground truth
    gt_path = data_path / "rdm" / distance

    for gt_file in gt_path.iterdir():
        test_file = tmp_path / distance / gt_file.name
        gt = LayerRDM.from_file(gt_file)
        test = LayerRDM.from_file(test_file)
        assert np.allclose(gt.rdm, test.rdm)
