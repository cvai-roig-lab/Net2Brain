import numpy as np
import pytest
from typing import Callable
from net2brain.rdm_creation import RDMCreator
from net2brain.rdm import LayerRDM
import torch.nn.functional as F


@pytest.fixture(params=["case3"])
def case(request):
    return request.param


def custom_cosine(x, y=None):
    x_norm = F.normalize(x, p=2, dim=1)
    return 1 - (x_norm @ x_norm.T)


custom_cosine.name = 'cosine'


@pytest.mark.parametrize('distance', ('euclidean', 'pearson', 'cosine', custom_cosine))
@pytest.mark.parametrize('save_format', ('npz', 'pt'))
@pytest.mark.parametrize('chunk_size', (None, 256,))
@pytest.mark.parametrize('features_path', ('ResNet18_Feat', 'ResNet18_Feat_consolidated'))
def test_rdm_creator(device, root_path, tmp_path, case, distance, save_format, chunk_size, features_path):
    data_path = root_path / "test_cases" / case

    distance_name = distance.name if isinstance(distance, Callable) else distance

    rdm = RDMCreator(verbose=False, device=device)
    rdm.create_rdms(feature_path=data_path / "features" / features_path, save_path=tmp_path / distance_name,
                    distance=distance, chunk_size=chunk_size, save_format=save_format)

    # Compare extractions with ground truth
    gt_path = data_path / "rdm" / distance_name / save_format

    for gt_file in gt_path.iterdir():
        test_file = tmp_path / distance_name / gt_file.name
        gt = LayerRDM.from_file(gt_file)
        test = LayerRDM.from_file(test_file)
        assert np.allclose(gt.rdm, test.rdm, rtol=1e-3, atol=1e-3)
