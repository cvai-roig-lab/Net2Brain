import numpy as np

from net2brain.rdm_creation import RDMCreator


def test_rdm_creator(root_path, tmp_path):
    data_path = root_path / "compare_files"
    rdm = RDMCreator(
        feat_path=str(data_path / "correct_data_feats"),
        save_path=str(tmp_path)
    )
    rdm.create_rdms()

    # Compare extractions with ground truth
    gt_path = data_path / "correct_data_rdm"

    for gt_file in gt_path.glob("*.npz"):
        test_file = tmp_path / gt_file.name
        gt = np.load(gt_file)["arr_0"]
        test = np.load(test_file)["arr_0"]
        assert np.allclose(gt, test)
