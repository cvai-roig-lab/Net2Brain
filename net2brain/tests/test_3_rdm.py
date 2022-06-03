
import glob
import filecmp
import os
import sys
sys.path.append(r'net2brain')
from rdm_generation import RDM
from helper.helper import get_paths
from os import path as op

def test_rdm():
    """Write down all relevant paths"""
    PATH_COLLECTION = get_paths()
    CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]

    # Set path for saving activations
    path = op.join(CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_rdm")
    path_feats = op.join(CURRENT_DIR, "net2brain/tests/compare_files/correct_data_feats")
    path_truth = op.join(CURRENT_DIR, "net2brain/tests/compare_files/correct_data_rdm")

    # Create folder if it does not exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Start RDM Generation
    save_path = path
    feats_data_path = path_feats

    rdm = RDM(save_path, feats_data_path)
    rdm.create_rdms()
    
    # Compare extractions with ground truth
    to_test = glob.glob(op.join(path, "*.npz"))
    to_compare = glob.glob(op.join(path_truth, "*.npz"))
    
    for i, truth in enumerate(to_compare):
        status = filecmp.cmp(truth, to_test[i])
        print(truth.split(os.sep)[-1], to_test[i].split(os.sep)[-1], status)
        if status == False:
            return status

    return status

