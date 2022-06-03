

import sys
sys.path.append(r'net2brain')

from helper.helper import get_paths



def test_check_loadup():
    import os
    import os.path as op
    import json
    from datetime import date
    from datetime import datetime
    from rdm_generation import RDM
    from evaluation import Evaluation
    from feature_extraction import FeatureExtraction
    
    """Write down all relevant paths"""
    PATH_COLLECTION = get_paths()
    CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]
    BASE_DIR = PATH_COLLECTION["BASE_DIR"]
    GUI_DIR = PATH_COLLECTION["GUI_DIR"]
    PARENT_DIR = PATH_COLLECTION["PARENT_DIR"]
    INPUTS_DIR = PATH_COLLECTION["INPUTS_DIR"]
    FEATS_DIR = PATH_COLLECTION["FEATS_DIR"]
    RDMS_DIR = PATH_COLLECTION["RDMS_DIR"]
    STIMULI_DIR = PATH_COLLECTION["STIMULI_DIR"]
    BRAIN_DIR = PATH_COLLECTION["BRAIN_DIR"]
        

