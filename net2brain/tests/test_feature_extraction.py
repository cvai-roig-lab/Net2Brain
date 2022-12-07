import pytest

from net2brain.feature_extraction import FeatureExtractor


@pytest.mark.parametrize(
    "netset,model", [
        ("standard","AlexNet"), ("timm", "vit_base_patch32_224_in21k"), 
        ("pytorch", "deeplabv3_resnet101"), ("unet", "unet"),
        ("taskonomy", "autoencoding"), ("pyvideo", "slowfast_r50"),
        ('clip', 'RN50'), ("cornet", "cornet_z")

    ]
)
def test_load_netset_model(netset, model):
    fe = FeatureExtractor(model, netset)
    assert fe.model == model
    pass


def test_extractor():
    pass


# def test_alexnet():
    
#     """Write down all relevant paths"""
#     PATH_COLLECTION = get_paths()
#     CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]

#     # Set path for saving activations    
#     path = op.join(CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_feats")
#     path_truth = op.join(CURRENT_DIR, "net2brain/tests/compare_files/correct_data_feats")
    
#     # Create folder if it does not exists
#     if not os.path.exists(path):
#         os.makedirs(path)

#     # Start extraction
#     extract = FeatureExtraction("AlexNet", "78images", "standard")
#     extract.feats_path = path
#     extract.start_extraction()
    
#     # Compare extractions with ground truth
#     to_test = glob.glob(op.join(path, "*"))
#     to_compare = glob.glob(op.join(path_truth, "*"))
    
#     for i, truth in enumerate(to_compare):
#         status = filecmp.cmp(truth, to_test[i])
#         print(truth.split(os.sep)[-1], to_test[i].split(os.sep)[-1], status)
#         if status == False:
#             return status
    
#     return status



# def test_only_cornet_generation():
#     """Write down all relevant paths"""
#     PATH_COLLECTION = get_paths()
#     CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]

#     # Set path for saving activations    
#     path = op.join(CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_feats")
    
#     # Create folder if it does not exists
#     if not os.path.exists(path):
#         os.makedirs(path)

#     # Start extraction
#     extract = FeatureExtraction("cornet_z", "78images", "cornet")
#     extract.feats_path = path

#     try:
#         extract.start_extraction()
#         return True
#     except:
#         return False
    

# def test_only_taskonomy_generation():
#     """Write down all relevant paths"""
#     PATH_COLLECTION = get_paths()
#     CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]

#     # Set path for saving activations
#     path = op.join(
#         CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_feats")

#     # Create folder if it does not exists
#     if not os.path.exists(path):
#         os.makedirs(path)

#     # Start extraction
#     extract = FeatureExtraction("autoencoding", "78images", "taskonomy")
#     extract.feats_path = path
    
#     try:
#         extract.start_extraction()
#         return True
#     except:
#         return False
    
    
# def test_only_timm_generation():
#     """Write down all relevant paths"""
#     PATH_COLLECTION = get_paths()
#     CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]

#     # Set path for saving activations
#     path = op.join(
#         CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_feats")

#     # Create folder if it does not exists
#     if not os.path.exists(path):
#         os.makedirs(path)

#     # Start extraction
#     extract = FeatureExtraction("adv_inception_v3", "78images", "timm")
#     extract.feats_path = path
    
#     try:
#         extract.start_extraction()
#         return True
#     except:
#         return False
    
    
