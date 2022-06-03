import os
import os.path as op
import glob
import scipy.io as sio
import numpy as np
from architectures.pytorch_models import MODELS as pymodels
from architectures.timm_models import MODELS as timmmodels
from architectures.torchhub_models import MODELS as torchmodels
from architectures.unet_models import MODELS as unetmodels
from architectures.yolo_models import MODELS as yolomodels
from architectures.taskonomy_models import MODELS as taskonomymodels
from architectures.slowfast_models import MODELS as pyvideomodels


"""Write down all relevant paths"""
CURRENT_DIR = op.abspath(os.curdir)
BASE_DIR = op.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = op.dirname(BASE_DIR)  # path to parent folder
FEATS_DIR = op.join(PARENT_DIR, 'feats')
GUI_DIR = op.join(BASE_DIR, 'helper', 'gui')
INPUTS_DIR = op.join(PARENT_DIR, 'input_data')
STIMULI_DIR = op.join(INPUTS_DIR, 'stimuli_data')
RDMS_DIR = op.join(PARENT_DIR, 'rdms')
BRAIN_DIR = op.join(INPUTS_DIR, 'brain_data')

PATH_COLLECTION = {"CURRENT_DIR": CURRENT_DIR,
                   "BASE_DIR": BASE_DIR,
                   "PARENT_DIR": PARENT_DIR,
                   "FEATS_DIR": FEATS_DIR,
                   "GUI_DIR": GUI_DIR,
                   "INPUTS_DIR": INPUTS_DIR,
                   "STIMULI_DIR": STIMULI_DIR,
                   "RDMS_DIR": RDMS_DIR,
                   "BRAIN_DIR": BRAIN_DIR}

# try importing Linux modules
global vissl_exist
global clip_exist
global detectron_exist


try:
    import clip
    clip_exist = True
except:
    print("Clip models are not installed.")
    clip_exist = False
    
try:
    import cornet
    cornet_exist = True
except:
    print("CORnet models are not installed.")
    cornet_exist = False

try:
    from vissl.utils.hydra_config import AttrDict
    vissl_exist = True
except:
    print("Vissl models are not installed")
    vissl_exist = False

try:
    import detectron2
    detectron_exist = True
except:
    print("Detectron2 is not installed.")
    detectron_exist = False


AVAILABLE_NETWORKS = {'standard (' + str(len(pymodels)) + ')' : list(pymodels.keys()),
                      'timm (' + str(len(timmmodels)) + ')': list(timmmodels.keys()),
                      'pytorch (' + str(len(torchmodels)) + ')': list(torchmodels.keys()),
                      'unet (' + str(len(unetmodels)) + ')': list(unetmodels.keys()),
                      'taskonomy (' + str(len(taskonomymodels)) + ')': list(taskonomymodels.keys()),
                      'pyvideo (' + str(len(pyvideomodels)) + ')': list(pyvideomodels.keys())}
                     # 'yolo (' + str(len(yolomodels)) + ')': list(yolomodels.keys())}
                      
if detectron_exist:
    from architectures.detectron2_models import MODELS as detectronmodels
    AVAILABLE_NETWORKS.update({'detectron2 (' + str(len(detectronmodels)) + ')': list(detectronmodels.keys())})
    
if vissl_exist:
    from architectures.vissl_models import MODELS as visslmodels
    AVAILABLE_NETWORKS.update({'vissl (' + str(len(visslmodels)) + ')': list(visslmodels.keys())})

if clip_exist:
    from architectures.clip_models import MODELS as clipmodels
    AVAILABLE_NETWORKS.update({'clip (' + str(len(clipmodels)) + ')': list(clipmodels.keys())})
    
if cornet_exist:
    from architectures.cornet_models import MODELS as cornetmodels
    AVAILABLE_NETWORKS.update({'cornet (' + str(len(cornetmodels)) + ')': list(cornetmodels.keys())})



# SlowFast Models
# slowfast_list = ["Kinetics/c2/I3D_8x8_R50",
#                  "Kinetics/c2/I3D_NLN_8x8_R50",
#                  "Kinetics/c2/SLOW_4x16_R50",
#                  "Kinetics/c2/SLOW_8x8_R50",
#                  "Kinetics/c2/SLOWFAST_4x16_R50",
#                  "Kinetics/c2/SLOWFAST_8x8_R50",
#                  "Kinetics/SLOWFAST_8x8_R50_stepwise",
#                  "Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid",
#                  "Kinetics/C2D_8x8_R50",
#                  "AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1",
#                  "AVA/c2/SLOWFAST_32x2_R101_50_50"]



def get_available_nets():
    """Returns dictionary of all available nets

    Returns:
        dict: netsetname: all networks in netset
    """
    return AVAILABLE_NETWORKS


def get_paths():
    """Returns dictionary of all paths

    Returns:
        dict: pathname: path
    """
    return PATH_COLLECTION


def get_available_metrics():
    """ Returns all available evaluation metrics
    
    Returns:
        list: all metrics
    """
    return ["RSA", "Weighted RSA", "Searchlight"]


"""Below you can find functions that might help"""



def available_nets_cli():

    # Check for DNNs
    available_nets_dict = get_available_nets()
    available_netsets = list(available_nets_dict.keys())


    for set in available_netsets:
        all_nets = available_nets_dict[set]
        for net in all_nets:
            print(" - " + set.split(" ")[0] + "_" + net)


def MATtoNPZ(path_to_folder):
    """Automate transformation from MAT zu NPZ

    Args:
        path_to_folder (str/Path): path to the folder containing .mat files
    """

    path = path_to_folder
    activations = glob.glob(path + "/*" + ".mat")  # list all .mat files in folder
    activations.sort()

    for mat in activations:
        data = sio.loadmat(mat)
        our_rdm = data["subject_rdm"]
        this_name = mat.split("\\")[-1].split(".")[0]
        save_path = os.path.join(path, "fmri_" + this_name + ".npz")
        np.savez(save_path, our_rdm)


