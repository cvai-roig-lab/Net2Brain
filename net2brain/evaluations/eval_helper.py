import numpy as np
import h5py
import os
from scipy import io
from scipy.spatial.distance import squareform
import re
from helper.helper import get_paths

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

"""Human sorting:
https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside"""
def atoi(text):
    """[summary]

    Args:
        text (str): splitted layernme (feautures, 10 , npz)

    Returns:
        int/str: depeneding on what it is
    """

    return int(text) if text.isdigit() else text

def natural_keys(text):
    """alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    Args:
        text (str): layer file name (features.10.npz)

    Returns:
        list: keys for sorting
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def loadmat(matfile):
    """loads the input files if in .mat format

    Args:
        matfile (str): path to file

    Returns:
        numpy array: loaded file
    """
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def loadnpy(npyfile):
    """load in npy format

    Args:
        npyfile (str): path to file

    Returns:
         numpy array: loaded file
    """
    return np.load(npyfile)


def loadnpz(npzfile):
    """load in npz format

    Args:
        npzfile (str): path to file

    Returns:
        numpy array: loaded file
    """
    return np.load(npzfile, allow_pickle=True)


def load(data_file):
    """organizing loading functions

    Args:
        data_file (str/path): path to roi file

    Returns:
        numpy array: loaded file
    """
    root, ext = os.path.splitext(data_file)
    return {'.npy': loadnpy,
            '.mat': loadmat,
            '.npz': loadnpz
            }.get(ext, loadnpy)(data_file)


def sq(x):
    """Converts a square-form distance matrix from a vector-form distance vector

    Args:
        x (numpy array): numpy array that should be vector

    Returns:
        numpy array: numpy array as vector
    """
    return squareform(x, force='tovector', checks=False)

def error_message(message):
    """Helping function to print an error message

    Args:
        message (str): What is the error?
    """
    print("")
    print("#############")
    print("#############")
    print("#############")
    print(message)
    print("#############")
    print("#############")
    print("#############")
    print("")
