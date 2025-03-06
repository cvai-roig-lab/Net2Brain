import numpy as np
import h5py
import os
from scipy import io
from scipy.spatial.distance import squareform
import re
import warnings
import glob

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


def get_npy_files(input_path):
    """
    Returns a list of .npy files from the given input, which can be a single file, 
    a list of files, or a folder. If the input is a folder, it retrieves all .npy 
    files in that folder. If the input is a list of files or folders, it filters out 
    the folders and returns only .npy files. A warning is raised if folders are present 
    in the input list.

    Parameters:
    -----------
    input_path : str or list
        A single file path (str), a list of file/folder paths (list), or a folder path (str).

    Returns:
    --------
    list
        A list of .npy file paths.
    """
    # Convert single string input to list for consistent processing
    if isinstance(input_path, str):
        input_path = [input_path]

    # Separate files and folders
    files = [f for f in input_path if os.path.isfile(f)]
    folders = [f for f in input_path if os.path.isdir(f)]

    # Raise warning if there are folders in the input list
    if folders:
        warnings.warn("Ignoring folders in the list.")
    
    # If input contains a folder, add its .npy files to the list
    for folder in folders:
        files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')])

    # Filter only .npy files
    npy_files = [f for f in files if f.endswith('.npy')]

    if not npy_files:
        raise ValueError("No valid .npy files found.")
    
    return npy_files


def get_layers_ncondns(feat_path):
    """
    Extracts information about the number of layers, the list of layer names, and the number of conditions (images)
    from the npz files in the specified feature path.

    Parameters:
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - num_layers (int): The number of layers found in the npz files.
    - layer_list (list of str): A list containing the names of the layers.
    - num_conds (int): The number of conditions (images) based on the number of npz files in the directory.
    """
    
    # Find all npz files in the specified directory
    activations = glob.glob(feat_path + '/*.np[zy]')
    
    # Count the number of npz files as the number of conditions (images)
    num_condns = len(activations)
    
    # Load the first npz file to extract layer information
    feat = np.load(activations[0], allow_pickle=True)

    num_layers = 0
    layer_list = []

    # Iterate through the keys in the npz file, ignoring metadata keys
    for key in feat:
        if "__" in key:  # key: __header__, __version__, __globals__
            continue
        else:
            num_layers += 1
            layer_list.append(key)  # collect all layer names ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    return num_layers, layer_list, num_condns