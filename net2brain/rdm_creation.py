import json
import glob
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler as SS
from datetime import datetime


def ensure_directory(path):
    """Method to ensure directory exists

    Args:
        path (str): path to folder to create
    """
    if not os.path.exists(path):
        os.mkdir(path)


def create_save_folder():
    """Creates folder to save the features in. They are structured after daytime

    Returns:
        save_path(str): Path to save folder
    """
    # Get current time
    now = datetime.now()
    now_formatted = now.strftime("%d.%m.%y %H:%M:%S")

    # Replace : through -
    log_time = now_formatted.replace(":", "-")

    # Combine to path
    save_path = f"rdms/{log_time}"

    # Create directory
    ensure_directory("rdms")
    ensure_directory(f"rdms/{log_time}")

    return save_path


class RDMCreator:
    """This class creates RDMs from the features that have been extracted witht the feature extraction
    module
    """

    def __init__(self, feat_path, save_path=None, distance="pearson"):
        """Initiation for RDM Creation

        Args:
            feat_path (str): path where to find earlier generated features
            save_path (str, optional): Path where to save RDMs Defaults to None.
            distance (str, optional): Distance metric for RDM creation. Defaults to "pearson".
        """

        self.feat_path = feat_path

        # Create save_path
        if save_path is None:
            self.save_path = create_save_folder()
        else:
            ensure_directory(save_path)
            self.save_path = save_path

        self.distance_name = distance
        # Create distance metric
        if distance == "pearson":
            self.distance = self.pearson_dist

    def create_json(self):
        """Saves arguments in json used for creating RDMs
        """

        args_file = os.path.join(self.save_path, 'args.json')
        args = {
            "distance": self.distance_name,
            "feat_dir": self.feat_path,
            "save_dir": self.save_path}

        with open(args_file, 'w') as fp:
            json.dump(args, fp, sort_keys=True, indent=4)

    def get_features(self, layer_id, i):
        """Get avtivations of a certain layer for image i.
        Returns flattened activations

        Args:
            layer_id (str): name of layer
            i (int): from which image the layer is extracted

        Returns:
            numpy array: activations from layer
        """

        activations = glob.glob(self.feat_path + "/*" + ".npz")
        activations.sort()
        feat = np.load(activations[i], allow_pickle=True)[layer_id]

        return feat.ravel()

    def get_layers_ncondns(self):
        """Function to return facts about the npz-file

        Returns:
            num_layers (int): Amount of layers
            layer_list (list): List of layers
            num_conds (int): Amount of images

        """

        activations = glob.glob(self.feat_path + "/*.npz")
        num_condns = len(activations)
        feat = np.load(activations[0], allow_pickle=True)

        num_layers = 0
        layer_list = []

        for key in feat:
            if "__" in key:  # key: __header__, __version__, __globals__
                continue
            else:
                num_layers += 1
                layer_list.append(key)  # collect all layer names

        # Liste: ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

        return num_layers, layer_list, num_condns

    def pearson_dist(self, activations):
        """This function calculates the pearson distance between the activations

        Args:
            activations (array):flattened activations for each image (78, 193600)

        Returns:
           array: image x image array
        """
        r_scaled = SS().fit_transform(np.array(activations))  # list to npy array and normalize the values
        rdm = 1 - np.corrcoef(r_scaled)  # Perform pearson correlation coefficient
        rdm = np.array(rdm)
        return rdm

    def create_rdms(self):
        """
        Main function to create RDMs from before created features

        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        save_dir : directory to save the computed RDM
        dist : dist used for computing RDM (e.g. 1-Pearson's R)

        Output:
        One RDM per Network Layer in npy-format
        """

        self.create_json()  # create json
        num_layers, layer_list, num_condns = self.get_layers_ncondns()  # get number of layers and number of conditions(images) for RDM

        # num_layers = 8
        # layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        # num_conds = 78 (amount of images)

        cwd = os.getcwd()  # current working dir

        for layer in range(num_layers):  # for each layer

            os.chdir(cwd)  # go to current dir, otherwise we will have path errors
            layer_id = layer_list[layer]  # current layer name
            RDM_filename_fmri = os.path.join(self.save_path, layer_id + ".npz")  # the savepaths
            activations = []

            for i in tqdm(range(num_condns)):  # for each datafile for the current layer
                feature_i = self.get_features(layer_id, i)  # get activations of the current layer
                activations.append(feature_i)  # collect in a list

            # Calculate distance of RDMs
            rdm = self.distance(activations)

            # saving RDM
            np.savez(RDM_filename_fmri, rdm)
            del rdm
