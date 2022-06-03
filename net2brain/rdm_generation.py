import json
import glob
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler as SS


class RDM():
    def __init__(self, save_dir, feat_dir, dist="pearson"):
        """initialize RDM creation

        Args:
            save_dir (str/path): Path to where we want to save the RDMs
            feat_dir (str/path): Path to where our features are saved
            dist (str, optional): Distance metric. Defaults to "pearson".
        """
        
        self.save_dir = save_dir
        self.feat_dir = feat_dir
        self.dist = dist
        
    def create_dir(self):
        """Creates directory if it does not exist
        """
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def create_json(self):
        """Saves arguments in json used for creating RDMs
        """
        
        args_file = os.path.join(self.save_dir, 'args.json')
        args = {
            "distance": self.dist,
            "feat_dir": self.feat_dir,
            "save_dir": self.save_dir}

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

        activations = glob.glob(self.feat_dir + "/*" + ".npz")
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

        activations = glob.glob(self.feat_dir + "/*.npz")
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

        self.create_dir()  # create dir
        
        self.create_json()  # create json

        # get number of layers and number of conditions(images) for RDM
        num_layers, layer_list, num_condns = self.get_layers_ncondns()

        """
        num_layers = 8
        layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        num_conds = 78 (amount of images)
        """

        cwd = os.getcwd()  # current working dir

        for layer in range(num_layers):  # for each layer
            
            os.chdir(cwd)  # go to current dir, otherwise we will have path errors

            layer_id = layer_list[layer]  # current layer name

            RDM_filename_fmri = os.path.join(self.save_dir, layer_id + ".npz")  # the savepaths

            activations = []
            for i in tqdm(range(num_condns)):  # for each datafile for the current layer
                
                feature_i = self.get_features(layer_id, i) # get activations of the current layer
                
                activations.append(feature_i)  # collect in a list

            r_scaled = SS().fit_transform(np.array(activations))  # list to npy array and normalize the values
            
            #TODO: Do not do it with every layer at the same time - filling one by one
            #For now: RSA up to x amount - later maybe encoding models

            rdm = 1 - np.corrcoef(r_scaled)  # Perform pearson correlation coefficient
            
            del r_scaled
            

            # saving RDMs
            
            rdm_fmri = np.array(rdm)
            
            np.savez(RDM_filename_fmri, rdm_fmri)
            
            del rdm
        


