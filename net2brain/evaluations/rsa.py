import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform
from .noiseceiling import NoiseCeiling
from .eval_helper import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





class RSA():
    """Evaluation with RSA
    """

    def __init__(self, model_rdms_path, brain_rdms_path, model_name, datatype="None", save_path="./", distance_metric="spearman"):
        """Initiate RSA
        Args:
            json_dir (str/path): Path to json dir
        """

        # Find all model RDMs
        self.model_rdms_path = model_rdms_path
        self.model_rdms = self.folderlookup(model_rdms_path)
        self.model_rdms.sort(key=natural_keys)

        # Find all Brain RDMs
        self.brain_rdms_path = brain_rdms_path
        self.brain_rdms = self.folderlookup(brain_rdms_path)

        # Other parameters
        self.save_path = save_path
        self.datatype = datatype
        self.model_name = model_name
        
        # For comparison
        self.other_rdms_path = None
        self.other_rdms = None

        if distance_metric.lower() == "spearman":
            self.distance = self.model_spearman

    def find_datatype(self, roi):
        """Function to find out if we should apply MEG or FMRI algorithm to the data
        Args:
            roi (str): Name of ROI
        """

        if "fmri" in roi.lower():
            self.rsa = self.rsa_fmri
        elif "meg" in roi.lower():
            self.rsa = self.rsa_meg
        else:
            # TODO: On Value Error or something
            error_message("No fmri/meg found in ROI-name. Error!")

    def model_spearman(self, model_rdm, rdms):
        """Calculate Spearman for model
        Args:
            model_rdm (numpy array): RDM of model
            rdms (list of numpy arrays): RDMs of ROI
        Returns:
            float: Spearman correlation of model and roi
        """

        model_rdm_sq = sq(model_rdm)
        return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]

    def folderlookup(self, path):
        """Looks at the available files and returns the chosen one
        Args:
            path (str/path): path to folder
        Returns:
            list: list of files in dir
        """

        files = os.listdir(path)  # Which folders do we have?
        file_sets = []

        for f in files:
            if ".json" not in f and ".DS_Store" not in f:
                if f != ".ipynb_checkpoints":
                    file_sets.append(f)

        return file_sets
    
    def check_squareform(self, rdm):
        """Ensure that the RDM is in squareform.

        Args:
            rdm (numpy array): The RDM in either squareform or vector form.

        Returns:
            numpy array: The RDM in squareform.
        """
        # Check if the RDM is in squareform. If the array is 2D and square, it's already in squareform.
        if rdm.ndim == 2 and rdm.shape[0] == rdm.shape[1]:
            return rdm
        else:
            # Convert to squareform.
            return squareform(rdm)

    def rsa_meg(self, model_rdm, brain_rdm, layername):
        """Creates the output dictionary for MEG scans. Returns {layername: R², Significance}
        Args:
            model_rdm (numpy array): DNN rdm
            brain_rdm (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]
        Returns:
            dict: {layername: [r2, significance, sem]}
        """

        key = list(model_rdm.keys())[0]  # You need to access the keys to open a npy file
        model_rdm = model_rdm[key]
        key = list(brain_rdm.keys())[0]  # You need to access the keys to open a npy file
        meg_rdm = brain_rdm[key]

        model_rdm = self.check_squareform(model_rdm) # Check if rdm is squareform #TODO Remove soon after reimplementing RSA

        # returns list of corrcoefs, depending on amount of participants in brain rdm
        corr = np.mean([self.distance(model_rdm, rdms)for rdms in meg_rdm], 1)

        # Square correlation
        corr_squared = np.square(corr)

        # Take mean to retrieve R2
        r2 = np.mean(corr_squared)

        # ttest: Ttest_1sampResult(statistic=3.921946, pvalue=0.001534)
        significance = stats.ttest_1samp(corr_squared, 0)[1]

        # standard error of mean
        sem = stats.sem(corr_squared)  # standard error of mean

        return r2, significance, sem, corr_squared

    def rsa_fmri(self, model_rdm, brain_rdm, layername):
        """Creates the output dictionary for fMRI scans. Returns {layername: R², Significance}
        Args:
            model_rdm (numpy array): DNN rdm
            brain_rdm (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]
        Returns:
            dict: {layername: [r2, significance, sem]}
        """

    
        key = list(model_rdm.keys())[0] # You need to access the keys to open a npy file
        model_rdm = model_rdm[key]
        key = list(brain_rdm.keys())[0]  # You need to access the keys to open a npy file
        fmri_rdm = brain_rdm[key]
   
        model_rdm = self.check_squareform(model_rdm) # Check if rdm is squareform #TODO Remove soon after reimplementing RSA

        # returns list of corrcoefs, depending on amount of participants in brain rdm
        corr = self.distance(model_rdm, fmri_rdm)

        # Square correlation
        corr_squared = np.square(corr)

        # Take mean to retrieve R2
        r2 = np.mean(corr_squared)

        # ttest: Ttest_1sampResult(statistic=3.921946, pvalue=0.001534)
        significance = stats.ttest_1samp(corr_squared, 0)[1]

        # standard error of mean
        sem = stats.sem(corr_squared)

        return r2, significance, sem, corr_squared

    def evaluate_roi(self, roi):
        """Functiion to evaulate the layers to the current roi , either fmri or meg
        Returns:
            dict: dictionary of all results to the current roi
        """

        all_layers_dicts = []

        # For each layer to RSA with the current ROI
        for counter, layer in enumerate(self.model_rdms):

            # Load RDMS
            roi_rdm = load(op.join(self.brain_rdms_path, roi))
            model_rdm = load(op.join(self.model_rdms_path, layer))
    

            # Calculate Correlations
            r2, significance, sem, corr_squared = self.rsa(model_rdm, roi_rdm, layer)

            # Add relationship to Noise Ceiling to this data
            lnc = self.this_nc["lnc"]
            unc = self.this_nc["unc"]
            area_percentNC = (r2 / lnc) * 100.

            # Create dictionary to save data
            layer_key = "(" + str(counter) + ") " + layer
            output_dict = {"Layer": [layer_key],
                           "R2": [r2],
                           "%R2": [area_percentNC],
                           "Significance": [significance],
                           "SEM": [sem],
                           "LNC": [lnc],
                           "UNC": [unc],
                           "R2_array" : corr_squared}

            # Add this dict to the total dickt
            all_layers_dicts.append(output_dict)

        return all_layers_dicts

    def evaluate(self,correction=None):
        """Function to evaluate all DNN RDMs to all ROI RDMs
        Returns:
            dict: final dict containing all results
        """

        all_rois_df = pd.DataFrame(columns=['ROI', 'Layer', "Model", 'R2', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])

        for counter, roi in enumerate(self.brain_rdms):

            self.find_datatype(roi)

            # Calculate Noise Ceiing for this ROI
            self.this_nc = NoiseCeiling(roi, op.join(self.brain_rdms_path, roi)).noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            all_layers_dict = self.evaluate_roi(roi)

            # Create dict with these results
            scan_key = "(" + str(counter) + ") " + roi[:-4]

            for layer_dict in all_layers_dict:
                layer_dict["ROI"] = scan_key
                layer_dict["Model"] = self.model_name
                del layer_dict["R2_array"]
                layer_df = pd.DataFrame.from_dict(layer_dict)
                if correction == "bonferroni":
                    layer_df['Significance'] = layer_df['Significance'] * len(all_layers_dict)
                all_rois_df = pd.concat([all_rois_df, layer_df], ignore_index=True)
            
        return all_rois_df
        
    def compare_model(self,other_RSA):
        """Function to evaluate all DNN RDMs to all ROI RDMs
        Returns:
            dict: final dict containing all results
        """
        
        comp_dic = dict()
        sig_pairs = []
	
        for counter, roi in enumerate(self.brain_rdms):

            self.find_datatype(roi)

            # Calculate Noise Ceiing for this ROI
            self.this_nc = NoiseCeiling(roi, op.join(self.brain_rdms_path, roi)).noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            model_layers_dict = self.evaluate_roi(roi)
            

            # Calculate Noise Ceiing for this ROI
            other_RSA.this_nc = NoiseCeiling(roi, op.join(other_RSA.brain_rdms_path, roi)).noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            other_layers_dict = other_RSA.evaluate_roi(roi)
            
            model_ii = np.argmin([layer_dict["R2"] for layer_dict in model_layers_dict])            
            other_ii = np.argmin([layer_dict["R2"] for layer_dict in other_layers_dict])
            
            tstat,p = stats.ttest_ind(other_layers_dict[other_ii]["R2_array"],model_layers_dict[model_ii]["R2_array"])
            
            scan_key = "(" + str(counter) + ") " + roi[:-4]
               
            comp_dic[scan_key] = (tstat,p)
            if p < 0.5:
                sig_pair = sorted(((scan_key,self.model_name),(scan_key,other_RSA.model_name)), key=lambda element: (element[1]))
                sig_pairs.append(sig_pair)
        return comp_dic,sig_pairs
