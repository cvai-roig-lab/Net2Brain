import numpy as np
import os
from scipy import stats
import os.path as op
from .noiseceiling import NoiseCeiling
from .eval_helper import *


class RSA():
    """Evaluation with RSA
    """

    def __init__(self, model_rdms_path, brain_rdms_path, datatype="None", save_path="./", distance_metric="spearman"):
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
            if ".json" not in f:
                if f != ".ipynb_checkpoints":
                    file_sets.append(f)

        return file_sets

    def rsa_meg(self, model_rdm, brain_rdm, layername):
        """Creates the output dictionary for MEG scans. Returns {layername: R², Significance}

        Args:
            model_rdm (numpy array): DNN rdm
            brain_rdm (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]

        Returns:
            dict: {layername: [r2, significance, sem]}
        """

        key = "arr_0"  # This is the standard key when saving npz files with savez
        model_rdm = model_rdm[key]
        meg_rdm = brain_rdm[key]

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

        return r2, significance, sem

    def rsa_fmri(self, model_rdm, brain_rdm, layername):
        """Creates the output dictionary for fMRI scans. Returns {layername: R², Significance}

        Args:
            model_rdm (numpy array): DNN rdm
            brain_rdm (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]

        Returns:
            dict: {layername: [r2, significance, sem]}
        """

        key = "arr_0"  # This is the standard key when saving npz files with savez
        model_rdm = model_rdm[key]
        fmri_rdm = brain_rdm[key]

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

        return r2, significance, sem

    def evaluate_layer(self, roi):
        """Functiion to evaulate the scans to the current layer , either fmri or meg

        Returns:
            dict: dictionary of all results to the current layer
        """

        all_layers_dict = {}

        # For each layer to RSA with the current ROI
        for counter, layer in enumerate(self.model_rdms):

            # Load RDMS
            roi_rdm = load(op.join(self.brain_rdms_path, roi))
            model_rdm = load(op.join(self.model_rdms_path, layer))

            # Calculate Correlations
            r2, significance, sem = self.rsa(model_rdm, roi_rdm, layer)

            # Add relationship to Noise Ceiling to this data
            lnc = self.this_nc["lnc"]
            unc = self.this_nc["unc"]
            area_percentNC = (r2 / lnc) * 100.

            # Create dictionary to save data
            layer_key = "(" + str(counter) + ") " + layer
            output_dict = {layer_key: [r2, area_percentNC, significance, sem, [lnc, unc]]}

            # Add this dict to the total dickt
            all_layers_dict.update(output_dict)

        return all_layers_dict

    def evaluate(self):
        """Function to evaluate all DNN RDMs to all ROI RDMs

        Returns:
            dict: final dict containing all results
        """

        all_rois_dict = {}

        for counter, roi in enumerate(self.brain_rdms):

            self.find_datatype(roi)

            # Calculate Noise Ceiing for this ROI
            self.this_nc = NoiseCeiling(roi, op.join(self.brain_rdms_path, roi)).noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            results_roi = self.evaluate_layer(roi)

            # Create dict with these results
            scan_key = "(" + str(counter) + ") " + roi[:-4]
            scan_dict = {scan_key: results_roi}
            all_rois_dict.update(scan_dict)

        return all_rois_dict
