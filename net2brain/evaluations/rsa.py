import numpy as np
import os
from scipy import stats
from scipy import stats
import json
import os.path as op
from .noiseceiling import NoiseCeiling
from .eval_helper import *
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

class RSA():
    """Evaluation with RSA
    """

    def __init__(self, json_dir):
        """Initiate RSA

        Args:
            json_dir (str/path): Path to json dir
        """
        json_file = op.join(json_dir, "args.json")

        with open(json_file) as json_file:
            data = json.load(json_file)[0]

        self.json_dir = json_dir
        self.day = data["1_Day"]
        self.time = data["2_Time"]
        self.networks = data["3_Networks"]
        self.dataset = data["4_Dataset"]
        self.rois = data["5_ROIs"]
        self.metric = data["6_Metric"]
        
        self.test = False

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
            if f != "args.json":
                if f != ".ipynb_checkpoints":
                    file_sets.append(f)

        return file_sets

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

    def evaluate_fmri(self, submission, targets, layername):
        """Creates the output dictionary for fMRI scans. Returns {layername: R², Significance}

        Args:
            submission (numpy array): DNN rdm
            targets (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]

        Returns:
            dict: {layername: [r2, significance, sem]}
        """

        key = "arr_0"  # This is the standard key when saving npz files with savez
        model_rdm = submission[key]
        fmri_rdm = targets[key]

        # returns list of corrcoefs, depending on amount of participants in brain rdm
        corr = self.model_spearman(model_rdm, fmri_rdm)

        corr_squared = np.square(corr)

        r2 = np.mean(corr_squared)

        significance = stats.ttest_1samp(corr_squared, 0)[1]
        # ttest: Ttest_1sampResult(statistic=3.921946, pvalue=0.001534)

        sem = stats.sem(corr_squared)  # standard error of mean

        out = {layername: [r2, significance, sem]}

        return out

    def evaluate_meg(self, submission, targets, layername):
        """Creates the output dictionary for MEG scans. Returns {layername: R², Significance}

        Args:
            submission (numpy array): DNN rdm
            targets (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]

        Returns:
            dict: {layername: [r2, significance, sem]}
        """

        key = "arr_0"  # This is the standard key when saving npz files with savez
        model_rdm = submission[key]
        meg_rdm = targets[key]

        corr = np.mean([self.model_spearman(model_rdm, rdms)
                       for rdms in meg_rdm], 1)

        corr_squared = np.square(corr)

        r2 = np.mean(corr_squared)

        significance = stats.ttest_1samp(corr_squared, 0)[1]

        sem = stats.sem(corr_squared)  # standard error of mean

        out = {layername: [r2, significance, sem]}

        return out

    def evaluate_scan(self):
        """Functiion to evaulate the scans to the current layer , either fmri or meg

        Returns:
            dict: dictionary of all results to the current layer
        """

        print("Calculating " + self.this_net)

        #Find all layers from folder
        if self.test:
            layer_folder = op.join(CURRENT_DIR, "net2brain/tests/compare_files/correct_data_rdm")
        else:
            layer_folder = op.join(RDMS_DIR, self.dataset, self.this_net)
            
        layers = self.folderlookup(layer_folder)
        layers.sort(key=natural_keys)

        all_layers_dict = {}

        for counter, layer in enumerate(layers):
            target = load(self.roi_path)
            submit = load(op.join(layer_folder, layer))

            if "fmri" in self.this_roi.lower():
                # {layername: [r2, significance, sem]}
                out = self.evaluate_fmri(submit, target, layer)
            elif "meg" in self.this_roi.lower():
                # {layername: [r2, significance, sem]}
                out = self.evaluate_meg(submit, target, layer)
            else:
                error_message("No fmri/meg found in ROI-name. Error!")

            # Create final dictionary
            r2 = out[layer][0]
            lnc = self.this_nc["lnc"]
            unc = self.this_nc["unc"]
            area_percentNC = (r2 / lnc) * 100.
            sig_value = out[layer][1]
            sem = out[layer][2]

            layer_key = "(" + str(counter) + ") " + layer

            output_dict = {layer_key: [
                r2, area_percentNC, sig_value, sem, [lnc, unc]]}

            all_layers_dict.update(output_dict)

        return all_layers_dict

    def evaluation(self):
        """Function to evaluate all DNN RDMs to all ROI RDMs

        Returns:
            dict: final dict containing all results
        """

        dict_of_each_roi = {}

        for roi_counter, roi in enumerate(self.rois):

            dict_of_this_roi = {}

            self.roi_path = op.join(BRAIN_DIR, self.dataset, roi)
            self.this_roi = roi
            self.this_nc = NoiseCeiling(
                self.this_roi, self.roi_path).noise_ceiling()

            for net_counter, network in enumerate(self.networks):
                self.this_net = network
                all_layers_dict = self.evaluate_scan()

                network_key = "(" + str(net_counter) + ") " + network
                network_dict = {network_key: all_layers_dict}

                dict_of_this_roi.update(network_dict)

            scan_key = "(" + str(roi_counter) + ") " + roi[:-4]
            scan_dict = {scan_key: dict_of_this_roi}

            if self.dataset in dict_of_each_roi:
                dict_of_each_roi[self.dataset].update(scan_dict)
            else:
                dict_of_each_roi.update({self.dataset: scan_dict})

        return dict_of_each_roi
