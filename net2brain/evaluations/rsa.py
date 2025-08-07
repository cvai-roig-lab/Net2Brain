import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform, euclidean, cityblock, cosine
from .noiseceiling import NoiseCeiling
from .eval_helper import *
from .distance_functions import registered_distance_functions

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class RSA():
    """Evaluation with RSA
    """

    def __init__(self, model_rdms_path, brain_rdms_path, model_name, layer_skips=(),
                 squared=True, timepoint_agg=True, model_timepoints=False, datatype="None", save_path="./"):
        """Initiate RSA

        Args:
            model_rdms_path (str): Path to the folder containing the model RDMs.
            brain_rdms_path (str): Path to the folder containing the brain RDMs.
            model_name (str): Name of the model.
            layer_skips (tuple, optional): Names of the model layers to skip. Use '_' instead of '.' in the names.
            squared (bool): Whether to square the correlation values.
            timepoint_agg (bool): Whether to aggregate timepoints for MEG/EEG data.
            model_timepoints (bool): Whether the model RDMs contain timepoints.
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
        self.layer_skips = layer_skips
        self.squared = squared
        self.timepoint_agg = timepoint_agg
        self.model_timepoints = model_timepoints

        # For comparison
        self.other_rdms_path = None
        self.other_rdms = None

    def find_datatype(self, file_path):
        """Function to determine if the input data corresponds to fMRI or MEG based on data shape.

        Args:
            file_path (str): Path to the numpy file (.npz) containing ROI data.
        """
        # Load the .npz file
        data = np.load(file_path, allow_pickle=True)

        # Get the first key from the loaded file
        keys = list(data.keys())
        if not keys:
            raise ValueError("The provided .npz file is empty.")

        rdm_data = data[keys[0]]

        # Get the shape of the RDM data
        shape = rdm_data.shape

        if len(shape) == 2 and shape[0] == shape[1]:
            # fMRI data with shape (images, images)
            self.rsa = self.rsa_fmri

        elif len(shape) == 3:
            if shape[1] == shape[2]:
                # fMRI data with shape (subjects, images, images)
                self.rsa = self.rsa_fmri
            else:
                raise ValueError(f"Invalid fMRI data shape: {shape}. Last two dimensions must be equal.")

        elif len(shape) == 4:
            if shape[2] == shape[3]:
                # MEG data with shape (subjects, times, images, images)
                self.rsa = self.rsa_meg_eeg
            else:
                raise ValueError(f"Invalid MEG data shape: {shape}. Last two dimensions must be equal.")

        else:
            raise ValueError(f"Unexpected data shape: {shape}. Expected 2D, 3D, or 4D array.")

        if (len(shape) == 3 and shape[1] != shape[2]) or (len(shape) == 4 and shape[2] != shape[3]):
            warnings.warn("The last two dimensions of the data do not match, which may indicate a problem.")

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

    def get_rnames(self):
        """Determine the names of the R variables based on the squared parameter"""
        r_name = "R" if not self.squared else "R2"
        r_percent_name = "%R" if not self.squared else "%R2"
        r_array_name = "R_array" if not self.squared else "R2_array"
        return r_name, r_percent_name, r_array_name

    def rsa_meg_eeg(self, model_rdm, brain_rdm, layername):
        """Creates the output dictionary for MEG or EEG scans.
        Args:
            model_rdm (numpy array): DNN rdm
            brain_rdm (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]
        Returns:
            dict: {layername:  [r, significance, sem, corr_list]}
        """

        key = list(model_rdm.keys())[0]  # You need to access the keys to open a npy file
        model_rdm = model_rdm[key]
        key = list(brain_rdm.keys())[0]  # You need to access the keys to open a npy file
        meg_rdm = brain_rdm[key]

        if not self.model_timepoints:
            model_rdm = self.check_squareform(
                model_rdm)  # Check if rdm is squareform #TODO Remove soon after reimplementing RSA

            # returns list of corrcoefs, depending on amount of participants in brain rdm
            corr = np.array([self.distance(model_rdm, rdms) for rdms in meg_rdm])
            if self.timepoint_agg:
                # Aggregate timepoints by taking the mean across the second dimension
                corr = np.mean(corr, axis=1)

            if self.squared:
                # Square correlation
                corr_list = np.square(corr)
            else:
                corr_list = corr

            # Take mean
            r = np.mean(corr_list, axis=0)

            # ttest: Ttest_1sampResult(statistic=3.921946, pvalue=0.001534)
            significance = stats.ttest_1samp(corr_list, 0)[1]
            # standard error of mean
            sem = stats.sem(corr_list)  # standard error of mean
        else:
            if model_rdm.ndim == 3 and not model_rdm.shape[-1] == model_rdm.shape[-2]:
                # this means that timepoints are divided in 2
                model_rdm = model_rdm.reshape(-1, model_rdm.shape[-1])

            timepoint_rs = []
            timepoint_corrs = []
            for model_rdm_timepoint in model_rdm:
                model_rdm_timepoint = self.check_squareform(model_rdm_timepoint)
                corr = np.array([self.distance(model_rdm_timepoint, rdms) for rdms in meg_rdm])
                if self.timepoint_agg:
                    corr = np.mean(corr, axis=1)
                if self.squared:
                    corr_list = np.square(corr)
                else:
                    corr_list = corr
                r = np.mean(corr_list, axis=0)
                timepoint_rs.append(r)
                timepoint_corrs.append(corr_list)
            r = np.array(timepoint_rs)
            corr_list = np.array(timepoint_corrs)
            significance = sem = None

        r = r.tolist() if isinstance(r, np.ndarray) else r
        corr_list = corr_list.tolist()
        return r, significance, sem, corr_list

    def rsa_fmri(self, model_rdm, brain_rdm, layername):
        """Creates the output dictionary for fMRI scans.
        Args:
            model_rdm (numpy array): DNN rdm
            brain_rdm (list of numpy arrays): Subjects RDMs
            layername ([type]): [description]
        Returns:
            dict: {layername: [r, significance, sem, corr_list]}
        """

        key = list(model_rdm.keys())[0]  # You need to access the keys to open a npy file
        model_rdm = model_rdm[key]
        key = list(brain_rdm.keys())[0]  # You need to access the keys to open a npy file
        fmri_rdm = brain_rdm[key]

        model_rdm = self.check_squareform(
            model_rdm)  # Check if rdm is squareform #TODO Remove soon after reimplementing RSA

        # returns list of corrcoefs, depending on amount of participants in brain rdm
        corr = self.distance(model_rdm, fmri_rdm)

        if self.squared:
            # Square correlation
            corr_list = np.square(corr)
        else:
            corr_list = corr

        # Take mean
        r = np.mean(corr_list)

        # ttest: Ttest_1sampResult(statistic=3.921946, pvalue=0.001534)
        significance = stats.ttest_1samp(corr_list, 0)[1]

        # standard error of mean
        sem = stats.sem(corr_list)  # standard error of mean

        return r, significance, sem, corr_list

    def evaluate_roi(self, roi):
        """Functiion to evaulate the layers to the current roi , either fmri or meg
        Returns:
            dict: dictionary of all results to the current roi
        """

        all_layers_dicts = []

        # For each layer to RSA with the current ROI
        for counter, layer in enumerate(self.model_rdms):
            layer_name = layer.split("RDM_")[1].split(".npz")[0] if "RDM_" in layer else layer.split(".npz")[0]
            if layer_name in self.layer_skips:
                continue

            # Load RDMS
            roi_rdm = load(op.join(self.brain_rdms_path, roi))
            model_rdm = load(op.join(self.model_rdms_path, layer))

            # Calculate Correlations
            r, significance, sem, corr = self.rsa(model_rdm, roi_rdm, layer)

            # Add relationship to Noise Ceiling to this data
            lnc = self.this_nc["lnc"]
            unc = self.this_nc["unc"]
            area_percentNC = (r / lnc) * 100.

            # Create dictionary to save data
            layer_key = "(" + str(counter) + ") " + layer
            r_name, r_percent_name, r_array_name = self.get_rnames()
            output_dict = {"Layer": [layer_key],
                           r_name: [r],
                           r_percent_name: [area_percentNC],
                           "Significance": [significance],
                           "SEM": [sem],
                           "LNC": [lnc],
                           "UNC": [unc],
                           r_array_name: [corr]}

            # Add this dict to the total dict
            all_layers_dicts.append(output_dict)

        return all_layers_dicts

    def evaluate(self, correction=None, distance_metric="spearman") -> pd.DataFrame:
        """Function to evaluate all DNN RDMs to all ROI RDMs
        Returns:
            dict: final dict containing all results
        """

        # Convert to lowercase for case-insensitive matching
        self.distance_metric = distance_metric.lower()

        # Check if the requested distance metric exists in the registered functions
        if distance_metric in registered_distance_functions:
            self.distance = registered_distance_functions[distance_metric]
        else:
            # Dynamically generate a list of available metrics for the warning
            available_metrics = list(registered_distance_functions.keys())
            warnings.warn(f"Invalid metric. Choose between: {', '.join(available_metrics)}")
            return None

        r_name, r_percent_name, r_array_name = self.get_rnames()
        all_rois_df = pd.DataFrame(
            columns=['ROI', 'Layer', "Model", r_name, r_percent_name, r_array_name, 'Significance', 'SEM', 'LNC', 'UNC'])

        for counter, roi in enumerate(self.brain_rdms):

            self.find_datatype(op.join(self.brain_rdms_path, roi))

            # Calculate Noise Ceiing for this ROI
            noise_ceiling_calc = NoiseCeiling(roi, op.join(self.brain_rdms_path, roi), distance_metric, self.squared)
            self.this_nc = noise_ceiling_calc.noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            all_layers_dict = self.evaluate_roi(roi)

            # Create dict with these results
            scan_key = "(" + str(counter) + ") " + roi[:-4]

            for layer_dict in all_layers_dict:
                layer_dict["ROI"] = [scan_key]
                layer_dict["Model"] = [self.model_name]
                layer_df = pd.DataFrame.from_dict(layer_dict)
                if correction == "bonferroni":
                    layer_df['Significance'] = layer_df['Significance'] * len(all_layers_dict)
                all_rois_df = pd.concat([all_rois_df, layer_df], ignore_index=True)

        return all_rois_df

    def compare_model(self, other_RSA):
        """Function to evaluate all DNN RDMs to all ROI RDMs
        Returns:
            dict: final dict containing all results
        """

        comp_dic = dict()
        sig_pairs = []

        for counter, roi in enumerate(self.brain_rdms):

            self.find_datatype(op.join(self.brain_rdms_path, roi))

            # Calculate Noise Ceiing for this ROI
            noise_ceiling_calc = NoiseCeiling(roi, op.join(self.brain_rdms_path, roi), self.distance_metric)
            self.this_nc = noise_ceiling_calc.noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            model_layers_dict = self.evaluate_roi(roi)

            # Calculate Noise Ceiing for this ROI
            other_RSA.this_nc = NoiseCeiling(roi, op.join(other_RSA.brain_rdms_path, roi),
                                             self.distance_metric).noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            other_layers_dict = other_RSA.evaluate_roi(roi)

            r_name, _, r_array_name = self.get_rnames()

            model_ii = np.argmin([layer_dict[r_name] for layer_dict in model_layers_dict])
            other_ii = np.argmin([layer_dict[r_name] for layer_dict in other_layers_dict])

            tstat, p = stats.ttest_ind(other_layers_dict[other_ii][r_array_name], model_layers_dict[model_ii][r_array_name])

            scan_key = "(" + str(counter) + ") " + roi[:-4]

            comp_dic[scan_key] = (tstat, p)
            if p < 0.5:
                sig_pair = sorted(((scan_key, self.model_name), (scan_key, other_RSA.model_name)),
                                  key=lambda element: (element[1]))
                sig_pairs.append(sig_pair)
        return comp_dic, sig_pairs

