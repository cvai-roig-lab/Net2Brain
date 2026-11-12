import os
import os.path as op
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import nnls
import pandas as pd
from scipy.spatial.distance import squareform
from .noiseceiling import NoiseCeiling
from .eval_helper import *


class WRSA():
    """Evaluation with RSA
    """

    def __init__(self, model_rdms_path, brain_rdms_path, model_name, squared=True, datatype="None", save_path="./", distance_metric="Euclidean", ridge_weight=0):
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
        self.distance_metric = distance_metric.lower()
        self.model_name = model_name
        self.squared = squared
        self.ridge_weight = ridge_weight
        if ridge_weight != 0:
            warnings.warn("ridge_weight is not used: the layer weights are fit with non-negative least squares.")

    def get_uppertriangular(self, rdm):
        """Get upper triangle of a RDM

        Args:
            rdm (numpy array): nxn rdm

        Returns:
            numpy array: only the upper triangle of rdm
        """

        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]

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

    def create_weighted_model(self, this_roi):
        """Create Weighted RSA model with all layers as training data

        The stimuli are split once at random (seeded) into folds. For every fold and every held-out subject, the
        non-negative layer weights (least squares with intercept) are fit on the mean RDM of the other subjects,
        using only the pairs of the training stimuli, and the weighted RDM is correlated (Pearson) with the
        held-out subject's RDM on the pairs of the held-out stimuli. R2 is the mean over subjects of r² (r averaged
        over the folds), Significance a one-sided t-test of r against 0 over subjects. The noise ceiling (Pearson)
        is computed on the same held-out pairs and aggregated like R2. MEG/EEG RDMs are averaged over time first.
        R2 and p depend on the random split; for weak effects they can differ considerably between splits.

        Returns:
            out (dict): {layername: [r2, area_percentNC, significance, sem, [lnc, unc]]}
        """

        """Create list with only the upper triangles of the layer RDMs"""
        layers_upper = []
        for layer in self.model_rdms:
            layer_dict = load(op.join(self.model_rdms_path, layer))
            key = list(layer_dict.keys())[0]  # You need to access the keys to open a npy file
            layer_RDM = layer_dict[key]
            layer_RDM = self.check_squareform(layer_RDM) # Check if rdm is squareform #TODO Remove soon after reimplementing RSA
            
            layers_upper.append(self.get_uppertriangular(layer_RDM))

        """Do the same with the current ROI"""
        roi_dict = load(op.join(self.brain_rdms_path, this_roi))
        keys = list(roi_dict.keys())
        roi_RDM = roi_dict[keys[0]]

        """Turn ROIs into arrays of upper triangle"""
        roi_upper = []

        if roi_RDM.ndim == 4:  # MEG/EEG [subjects, time, stimuli, stimuli]: mean the RDMs over time first
            roi_RDM = np.mean(roi_RDM, axis=1)

        for roi in roi_RDM:
            roi_upper.append(self.get_uppertriangular(roi))

        layers_upper = np.array(layers_upper)
        roi_upper = np.array(roi_upper)

        """Cross-validation: one seeded random split of the stimuli into folds x leave-one-subject-out"""
        n_stim = roi_RDM.shape[-1]
        rows, cols = np.triu_indices(n_stim, 1)
        folds = np.array_split(np.random.default_rng(0).permutation(n_stim), min(5, n_stim // 3))
        r = np.zeros((len(roi_upper), len(folds)))
        lnc, unc = np.zeros_like(r), np.zeros_like(r)  # noise ceiling on the same held-out pairs as r
        for f, test_stimuli in enumerate(folds):
            is_test = np.isin(np.arange(n_stim), test_stimuli)
            train, test = ~is_test[rows] & ~is_test[cols], is_test[rows] & is_test[cols]
            X = layers_upper[:, train].T
            X = X - X.mean(axis=0)  # free intercept
            for s in range(len(roi_upper)):
                others = np.delete(roi_upper, s, axis=0)
                y = others[:, train].mean(axis=0)
                prediction = layers_upper[:, test].T @ nnls(X, y - y.mean())[0]
                # All weights 0: the model predicts nothing for this fold
                r[s, f] = np.corrcoef(prediction, roi_upper[s, test])[0, 1] if np.ptp(prediction) > 0 else 0.0
                lnc[s, f] = np.corrcoef(others[:, test].mean(axis=0), roi_upper[s, test])[0, 1]
                unc[s, f] = np.corrcoef(roi_upper[:, test].mean(axis=0), roi_upper[s, test])[0, 1]
        r = r.mean(axis=1)
        # Pearson noise ceiling, aggregated like R2 (r averaged over folds, squared, mean over subjects)
        self.this_nc = {"lnc": np.mean(lnc.mean(axis=1) ** 2), "unc": np.mean(unc.mean(axis=1) ** 2)}

        """Get final R²"""
        r2 = np.mean(r ** 2)

        """ Get the significance """
        significance = stats.ttest_1samp(r, 0, alternative="greater")[1]

        """ Standard Error of mean """
        sem = stats.sem(r ** 2)  # standard error of mean

        """ Percentage of NC """
        area_percentNC = (r2 / self.this_nc["lnc"]) * 100.

        output_dict = {"Layer": ["All"],
                       "R2": [r2],
                       "%R2": [area_percentNC],
                       "Significance": [significance],
                       "SEM": [sem],
                       "LNC": [self.this_nc["lnc"]],
                       "UNC": [self.this_nc["unc"]]}
        return output_dict

    def evaluate(self):
        """Function to evaluate all DNN RDMs to all ROI RDMs

        Returns:
            dict: final dict containing all results
        """

        all_rois_df = pd.DataFrame(columns=['ROI', 'Layer', "Model", 'R2', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])

        for counter, roi in enumerate(self.brain_rdms):

            # Return Correlation Values for this ROI to all model layers
            layer_dict = self.create_weighted_model(roi)

            # Create dict with these results
            scan_key = "(" + str(counter) + ") " + roi[:-4]
            layer_dict["ROI"] = scan_key
            layer_dict["Model"] = self.model_name
            layer_df = pd.DataFrame.from_dict(layer_dict)
            all_rois_df = pd.concat([all_rois_df, layer_df], ignore_index=True)

        return all_rois_df
