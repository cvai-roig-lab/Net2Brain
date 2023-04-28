import numpy as np
import os
from scipy import stats
import os.path as op
import rsatoolbox
from .noiseceiling import NoiseCeiling
from .eval_helper import *
from sklearn.model_selection import KFold
import pandas as pd


class WRSA():
    """Evaluation with RSA
    """

    def __init__(self, model_rdms_path, brain_rdms_path, model_name, datatype="None", save_path="./", distance_metric="Euclidian"):
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
        self.distance_metric = distance_metric
        self.model_name = model_name

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
            if not ".json" and ".DS_Store" not in f:
                if f != ".ipynb_checkpoints":
                    file_sets.append(f)

        return file_sets

    def create_weighted_model(self, this_roi):
        """Create Weighted RSA model with all layers as training data

        Returns:
            out (dict): {layername: [r2, area_percentNC, significance, sem, [lnc, unc]]}
        """

        """Create list with only the upper triangles of the layer RDMs"""
        layers_upper = []
        for layer in self.model_rdms:
            layer_dict = load(op.join(self.model_rdms_path, layer))
            layer_RDM = layer_dict["arr_0"]
            layers_upper.append(self.get_uppertriangular(layer_RDM))

        """Do the same with the current ROI"""
        roi_dict = load(op.join(self.brain_rdms_path, this_roi))
        roi_RDM = roi_dict["arr_0"]

        """Turn ROIs into arrays of upper triangle"""
        roi_upper = []

        if "meg" in this_roi.lower():  # Mean the MEG_RDMs first
            roi_RDM = np.mean(roi_RDM, axis=1)

        for roi in roi_RDM:
            roi_upper.append(self.get_uppertriangular(roi))

        """Turn layers into RSA Dataset"""
        layers_upper = np.array(layers_upper)
        layer_RSA_dataset = rsatoolbox.rdm.RDMs(layers_upper, dissimilarity_measure=self.distance_metric)

        """ Create permuations for cross validation"""
        range_rois = list(range(len(roi_upper)))

        """Find 80% train test split"""
        splitter = int(len(roi_upper) * 0.2)

        if splitter == 0:
            splitter = 2

        """Do KFold"""
        kf = KFold(n_splits=splitter, shuffle=True, random_state=1)

        all_correlations = []

        for train_index, test_index in kf.split(range_rois):

            all_roi_upper = np.array(roi_upper.copy())

            test_roi = all_roi_upper[test_index]
            train_roi = all_roi_upper[train_index]

            brain_test_dataset = rsatoolbox.rdm.RDMs(test_roi, dissimilarity_measure=self.distance_metric)
            brain_train_dataset = rsatoolbox.rdm.RDMs(train_roi, dissimilarity_measure=self.distance_metric)

            """Initiate Model with the layer rdms"""
            model = rsatoolbox.model.ModelWeighted('testModel', layer_RSA_dataset)

            """Optimize model with the brain RDMs"""
            theta_corr_regress = rsatoolbox.model.fit_regress(
                model, brain_train_dataset, method='corr')  # returns 15 thetas because 15 participants

            """Prediciton"""
            rdm_corr = model.predict_rdm(theta_corr_regress)

            """Calculate Correlation"""
            corr = rsatoolbox.rdm.compare(
                rdm_corr, brain_test_dataset, 'corr')

            """ Get R² """

            corr_squared = np.square(corr[0])
            r2 = np.mean(corr_squared)

            all_correlations.append(r2)

        """Get final R²"""
        r2 = np.mean(all_correlations)

        """ Get the significance """
        significance = stats.ttest_1samp(all_correlations, 0)[1]

        """ Standard Error of mean """
        sem = stats.sem(all_correlations)  # standard error of mean

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

            # Calculate Noise Ceiing for this ROI
            self.this_nc = NoiseCeiling(roi, op.join(self.brain_rdms_path, roi)).noise_ceiling()

            # Return Correlation Values for this ROI to all model layers
            layer_dict = self.create_weighted_model(roi)

            # Create dict with these results
            scan_key = "(" + str(counter) + ") " + roi[:-4]
            layer_dict["ROI"] = scan_key
            layer_dict["Model"] = self.model_name
            layer_df = pd.DataFrame.from_dict(layer_dict)
            all_rois_df = pd.concat([all_rois_df, layer_df], ignore_index=True)

        return all_rois_df
