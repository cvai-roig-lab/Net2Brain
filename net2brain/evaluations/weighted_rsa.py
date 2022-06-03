import numpy as np
import os
from scipy import stats
from scipy import stats
import json
import os.path as op
import rsatoolbox
from .noiseceiling import NoiseCeiling
from .eval_helper import *
from helper.helper import get_paths
from sklearn.model_selection import KFold


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


    
class WRSA():
    """Evaluation with weighted RSA
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
        
    
    def create_weighted_model(self):
        """Create Weighted RSA model with all layers as training data
        
        Returns:
            out (dict): {layername: [r2, area_percentNC, significance, sem, [lnc, unc]]}
        """
        
        """Get all paths to the layers and sort them"""
        layer_folder = op.join(RDMS_DIR, self.dataset, self.this_net)
        layers = self.folderlookup(layer_folder)
        layers.sort(key=natural_keys)
        
        """Create list with only the upper triangles of the layer RDMs"""
        layers_upper = []
        for layer in layers:
            layer_dict = load(op.join(layer_folder, layer))
            layer_RDM = layer_dict["arr_0"]
            layers_upper.append(self.get_uppertriangular(layer_RDM))
        #layers_upper = np.array(layers_upper)
        
        """Do the same with the current ROI"""
        roi_dict = load(op.join(BRAIN_DIR, self.dataset, self.this_roi))
        roi_RDM = roi_dict["arr_0"]
        
        """Turn ROIs into arrays of upper triangle"""
        roi_upper = []
        
        if "meg" in self.this_roi.lower(): # Mean the MEG_RDMs first
            roi_RDM = np.mean(roi_RDM, axis=1)
        
        for roi in roi_RDM:
            roi_upper.append(self.get_uppertriangular(roi))
        # roi_upper = np.array(roi_upper)
        
        
        """Turn into RSA Dataset"""
        # layer_RSA_dataset = rsatoolbox.rdm.RDMs(
        #     layers_upper, dissimilarity_measure='Euclidean')
        # brain_RSA_dataset = rsatoolbox.rdm.RDMs(
        #     roi_upper, dissimilarity_measure='Euclidean')
        
        
        
        """Depending on type of WRSA, train on brain or on model data"""
        
        if self.metric == "Weighted RSA":
            
            """Turn layers into RSA Dataset"""
            layers_upper = np.array(layers_upper)
            layer_RSA_dataset = rsatoolbox.rdm.RDMs(layers_upper, dissimilarity_measure='Euclidean')
            
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
                
                brain_test_dataset = rsatoolbox.rdm.RDMs(test_roi, dissimilarity_measure='Euclidean')
                brain_train_dataset = rsatoolbox.rdm.RDMs(train_roi, dissimilarity_measure='Euclidean')
                
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

            """ Finish Dict """
            layername = "(0) " + self.this_net.split("_")[1] + "_pred"
            
            output_dict = {layername: [
                r2, area_percentNC, significance, sem, [self.this_nc["lnc"], self.this_nc["unc"]]]}

            return output_dict
        
        elif self.metric == "Weighted RSA (predicting layers)":
            
            pass
            
        #     """Initiate Model with the brain rdms"""
        #     model = rsatoolbox.model.ModelWeighted('testModel', brain_RSA_dataset)
            
        #     """Optimize model with the layer RDMs"""
        #     theta_corr_regress = rsatoolbox.model.fit_regress(
        #         model, layer_RSA_dataset, method='corr')  # returns 15 thetas because 15 participants
        
        
        # """Prediciton"""
        # rdm_corr = model.predict_rdm(theta_corr_regress)

        # """Calculate Correlation"""
        # corr = rsatoolbox.rdm.compare(
        #     rdm_corr, brain_RSA_dataset, 'corr') 

        # """ Get R² """
        # print(corr)
        # corr_squared = np.square(corr[0])
        # r2 = np.mean(corr_squared)
        
        # """ Get the significance """
        # significance = stats.ttest_1samp(corr_squared, 0)[1]

        # """ Standard Error of mean """
        # sem = stats.sem(corr_squared)  # standard error of mean
        
        # """ Percentage of NC """
        # area_percentNC = (r2 / self.this_nc["lnc"]) * 100.

        # """ Finish Dict """
        # layername = "(0) " + self.this_net.split("_")[1] + "_pred"
        
        # output_dict = {layername: [
        #     r2, area_percentNC, significance, sem, [self.this_nc["lnc"], self.this_nc["unc"]]]}

        # return output_dict
            

    def evaluation(self):
        """Function to evaluate all DNN RDMs to all ROI RDMs

        Returns:
            dict: final dict containing all results 
        """
        
        dict_of_each_roi = {}
        
        for roi_counter, roi in enumerate(self.rois):  # for each region of interest
            dict_of_this_roi = {}
            self.this_roi = roi        
            self.roi_path = op.join(BRAIN_DIR, self.dataset, roi)
            
            for net_counter, network in enumerate(self.networks):  # for each network selected
                self.this_net = network
                self.this_nc = NoiseCeiling(self.this_roi, self.roi_path).noise_ceiling()
                
                output_dict = self.create_weighted_model()  # Calculate R²
                
                """Update Dicts"""
                network_key = "(" + str(net_counter) + ") " + network
                network_dict = {network_key: output_dict}
                dict_of_this_roi.update(network_dict)
                
            scan_key = "(" + str(roi_counter) + ") " + roi[:-4]
            scan_dict = {scan_key: dict_of_this_roi}
            
            if self.dataset in dict_of_each_roi:
                dict_of_each_roi[self.dataset].update(scan_dict)
            else:
                dict_of_each_roi.update({self.dataset: scan_dict})
            
        return dict_of_each_roi
