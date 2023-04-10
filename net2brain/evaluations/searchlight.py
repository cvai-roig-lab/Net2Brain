import numpy as np
import os
from scipy import stats
import torch
import os.path as op
from .eval_helper import *


class Searchlight():

    def __init__(self, model_rdms_path, searchlight_file, datatype="None", save_path="./", distance_metric="pearson"):
        """Initiate RSA

        Args:
            json_dir (str/path): Path to json dir
        """

        # Find all model RDMs
        self.model_rdms_path = model_rdms_path
        self.model_rdms = self.folderlookup(model_rdms_path)
        self.model_rdms.sort(key=natural_keys)

        # Open Searchlight Path
        searchlight_rdm = np.load(searchlight_file)['arr_0']
        self.searchlight_rdm = searchlight_rdm.transpose(1, 0, 2, 3)

        # Other parameters
        self.save_path = save_path
        self.datatype = datatype

        if distance_metric.lower() == "pearson":
            self.distance = self.pearson_matrix

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

    def get_uppertriangular(self, rdm):
        """Get upper triangle of a RDM

        Args:
            rdm (numpy array): nxn rdm

        Returns:
            numpy array: only the upper triangle of rdm
        """

        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]

    def pearsons_pytorch(self, sl_rdms_ut, dnn_rdm_ut):
        """Caclulates correlaion between searchlight and dnn rdms

        Args:
            sl_rdms_ut (numpy array): upper triangle of searchlight rdm
            dnn_rdm_ut (numpty array): Upper trigangle of dnn rdm

        Returns:
            int: correlation value
        """

        # Reshape RDMs
        dim = -1
        dnn_rdm_ut = dnn_rdm_ut.view(sl_rdms_ut.shape[1], -1)
        x = sl_rdms_ut
        y = dnn_rdm_ut.T

        # Centering
        centered_x = x - x.mean(dim=dim, keepdim=True)
        centered_y = y - y.mean(dim=dim, keepdim=True)

        # Calculate Covariance
        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
        bessel_corrected_covariance = covariance / (x.shape[dim])

        x_std = x.std(dim=dim, keepdim=True)
        y_std = y.std(dim=dim, keepdim=True)

        # Correlation
        corr = bessel_corrected_covariance / (x_std * y_std)

        return corr

    def return_ranks_sl(self, array):
        """Returns the ranks of the searchlight array

        Args:
            array (numpy array): serachlight arrray

        Returns:
            numpy array: array of all ranks
        """

        if len(array.shape) == 1:
            array = np.expand_dims(array, axis=0)

        num_sl_cubes = array.shape[0]
        rank_array = np.empty_like(array)

        for i in range(num_sl_cubes):
            temp = array[i, :].argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(array.shape[1])
            rank_array[i, :] = ranks

        return rank_array

    def return_ranks(self, array):
        """Returns the ranks of the dnn array

        Args:
            array (numpy array): dnn array

        Returns:
            numpy array: array of all ranks
        """

        temp = array.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(array))

        return ranks

    def pearson_matrix(self, ind_vars, dep_var):
        """[summary]

        Args:
            ind_vars (list of tensor): ?
            dep_var (torch tensor): ?

        Returns:
            list: all correlations of model and searchlight ROIs
        """

        correlations = []

        for i, ind_var_rank in (enumerate(ind_vars)):
            fast_sl_result = self.pearsons_pytorch(dep_var, ind_var_rank)
            correlations.append(fast_sl_result.cpu().detach().numpy().ravel())

        correlations = np.array(correlations)

        return correlations

    def evaluate_searchlight(self, noise_ceiling, this_model_rdm):
        """Evaluation function for searchlight

        Args:
            noise_ceiling (dict): {"lnc": lnc, "unc": unc}

        Returns:
            [type]: [description]
        """

        model_ranks_cuda = []  # model rdms list to feed in gpu

        for i, model_rdm in enumerate(this_model_rdm):
            model_rdm_ranks = self.return_ranks(self.get_uppertriangular(
                model_rdm))  # return rank of model rdms
            model_rdm_ranks = torch.from_numpy(
                model_rdm_ranks).float().cuda()  # convert to cuda
            model_ranks_cuda.append(model_rdm_ranks)


        # Declare variables
        model_rdms = model_ranks_cuda
        fmri_rdm = self.searchlight_rdm
        num_subjects = fmri_rdm.shape[0]
        num_searchlights = fmri_rdm.shape[1]


        # Calculation
        fmri_rdm_ut = []
        for subject in range(num_subjects):
            fmri_rdm_ut_sub = []
            for searchlight in range(num_searchlights):
                fmri_rdm_ut_sub.append(self.get_uppertriangular(
                    fmri_rdm[subject, searchlight, :, :]))
            fmri_rdm_ut.append(fmri_rdm_ut_sub)

        fmri_rdm_ut = np.array(fmri_rdm_ut)
        fmri_rdm = np.zeros_like(fmri_rdm_ut)

        # calculate rank of searchlight rdms
        for subject in range(num_subjects):
            fmri_rdm_ranks = self.return_ranks_sl(fmri_rdm_ut[subject, :])
            fmri_rdm[subject, :] = fmri_rdm_ranks  # rank RDMs of subjects

        fmri_rdm = torch.from_numpy(fmri_rdm).float().cuda()  # to GPU

        results = []

        for subject in range(num_subjects):
            results.append(self.distance(model_rdms, fmri_rdm[subject, :]))

        """We have one r2 value per region of interest"""

        r2 = np.array(results).mean(axis=0)[0]  # returns array of R2

        sem = stats.sem(results, axis=0)[0]

        sig_value = stats.ttest_1samp(results, 0, axis=0)[1][0]

        amount_rois = len(r2)

        # TODO: Netkey is not used anymore. As soon as this is tested it can be removed along with the additon of
        # Pandas code down below
        net_key = "(-) " + "-"
        layer_key = "(" + str(self.layer_counter) + ") " + self.current_layer


        # Create final dictionary
        for i in range(amount_rois):
            roi_name = "(" + str(i) + ") ROI" + str(i)

            this_r2 = r2[i]
            this_sem = sem[i]
            this_sig = sig_value[i]
            lnc = noise_ceiling["lnc"]
            unc = noise_ceiling["unc"]
            # area_percentNC = (this_r2 / lnc) * 100.

            # Turn into output dictionary
            output_dict = {layer_key: [this_r2, 0, this_sig, this_sem, [lnc, unc]]}


            if roi_name in self.final_dict['searchlight']:
                if net_key in self.final_dict['searchlight'][roi_name]:
                    self.final_dict['searchlight'][roi_name][net_key].update(
                        output_dict)
                else:
                    self.final_dict['searchlight'][roi_name].update(
                        {net_key: output_dict})

            else:
                self.final_dict['searchlight'].update(
                    {roi_name: {net_key: output_dict}})

    def evaluate(self):
        """Function to prepare for searchlight evaluation

        Returns:
            dict: final dictionary with each correlation value
        """

        self.final_dict = {'searchlight': {}}

        # TODO: Add actual NoiseCeiling!
        noise_ceiling = {'lnc': 0, 'unc': 0}

        for counter, layer in self.model_rdms:
            self.current_layer = layer
            self.layer_counter = counter

            this_model_rdm = [np.load(op.join(self.model_rdms_path, layer))['arr_0']]
            self.evaluate_searchlight(noise_ceiling, this_model_rdm)

        return self.final_dict
