import numpy as np
from scipy.stats import spearmanr
from .eval_helper import *
from .distance_functions import registered_nc_distance_functions


class NoiseCeiling():
    def __init__(self, roi, roi_path, distance_metric, squared, timepoint_agg=True):
        """Initialize NoiseCeiling calculation

        Args:
            roi (str): name of roi
            roi_path (str/path): path to this roi
            distance_func (function): distance function to be used (from the other module)
            timepoint_agg (bool): MEG/EEG: average over timepoints like RSA, else one value per timepoint
        """
        self.roi = roi
        self.roi_path = roi_path
        
        # Get distance function
        self.distance_metric = distance_metric
        self.distance_func = registered_nc_distance_functions[distance_metric]
        self.squared = squared
        self.timepoint_agg = timepoint_agg
        
        
    def get_uppertriangular(self, rdm):
        """Get upper triangle of a RDM

        Args:
            rdm (numpy array): nxn rdm

        Returns:
            numpy array: only the upper triangle of rdm
        """
        
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]

    

    def noise_ceiling_spearman(self, rdm1, rdm2):
        """Calculate Spearman for noiseceiling

        Args:
            rdm1 (numpy array): rdm of subject
            rdm2 (numpy array): mean rdm of all subjects

        Returns:
            float: squared correlation between both rdms
        """
        lt_rdm1 = self.get_uppertriangular(rdm1)
        lt_rdm2 = self.get_uppertriangular(rdm2)
        return self.distance_func(lt_rdm1, lt_rdm2) ** 2 if self.squared else self.distance_func(lt_rdm1, lt_rdm2)



    def get_uppernoiseceiling(self, rdm):
        """Calculate upper noise ceiling
            1. Take the mean of all RDMs without removing subjects
            2. Spearman of subject and average RDMs
            3. Average all of that
            => How good are the RDMs generalized

        Args:
            rdm (list of rdms): all subject rdms

        Returns:
            float: upper noise ceiling
        """

        num_subs = rdm.shape[0]
        unc = 0.0
        mean_sub_rdm = np.mean(rdm, axis=0)  # take mean
        for i in range(num_subs):
            sub_rdm = rdm[i, :, :]
            # calculate spearman
            unc += self.noise_ceiling_spearman(sub_rdm, mean_sub_rdm)

        unc = unc / num_subs
        return unc


    def get_lowernoiseceiling(self, rdm):
        """Take the lower noise ceiling
        1. Extracting one subject from the overall rdm
        2. Take the mean of all the other RDMs
        3. Take spearman correlation of subject RDM and mean subject RDM
        4. We do this for all the subjects and then calculate the average
        => Can we predict person 15 from the rest of the subjects?
        => Low Noise-Ceiling means we need better data

        Args:
            rdm (list of rdms): all subject rdms

        Returns:
            float: lower noise ceiling
        """

        num_subs = rdm.shape[0]
        lnc = 0.0

        for i in range(num_subs):
            sub_rdm = rdm[i, :, :]
            rdm_sub_removed = np.delete(rdm, i, axis=0)  # remove one person
            # take mean of other RDMs
            mean_sub_rdm = np.mean(rdm_sub_removed, axis=0)
            lnc += self.noise_ceiling_spearman(sub_rdm, mean_sub_rdm)  # take spearman

        lnc = lnc / num_subs  # average it
        return lnc
        
        
    def noise_ceiling_fmri(self):
        """Gets noise ceiling for fmri scans

        Returns:
            dict: {"lnc": lnc, "unc": unc}
        """
        target = np.load(self.roi_path)
        key_list = []
        for keys, values in target.items():
            key_list.append(keys)

        # lower nc and upper nc
        lnc = self.get_lowernoiseceiling(target[key_list[0]])
        unc = self.get_uppernoiseceiling(target[key_list[0]])

        noise_ceilings = {"lnc": lnc, "unc": unc}
        return noise_ceilings
        
        
    def noise_ceiling_meg(self):
        """Gets noise ceiling for meg scans [subjects, time, stimuli, stimuli]

        The ceiling is computed per timepoint and aggregated like RSA's R: the per-subject
        correlations are averaged over time (if timepoint_agg), squared (if squared) and
        averaged over subjects.

        Returns:
            dict: {"lnc": lnc, "unc": unc}, per-timepoint lists if not timepoint_agg
        """
        target = np.load(self.roi_path)
        rdm = target[list(target.keys())[0]]
        num_subs, num_times = rdm.shape[:2]

        lower = np.empty((num_subs, num_times))
        upper = np.empty((num_subs, num_times))
        for t in range(num_times):
            mean_rdm = self.get_uppertriangular(np.mean(rdm[:, t], axis=0))
            for i in range(num_subs):
                sub_rdm = self.get_uppertriangular(rdm[i, t])
                mean_others = self.get_uppertriangular(np.mean(np.delete(rdm[:, t], i, axis=0), axis=0))
                lower[i, t] = self.distance_func(sub_rdm, mean_others)
                upper[i, t] = self.distance_func(sub_rdm, mean_rdm)

        if self.timepoint_agg:
            lower, upper = lower.mean(axis=1), upper.mean(axis=1)
        if self.squared:
            lower, upper = np.square(lower), np.square(upper)

        noise_ceilings = {"lnc": lower.mean(axis=0).tolist(), "unc": upper.mean(axis=0).tolist()}
        return noise_ceilings
    

    def find_datatype(self, file_path):
        """Function to determine if the input data corresponds to fMRI or MEG based on data shape.

        Args:
            file_path (str): Path to the numpy file (.npz) containing ROI data.

        Returns:
            str: Returns 'fmri' or 'meg' based on the data shape.
        """
        # Load the .npz file
        data = np.load(file_path, allow_pickle=True)

        # Get the first key from the loaded file
        keys = list(data.keys())
        if not keys:
            raise ValueError("The provided .npz file is empty.")

        rdm_data = data[keys[0]]  # Get the first RDM from the file
        shape = rdm_data.shape  # Get the shape of the RDM

        if len(shape) == 2 and shape[0] == shape[1]:
            # fMRI data with shape (images, images)
            return 'fmri'

        elif len(shape) == 3:
            if shape[1] == shape[2]:
                # fMRI data with shape (subjects, images, images)
                return 'fmri'
            else:
                raise ValueError(f"Invalid fMRI data shape: {shape}. Last two dimensions must be equal.")

        elif len(shape) == 4:
            if shape[2] == shape[3]:
                # MEG data with shape (subjects, times, images, images)
                return 'meg'
            else:
                raise ValueError(f"Invalid MEG data shape: {shape}. Last two dimensions must be equal.")

        else:
            raise ValueError(f"Unexpected data shape: {shape}. Expected 2D, 3D, or 4D array.")

        
        
    def noise_ceiling(self):
        """Calculates the noise ceiling for either meg or fmri data

        Returns:
            dict: {"lnc": lnc, "unc": unc}
        """
        # Use find_datatype to determine if it's fMRI or MEG data
        datatype = self.find_datatype(self.roi_path)

        # Based on the datatype, calculate the noise ceiling
        if datatype == 'fmri':
            noise_ceiling = self.noise_ceiling_fmri()  # fMRI case
        elif datatype == 'meg':
            noise_ceiling = self.noise_ceiling_meg()  # MEG case
        else:
            error_message("Could not calculate NC. Unexpected data type.")
            noise_ceiling = {"lnc": 0, "unc": 0}

        return noise_ceiling
