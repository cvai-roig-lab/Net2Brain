import numpy as np
from scipy.stats import spearmanr
import json
import os.path as op
from .eval_helper import *
from .distance_functions import registered_nc_distance_functions


class NoiseCeiling():
    def __init__(self, roi, roi_path, distance_metric):
        """Initialize NoiseCeiling calculation

        Args:
            roi (str): name of roi
            roi_path (str/path): path to this roi
            distance_func (function): distance function to be used (from the other module)
        """
        self.roi = roi
        self.roi_path = roi_path
        
        # Get distance function
        self.distance_metric = distance_metric
        self.distance_func = registered_nc_distance_functions[distance_metric]
        
        
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
        return self.distance_func(lt_rdm1, lt_rdm2)



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
        lnc = self.get_lowernoiseceiling(target[key_list[0]])  # **2
        unc = self.get_uppernoiseceiling(target[key_list[0]])  # **2

        noise_ceilings = {"lnc": lnc, "unc": unc}
        return noise_ceilings
        
        
    def noise_ceiling_meg(self):
        """Gets noise ceiling for meg scans

        Returns:
            dict: {"lnc": lnc, "unc": unc}
        """
        target = np.load(self.roi_path)
        
        avg_stamps = []
        keys = []

        for person, dict_infos in target.items():
            keys.append(person)
            for timestamp in dict_infos:
                avg = np.mean(timestamp, 0)  # average over timestamp axis
                avg_stamps.append(avg)

        avg_stamps = np.array(avg_stamps)
        new_dict = {keys[0]: avg_stamps}
        target = new_dict

        key_list = []
        for keys, values in target.items():
            key_list.append(keys)

        # get noise ceilings
        lnc = self.get_lowernoiseceiling(target[key_list[0]])  # **2
        unc = self.get_uppernoiseceiling(target[key_list[0]])  # **2

        noise_ceilings = {"lnc": lnc, "unc": unc}
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
        Checks if noise ceiling has already been calculated, if not it
        calculates it.

        Returns:
            dict: {"lnc": lnc, "unc": unc}
        """
        
        # First find out if there is a json file that already contains these values
        json_path = op.dirname(self.roi_path)
        json_file = op.join(json_path, "noise_ceiling_log.json")
        
        nc_exists = False 
        data = {}
        
        # If we have a json file, grab its data
        if op.exists(json_file):
            with open(json_file) as json_file_wrapper:
                data = json.load(json_file_wrapper)
                if self.roi in data:
                    if self.distance_metric in data[self.roi]:
                        noise_ceiling = data[self.roi][self.distance_metric]
                        nc_exists = True  # Set flag to true if noise ceiling already exists for the metric
                        return noise_ceiling
        
        # If we don't have a json file or if the distance metric doesn't exist, calculate NC and save into json file
        if not nc_exists:
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

            # Add the distance metric to the noise ceiling data
            if self.roi not in data:
                data[self.roi] = {}
            
            data[self.roi][self.distance_metric] = noise_ceiling

            # Write NoiseCeiling to JSON
            with open(json_file, 'w') as fp:
                json.dump(data, fp, sort_keys=True, indent=4)

        return noise_ceiling
