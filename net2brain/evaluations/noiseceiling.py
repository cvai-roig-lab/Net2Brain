import numpy as np
from scipy.stats import spearmanr
import json
import os.path as op
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


class NoiseCeiling():
    def __init__(self, roi, roi_path):
        """Initialize NoiseCeiling calculation

        Args:
            roi (str): name of roi
            roi_path (str/path): path to this roi
        """
        self.roi = roi
        self.roi_path = roi_path
        
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
        return (spearmanr(lt_rdm1, lt_rdm2)[0]) ** 2


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
        for i in range(num_subs):
            sub_rdm = rdm[i, :, :]
            mean_sub_rdm = np.mean(rdm, axis=0)  # take mean
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
        
        
    def noise_ceiling(self):
        """Calculates the noise ceiling for either meg or fmri data
        Checks if noise ceiling has already been calculated, if not it
        calculates it

        Returns:
            dict: {"lnc": lnc, "unc": unc}
        """
        
        # First find out if there is a json file that already contains these values
        json_path = op.dirname(self.roi_path)
        json_file = op.join(json_path, "log.json")
        
        
        nc_exists = False 
        
        # If we have json file, grab its data
        if op.exists(json_file):
            with open(json_file) as json_file_wrapper:
                data = json.load(json_file_wrapper)
                if self.roi in data:
                    noise_ceiling = data[self.roi]
                    nc_exists == True
                    return noise_ceiling
                else:
                    pass
        else:
            pass
        
        # If we dont have json file, calculate NC and save into json file
        if nc_exists == False:
        
            if "meg" in self.roi.lower():
                # Returns {"lnc": lnc, "unc" : unc}
                noise_ceiling = self.noise_ceiling_meg()
            elif "fmri" in self.roi.lower():
                noise_ceiling =self.noise_ceiling_fmri()  # Returns {"lnc": lnc, "unc" : unc}
            else:
                error_message(
                    "Could not calculate NC. No 'fmri' or 'meg' in scan name")
                noise_ceiling = {"lnc": 0, "unc": 0}
                

            # Write NoiseCeiling to json
            args = [{self.roi: noise_ceiling}]
            
            
            # If we already have a json, add the data, else create new
            if op.exists(json_file):
                with open(json_file) as json_file_wrapper:
                    data = json.load(json_file_wrapper)
                data[self.roi] = noise_ceiling
                    
                with open(json_file, 'w') as fp:
                    json.dump(data, fp, sort_keys=True, indent=4)

            else:
                args = {self.roi: noise_ceiling}
                # write individual json
                with open(json_file, 'w') as fp:
                    json.dump(args, fp, sort_keys=True, indent=4)
            
        return noise_ceiling

