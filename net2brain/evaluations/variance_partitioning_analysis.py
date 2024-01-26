import os
import os.path as op

import numpy as np
import pandas as pd
import scipy

from .eval_helper import *
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_1samp
from scipy.spatial.distance import squareform
import statsmodels.stats.multitest

class VPA():
    """Class for Variance Partitioning Analysis
    """

    def __init__(self, dependent_variable, independent_variables, variable_names):
        """Initiating VPA

        Args:
            dependent_variable (list): Variable that the VPA depends on
            independent_variables (list): List of Lists of independent variables. Must have same # of features as dependent var
            variable_names (list): Names for the independent variables for plotting 
        """
        # Check if the lengths of independent_variables and variable_names match
        if len(independent_variables) != len(variable_names):
            raise ValueError("The length of independent_variables and variable_names should match.")
        
        # Ensure there are at least 2 and at most 4 independent variables
        if not (2 <= len(independent_variables) <= 4):
            raise ValueError("You should provide between 2 to 4 independent variables.")

        # Declare dependent variable
        self.dependent_variable = dependent_variable

        # Assign independent variables
        independents = [None, None, None, None]  # Initiate a list of 4 None values
        for i, var in enumerate(independent_variables):
            independents[i] = var
        
        self.independent_1, self.independent_2, self.independent_3, self.independent_4 = independents

        # Other attributes
        self.variable_names = variable_names

        # Declare function 
        function_map = {
            2: self.evaluate_2,
            3: self.evaluate_3,
            4: self.evaluate_4
        }
        self.VPA_function = function_map[len(independent_variables)]





    def dim_fitter(self, ind_var):
        """Function to fit dimensions for VPA depending on its dimension

        Args:
            ind_var (array): Independent variable

        Returns:
            ind_var (array): Reshaped
        """
        if ind_var.ndim==1:
            return ind_var.reshape(-1, 1)
        else:
            return ind_var.T
        

    def noise_ceiling_r2(self, rdm1, rdm2):
        """Calculate Spearman for noiseceiling

        Args:
            rdm1 (numpy array): rdm of subject
            rdm2 (numpy array): mean rdm of all subjects

        Returns:
            float: squared correlation between both rdms
        """
        lt_rdm1 = rdm1
        lt_rdm2 = rdm2
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lt_rdm1, lt_rdm2)
        return r_value ** 2

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
            sub_rdm = rdm[i]
            # calculate spearman
            unc += self.noise_ceiling_r2(sub_rdm, mean_sub_rdm)

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
            sub_rdm = rdm[i]
            rdm_sub_removed = np.delete(rdm, i, axis=0)  # remove one person
            # take mean of other RDMs
            mean_sub_rdm = np.mean(rdm_sub_removed, axis=0)
            lnc += self.noise_ceiling_r2(sub_rdm, mean_sub_rdm)  # take spearman

        lnc = lnc / num_subs  # average it
        return lnc
        

    def get_uppertriangular(self, rdm):
        """Get the upper triangular of the square RDM

        Args:
            rdm (array): RDM in question

        Returns:
            rdm (array): Returns the upper triangular
        """
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions,1)]
    

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


    def load_rdms(self, rdm_paths):
        """Function to load the RDMs for VPA depending on the type of RDM

        Args:
            rdm_paths (list of string): Path to the actual RDMs

        Returns:
            rdm_list: List of opened RDMs
        """

        # EEG Data Flag
        is_eeg = False

        # Turn it into list if not list
        if isinstance(rdm_paths, str):
            is_eeg = True
            rdm_paths = [rdm_paths]

        # Create empty list for RDMS
        rdm_list = []

        if is_eeg:
            # Iterate through paths and open the RDMs
            for model in rdm_paths:
                this_rdm = load(model)
                rdm_list.append(this_rdm)
            
            # Turn into numpy array
            rdm_list = np.array(rdm_list)

            return rdm_list
        else:
            # Iterate through paths and open the RDMs
            for model in rdm_paths:
                this_rdm = load(model)

                if 'rdm' in this_rdm:
                    this_rdm = this_rdm['rdm']
                elif 'arr_0' in this_rdm:
                    this_rdm = this_rdm['arr_0']
                else:
                    raise ValueError(f"The RDM file does not contain 'rdm' or 'arr_0' keys.")
                
                this_rdm = self.check_squareform(this_rdm) # Check if rdm is squareform #TODO Remove soon after reimplementing RSA
        
                rdm_list.append(self.get_uppertriangular(this_rdm))

            # Turn into numpy array
            rdm_list = np.array(rdm_list)

            return rdm_list
    
    



    def calculate_p_values(self, models_data):
        """
        Calculate p-values for each time-step in models_data using one-sample t-test among participants.

        Args:
            models_data (numpy array): Data array with shape (num_subs, num_time_steps).

        Returns:
            numpy array: Array of p-values for each time-step.
        """
        num_subs, num_time_steps = models_data.shape
        p_values = np.zeros(num_time_steps)

        # For each time step calculate p-value across all participants
        for time_step in range(num_time_steps):
            t_stat, p_val = ttest_1samp(models_data[:, time_step], 0)
            p_values[time_step] = p_val

        # Correct P-Values
        bools, corrected_p_values = statsmodels.stats.multitest.fdrcorrection(p_values)

        return corrected_p_values



    def VPA_2(self, dep_var):
        """VPA for 2 independent variables

        Args:
            dep_var (array): Dependent variable

        Returns:
            y12: Total variance
            y1: Unique Variance 1
            y2: Unique variance 2
        """

        # Open RDMs
        ind_var_1 = self.load_rdms(self.independent_1)
        ind_var_2 = self.load_rdms(self.independent_2)

        # Average if wanted
        if self.average_models:
            ind_var_1 = np.mean(ind_var_1, axis=0)
            ind_var_2 = np.mean(ind_var_2, axis=0)


        # Initialize the linear regression model
        lm = linear_model.LinearRegression()

        # Calculate R-squared values for each independent variable
        R1 = lm.fit(self.dim_fitter(ind_var_1), dep_var).score(self.dim_fitter(ind_var_1), dep_var)
        R2 = lm.fit(self.dim_fitter(ind_var_2), dep_var).score(self.dim_fitter(ind_var_2), dep_var)

        # Calculate R-squared values for combinations of independent variables
        R12 = lm.fit(np.vstack((ind_var_1, ind_var_2)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2)).T, dep_var)

        # Calculate variance partitioning components
        y12 = R1 + R2 - R12
        y1 = R1 - y12
        y2 = R2 - y12

        return {'R1': R1, 'R2': R2, 'R12': R12, 'y12': y12, 'y1': y1, 'y2': y2}



    def VPA_3(self, dep_var):
        """VPA for 3 independent variables

        Args:
            dep_var (array): Dependent variable

        Returns:
            y123: Total variance
            y1: Unique Variance 1
            y2: Unique variance 2
            y3: Unique variance 3
        """

        # Open RDMs
        ind_var_1 = self.load_rdms(self.independent_1)
        ind_var_2 = self.load_rdms(self.independent_2)
        ind_var_3 = self.load_rdms(self.independent_3)

        # Average if wanted
        if self.average_models:
            ind_var_1 = np.mean(ind_var_1, axis=0)
            ind_var_2 = np.mean(ind_var_2, axis=0)
            ind_var_3 = np.mean(ind_var_3, axis=0)


        # Initialize the linear regression model
        lm = linear_model.LinearRegression()

        # Calculate R-squared values for each independent variable
        R1 = lm.fit(self.dim_fitter(ind_var_1), dep_var).score(self.dim_fitter(ind_var_1), dep_var)
        R2 = lm.fit(self.dim_fitter(ind_var_2), dep_var).score(self.dim_fitter(ind_var_2), dep_var)
        R3 = lm.fit(self.dim_fitter(ind_var_3), dep_var).score(self.dim_fitter(ind_var_3), dep_var)

        # Calculate R-squared values for combinations of independent variables
        R12 = lm.fit(np.vstack((ind_var_1, ind_var_2)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2)).T, dep_var)
        R13 = lm.fit(np.vstack((ind_var_1, ind_var_3)).T, dep_var).score(np.vstack((ind_var_1, ind_var_3)).T, dep_var)
        R23 = lm.fit(np.vstack((ind_var_2, ind_var_3)).T, dep_var).score(np.vstack((ind_var_2, ind_var_3)).T, dep_var)

        R123 = lm.fit(np.vstack((ind_var_1, ind_var_2, ind_var_3)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2, ind_var_3)).T, dep_var)

        # Calculate variance partitioning components
        y123 = R1 + R2 + R3 - R12 - R13 - R23 + R123
        y12 = R1 + R2 - R12 - y123
        y13 = R1 + R3 - R13 - y123
        y23 = R2 + R3 - R23 - y123
        y1 = R1 - y12 - y13 - y123
        y2 = R2 - y12 - y23 - y123
        y3 = R3 - y13 - y23 - y123

        return {'R1': R1, 'R2': R2, 'R3': R3, 'R12': R12, 'R13': R13, 'R23': R23, 'R123': R123, 'y123': y123, 'y12': y12, 'y13': y13, 'y23': y23, 'y1': y1, 'y2': y2, 'y3': y3}
    


    def VPA_4(self, dep_var):
        """VPA for 4 independent variables

        Args:
            dep_var (array): Dependent variable

        Returns:
            y123: Total variance
            y1: Unique Variance 1
            y2: Unique variance 2
            y3: Unique variance 3
            y4: Unique variance 4
        """
        
        # Open RDMs
        ind_var_1 = self.load_rdms(self.independent_1)
        ind_var_2 = self.load_rdms(self.independent_2)
        ind_var_3 = self.load_rdms(self.independent_3)
        ind_var_4 = self.load_rdms(self.independent_4)

        # Average if wanted
        if self.average_models:
            ind_var_1 = np.mean(ind_var_1, axis=0)
            ind_var_2 = np.mean(ind_var_2, axis=0)
            ind_var_3 = np.mean(ind_var_3, axis=0)
            ind_var_4 = np.mean(ind_var_4, axis=0)

        # Initialize the linear regression model
        lm = linear_model.LinearRegression()

        # Calculate R-squared values for each independent variable
        R1 = lm.fit(self.dim_fitter(ind_var_1), dep_var).score(self.dim_fitter(ind_var_1), dep_var)
        R2 = lm.fit(self.dim_fitter(ind_var_2), dep_var).score(self.dim_fitter(ind_var_2), dep_var)
        R3 = lm.fit(self.dim_fitter(ind_var_3), dep_var).score(self.dim_fitter(ind_var_3), dep_var)
        R4 = lm.fit(self.dim_fitter(ind_var_4), dep_var).score(self.dim_fitter(ind_var_4), dep_var)

        # Calculate R-squared values for combinations of independent variables
        R12 = lm.fit(np.vstack((ind_var_1, ind_var_2)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2)).T, dep_var)
        R13 = lm.fit(np.vstack((ind_var_1, ind_var_3)).T, dep_var).score(np.vstack((ind_var_1, ind_var_3)).T, dep_var)
        R14 = lm.fit(np.vstack((ind_var_1, ind_var_4)).T, dep_var).score(np.vstack((ind_var_1, ind_var_4)).T, dep_var)
        R23 = lm.fit(np.vstack((ind_var_2, ind_var_3)).T, dep_var).score(np.vstack((ind_var_2, ind_var_3)).T, dep_var)
        R24 = lm.fit(np.vstack((ind_var_2, ind_var_4)).T, dep_var).score(np.vstack((ind_var_2, ind_var_4)).T, dep_var)
        R34 = lm.fit(np.vstack((ind_var_3, ind_var_4)).T, dep_var).score(np.vstack((ind_var_3, ind_var_4)).T, dep_var)

        # Calculate R-squared values for combinations of independent variables
        R123 = lm.fit(np.vstack((ind_var_1, ind_var_2,ind_var_3)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2,ind_var_3)).T, dep_var)
        R134 = lm.fit(np.vstack((ind_var_1, ind_var_3,ind_var_4)).T, dep_var).score(np.vstack((ind_var_1, ind_var_3,ind_var_4)).T, dep_var)
        R124 = lm.fit(np.vstack((ind_var_1, ind_var_2,ind_var_4)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2,ind_var_4)).T, dep_var)
        R234 = lm.fit(np.vstack((ind_var_2, ind_var_3,ind_var_4)).T, dep_var).score(np.vstack((ind_var_2, ind_var_3,ind_var_4)).T, dep_var)
        
        R1234 = lm.fit(np.vstack((ind_var_1, ind_var_2, ind_var_3, ind_var_4)).T, dep_var).score(np.vstack((ind_var_1, ind_var_2, ind_var_3, ind_var_4)).T, dep_var)

        # Calculate variance partitioning components
        y1234 = R1 + R2 + R3 + R4 - R12 - R13 - R14 - R23 - R24 - R34 + R123 + R134+ R124 + R234 - R1234
        y123 = R1 + R2 + R3 - R12 - R13 - R23 + R123 - y1234
        y124 = R1 + R2 + R4 - R12 - R14 - R24 + R124 - y1234
        y134 = R1 + R3 + R4 - R13 - R14 - R34 + R134 - y1234
        y234 = R2 + R3 + R4 - R23 - R24 - R34 + R234 - y1234
        y12 = R1 + R2 - R12 - y123 - y124 - y1234
        y13 = R1 + R3 - R13 - y123 - y134 - y1234
        y14 = R1 + R4 - R14 - y124 - y134 - y1234
        y23 = R2 + R3 - R23 - y123 - y234 - y1234
        y24 = R2 + R4 - R24 - y124 - y234 - y1234
        y34 = R3 + R4 - R34 - y134 - y234 - y1234
        y1 = R1 - y12 - y13 - y14 - y123 - y124 - y134 - y1234
        y2 = R2 - y12 - y23 - y24 - y123 - y234 - y124 - y1234
        y3 = R3 - y13 - y23 - y34 - y123 - y234 - y134 - y1234
        y4 = R4 - y14 - y24 - y34 - y124 - y134 - y234 - y1234

        return {
        'R1': R1, 'R2': R2, 'R3': R3, 'R4': R4,
        'R12': R12, 'R13': R13, 'R14': R14, 'R23': R23, 'R24': R24, 'R34': R34,
        'R123': R123, 'R124': R124, 'R134': R134, 'R234': R234, 'R1234': R1234,
        'y1234': y1234, 'y123': y123, 'y124': y124, 'y134': y134, 'y234': y234,
        'y12': y12, 'y13': y13, 'y14': y14, 'y23': y23, 'y24': y24, 'y34': y34,
        'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4}

    


    def evaluate_2(self, dep_var, num_subjects, eeg_time):
        """Evaluation function for VPA with 2 variables

        Args:
            dep_var (array): Dependent variable
            num_subjects (int): Number of Subjects in independent variable
            eeg_time (int): Number of ms captured in EEG data

        Returns:
            all_variances_df: Pandas dataframe with results
        """

        all_variances_df = pd.DataFrame(columns=['Variable', 'Description', 'Values', 'Significance', 'Color'])

        # Initialize arrays for storing values
        R1_values = np.zeros((num_subjects, eeg_time))
        R2_values = np.zeros((num_subjects, eeg_time))
        R12_values = np.zeros((num_subjects, eeg_time))
        y12_values = np.zeros((num_subjects, eeg_time))
        y1_values = np.zeros((num_subjects, eeg_time))
        y2_values = np.zeros((num_subjects, eeg_time))

        R_un = np.zeros(eeg_time)
        R_ln = np.zeros(eeg_time)

        for time in range(eeg_time):
            R_un[time] = self.get_uppernoiseceiling(np.squeeze(dep_var[:,time]))
            R_ln[time] = self.get_lowernoiseceiling(np.squeeze(dep_var[:,time]))

            for i in range(num_subjects):
                this_dep_var = np.squeeze(dep_var[i,time,:])
                results = self.VPA_2(this_dep_var)
                R1_values[i, time] = results['R1']
                R2_values[i, time] = results['R2']
                R12_values[i, time] = results['R12']
                y12_values[i, time] = results['y12']
                y1_values[i, time] = results['y1']
                y2_values[i, time] = results['y2']

        # Calculate significance for all values
        sig_R1 = self.calculate_p_values(R1_values)
        sig_R2 = self.calculate_p_values(R2_values)
        sig_R12 = self.calculate_p_values(R12_values)
        sig_y12 = self.calculate_p_values(y12_values)  # Calculate significance for y12
        sig_y1 = self.calculate_p_values(y1_values)
        sig_y2 = self.calculate_p_values(y2_values)

        # Define naming convention for the dataframe
        variables = ['R1', 'R2', 'R12', 'y12', 'y1', 'y2']  # Include y12 in the list
        descriptions = [self.variable_names[0] + ' Influence', self.variable_names[1] + ' Influence', 
                        'Combined Influence', 'Shared Variance', 'Unique ' + self.variable_names[0], 'Unique ' + self.variable_names[1]]
        values = [R1_values, R2_values, R12_values, y12_values, y1_values, y2_values]
        significances = [sig_R1, sig_R2, sig_R12, sig_y12, sig_y1, sig_y2]

        data_to_append = []

        for var, desc, val, sig in zip(variables, descriptions, values, significances):
            data_to_append.append({
                'Variable': var,
                'Description': desc,
                'Values': val,
                'Significance': sig,
                'Color': None  # Assign colors if desired for visualization
            })

        all_variances_df = pd.concat([all_variances_df, pd.DataFrame(data_to_append)], ignore_index=True)

        return all_variances_df




    def evaluate_3(self, dep_var, num_subjects, eeg_time):
        """Evaluation function for VPA with 3 variables

        Args:
            dep_var (array): Dependent variable
            num_subjects (int): Number of Subjects in independent variable
            eeg_time (int): Number of ms captured in EEG data

        Returns:
            all_variances_df: Pandas dataframe with results
        """

        all_variances_df = pd.DataFrame(columns=['Variable', 'Description', 'Values', 'Significance', 'Color'])

        # Initialize arrays for storing values
        R1_values = np.zeros((num_subjects, eeg_time))
        R2_values = np.zeros((num_subjects, eeg_time))
        R3_values = np.zeros((num_subjects, eeg_time))
        R12_values = np.zeros((num_subjects, eeg_time))
        R13_values = np.zeros((num_subjects, eeg_time))
        R23_values = np.zeros((num_subjects, eeg_time))
        R123_values = np.zeros((num_subjects, eeg_time))
        y123_values = np.zeros((num_subjects, eeg_time))
        y12_values = np.zeros((num_subjects, eeg_time))
        y13_values = np.zeros((num_subjects, eeg_time))
        y23_values = np.zeros((num_subjects, eeg_time))
        y1_values = np.zeros((num_subjects, eeg_time))
        y2_values = np.zeros((num_subjects, eeg_time))
        y3_values = np.zeros((num_subjects, eeg_time))

        R_un = np.zeros(eeg_time)
        R_ln = np.zeros(eeg_time)

        for time in range(eeg_time):
            R_un[time] = self.get_uppernoiseceiling(np.squeeze(dep_var[:,time]))
            R_ln[time] = self.get_lowernoiseceiling(np.squeeze(dep_var[:,time]))

            for i in range(num_subjects):
                this_dep_var = np.squeeze(dep_var[i,time,:])
                results = self.VPA_3(this_dep_var)
                R1_values[i, time], R2_values[i, time], R3_values[i, time], R12_values[i, time], R13_values[i, time], R23_values[i, time], R123_values[i, time], y123_values[i, time], y12_values[i, time], y13_values[i, time], y23_values[i, time], y1_values[i, time], y2_values[i, time], y3_values[i, time] = results

        # Calculate significance for all values
        sig_R1 = self.calculate_p_values(R1_values)
        sig_R2 = self.calculate_p_values(R2_values)
        sig_R3 = self.calculate_p_values(R3_values)
        sig_R12 = self.calculate_p_values(R12_values)
        sig_R13 = self.calculate_p_values(R13_values)
        sig_R23 = self.calculate_p_values(R23_values)
        sig_R123 = self.calculate_p_values(R123_values)
        sig_y123 = self.calculate_p_values(y123_values)
        sig_y12 = self.calculate_p_values(y12_values)
        sig_y13 = self.calculate_p_values(y13_values)
        sig_y23 = self.calculate_p_values(y23_values)
        sig_y1 = self.calculate_p_values(y1_values)
        sig_y2 = self.calculate_p_values(y2_values)
        sig_y3 = self.calculate_p_values(y3_values)

        # Define naming convention for the dataframe
        variables = ['R1', 'R2', 'R3', 'R12', 'R13', 'R23', 'R123', 'y123', 'y12', 'y13', 'y23', 'y1', 'y2', 'y3']
        descriptions = [
            self.variable_names[0] + ' Influence', self.variable_names[1] + ' Influence', self.variable_names[2] + ' Influence',
            'Combined ' + self.variable_names[0] + ' and ' + self.variable_names[1] + ' Influence',
            'Combined ' + self.variable_names[0] + ' and ' + self.variable_names[2] + ' Influence',
            'Combined ' + self.variable_names[1] + ' and ' + self.variable_names[2] + ' Influence',
            'Combined All Variables Influence',
            'Shared Variance All', 'Shared Variance ' + self.variable_names[0] + ' and ' + self.variable_names[1],
            'Shared Variance ' + self.variable_names[0] + ' and ' + self.variable_names[2],
            'Shared Variance ' + self.variable_names[1] + ' and ' + self.variable_names[2],
            'Unique ' + self.variable_names[0], 'Unique ' + self.variable_names[1], 'Unique ' + self.variable_names[2]
        ]
        values = [R1_values, R2_values, R3_values, R12_values, R13_values, R23_values, R123_values, y123_values, y12_values, y13_values, y23_values, y1_values, y2_values, y3_values]
        significances = [sig_R1, sig_R2, sig_R3, sig_R12, sig_R13, sig_R23, sig_R123, sig_y123, sig_y12, sig_y13, sig_y23, sig_y1, sig_y2, sig_y3]

        data_to_append = []

        for var, desc, val, sig in zip(variables, descriptions, values, significances):
            data_to_append.append({
                'Variable': var,
                'Description': desc,
                'Values': val,
                'Significance': sig,
                'Color': None  # Assign colors if desired for visualization
            })

        all_variances_df = pd.concat([all_variances_df, pd.DataFrame(data_to_append)], ignore_index=True)

        return all_variances_df



    def evaluate_4(self, dep_var, num_subjects, eeg_time):
        """Evaluation function for VPA with 4 variables

        Args:
            dep_var (array): Dependent variable
            num_subjects (int): Number of Subjects in independent variable
            eeg_time (int): Number of ms captured in EEG data

        Returns:
            all_variances_df: Pandas dataframe with results
        """

        all_variances_df = pd.DataFrame(columns=['Variable', 'Description', 'Values', 'Significance', 'Color'])

        # Initialize arrays for storing values, including combined R values
        R_values = {f'R{i+1}': np.zeros((num_subjects, eeg_time)) for i in range(4)}
        for i in range(4):
            for j in range(i+1, 4):
                R_values[f'R{i+1}{j+1}'] = np.zeros((num_subjects, eeg_time))
        R_values['R1234'] = np.zeros((num_subjects, eeg_time))
        y_values = {f'y{i+1}': np.zeros((num_subjects, eeg_time)) for i in range(4)}
        for i in range(3):
            for j in range(i+1, 4):
                y_values[f'y{i+1}{j+1}'] = np.zeros((num_subjects, eeg_time))
        y_values['y1234'] = np.zeros((num_subjects, eeg_time))

        R_un = np.zeros(eeg_time)
        R_ln = np.zeros(eeg_time)

        for time in range(eeg_time):
            R_un[time] = self.get_uppernoiseceiling(np.squeeze(dep_var[:,time]))
            R_ln[time] = self.get_lowernoiseceiling(np.squeeze(dep_var[:,time]))

            for i in range(num_subjects):
                this_dep_var = np.squeeze(dep_var[i,time,:])
                results = self.VPA_4(this_dep_var)
                for key, value in results.items():
                    if key in R_values:
                        R_values[key][i, time] = value
                    elif key in y_values:
                        y_values[key][i, time] = value

        # Calculate significance for all R and y values
        sig_values = {}
        for key, value in {**R_values, **y_values}.items():
            sig_values[key] = self.calculate_p_values(value)

        # Define naming convention for the dataframe
        data_to_append = []
        for key, value in R_values.items():
            desc = ' and '.join(key[1:]) + ' Influence' if len(key) > 2 else self.variable_names[int(key[1])-1] + ' Influence'
            data_to_append.append({
                'Variable': key,
                'Description': desc,
                'Values': value,
                'Significance': sig_values[key],
                'Color': None
            })
        for key, value in y_values.items():
            desc = 'Shared Variance of ' + ' and '.join(key[1:]) if len(key) > 4 else 'Unique Variance ' + self.variable_names[int(key[1])-1]
            data_to_append.append({
                'Variable': key,
                'Description': desc,
                'Values': value,
                'Significance': sig_values[key],
                'Color': None
            })

        all_variances_df = pd.concat([all_variances_df, pd.DataFrame(data_to_append)], ignore_index=True)

        return all_variances_df




    def evaluate(self, average_models=False):
        """Function wrapper for the entire VPA evaluation

        Args:
            average_models (bool, optional): Bool if independent variables should be averaged within. Defaults to False.

        Returns:
            all_variances_df: Pandas dataframe with results
        """

        self.average_models=average_models

        # Load dependent variable
        dep_var = self.load_rdms(self.dependent_variable)[0]

        # Get Subjects and time from the dimensions (16, 100, 1225)
        num_subjects = dep_var.shape[0]
        eeg_time = dep_var.shape[1]

        # Calculate VPA
        all_variances_df = self.VPA_function(dep_var, num_subjects, eeg_time)

        return all_variances_df