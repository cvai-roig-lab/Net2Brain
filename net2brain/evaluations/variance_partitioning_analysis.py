import itertools
import os
import warnings
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
        self.independent_variables = list(independent_variables)





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
        rho, _ = scipy.stats.spearmanr(lt_rdm1, lt_rdm2)
        return rho ** 2

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
    
    



    def _as_array(self, loaded):
        """Return a plain numpy array from whatever ``load`` produced.

        Args:
            loaded: Either a numpy array (e.g. from a .npy) or a dict-like
                object (e.g. an .npz/.mat) holding the RDM under 'rdm'/'arr_0'.

        Returns:
            numpy array: The extracted RDM.
        """
        if isinstance(loaded, np.ndarray):
            return loaded
        if 'rdm' in loaded:
            return np.asarray(loaded['rdm'])
        if 'arr_0' in loaded:
            return np.asarray(loaded['arr_0'])
        raise ValueError("The RDM file does not contain 'rdm' or 'arr_0' keys.")

    def _load_dependent(self, dependent_variable):
        """Load the dependent variable into shape (subjects, time, pairs).

        A single path is treated as one pre-stacked array. A list of paths is
        treated as one RDM per subject and stacked. Square per-subject RDMs are
        reduced to their upper triangle; condensed vectors and (time, pairs)
        matrices are kept as-is. Data without a time axis is given a singleton
        time axis so the time-resolved machinery can run on it unchanged.

        Args:
            dependent_variable (str or list): Path or list of paths to the RDMs.

        Returns:
            tuple: (dep_var, is_timeresolved), where dep_var has shape
                (subjects, time, pairs) and is_timeresolved is False when a
                singleton time axis had to be inserted.
        """
        if isinstance(dependent_variable, str):
            dep_var = self._as_array(load(dependent_variable))
        else:
            subject_rdms = []
            for path in dependent_variable:
                rdm = self._as_array(load(path))
                if rdm.ndim == 2 and rdm.shape[0] == rdm.shape[1]:
                    rdm = self.get_uppertriangular(rdm)
                subject_rdms.append(rdm)
            dep_var = np.array(subject_rdms)

        if dep_var.ndim not in (2, 3):
            raise ValueError(
                f"Dependent variable resolved to shape {dep_var.shape}; expected "
                "2D (subjects, pairs) or 3D (subjects, time, pairs)."
            )

        if dep_var.shape[-1] < 2:
            raise ValueError(
                f"Dependent variable has only {dep_var.shape[-1]} condition pair(s); "
                "VPA needs at least 2 (i.e. 3 or more conditions)."
            )

        is_timeresolved = dep_var.ndim == 3
        if not is_timeresolved:
            dep_var = dep_var[:, np.newaxis, :]

        return dep_var, is_timeresolved

    def _load_independent(self, independent_variable):
        """Load an independent variable into shape (n_models, pairs).

        Accepts either a single path or a list of paths. Each RDM is reduced to
        its upper-triangular vector, so a bare string is handled the same as a
        one-element list instead of falling through to the EEG code path in
        ``load_rdms``.

        Args:
            independent_variable (str or list): Path or list of paths to RDMs.

        Returns:
            numpy array: Stacked upper-triangular RDM vectors.
        """
        paths = [independent_variable] if isinstance(independent_variable, str) else list(independent_variable)
        vectors = []
        for path in paths:
            rdm = self.check_squareform(self._as_array(load(path)))
            vectors.append(self.get_uppertriangular(rdm))
        return np.array(vectors)

    def _subsets(self, k):
        """All non-empty subsets of the k variables, ordered by size"""
        return [c for r in range(1, k + 1) for c in itertools.combinations(range(k), r)]

    def _components(self, r2, k):
        """Unique and shared variance from the R2 of every subset (last axis, ordered as _subsets).

        The part shared by exactly the variables in c is the sum over non-empty A in c of
        (-1)^(|A|+1) * (R2(rest + A) - R2(rest)), where rest are the other variables.
        """
        subsets = self._subsets(k)
        pos = {c: i for i, c in enumerate(subsets)}
        f = lambda key: r2[..., pos[key]] if key else 0
        out = []
        for c in subsets:
            rest = tuple(j for j in range(k) if j not in c)
            value = 0
            for r in range(1, len(c) + 1):
                for A in itertools.combinations(c, r):
                    value = value + (-1) ** (r + 1) * (f(tuple(sorted(rest + A))) - f(rest))
            out.append(value)
        return np.stack(out, -1)

    def _pair_index(self, perm, n_stim, rows, cols):
        """Index into the condensed RDM vector after relabelling the stimuli with perm (rows, cols: np.triu_indices)"""
        a, b = perm[rows], perm[cols]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return n_stim * lo - lo * (lo + 1) // 2 + hi - lo - 1

    def _design(self, n_pairs):
        """Load the predictor RDMs once. Returns the centred design (pairs, columns) and the columns per variable."""
        blocks, columns, start = [], [], 0
        for variable in self.independent_variables:
            rdms = self._load_independent(variable)
            if self.average_models:
                rdms = np.mean(rdms, axis=0, keepdims=True)
            if rdms.shape[1] != n_pairs:
                raise ValueError(f"The predictor RDMs have {rdms.shape[1]} pairs, the brain RDMs {n_pairs}.")
            blocks.append(rdms.T)
            columns.append(list(range(start, start + len(rdms))))
            start += len(rdms)
        X = np.concatenate(blocks, axis=1).astype(float)
        return X - X.mean(0), columns

    def _subset_r2(self, Y, yy, X, subset_columns, G_inv):
        """In-sample OLS R2 (with intercept) of every subset for every row of Y, (rows, subsets).

        Y holds the centred brain RDMs (rows, pairs) and yy their squared norms; X is the centred, possibly
        row-permuted design (pairs, columns). A permutation does not change X^T X, so
        R2 = b^T (X_s^T X_s)^-1 b / |y|^2 with b = X_s^T y.
        """
        B = Y @ X
        r2 = np.stack([np.einsum("ni,ij,nj->n", B[:, c], g, B[:, c]) for c, g in zip(subset_columns, G_inv)], axis=-1)
        return r2 / yy[:, None]

    def _row_description(self, kind, subset, k):
        """Description of an R (kind 'R') or variance component (kind 'y') row"""
        names = [self.variable_names[j] for j in subset]
        if kind == "R":
            if len(subset) == 1:
                return names[0] + " Influence"
            return "Combined All Variables Influence" if len(subset) == k else "Combined " + " and ".join(names) + " Influence"
        if len(subset) == 1:
            return "Unique " + names[0]
        return "Shared Variance All" if len(subset) == k else "Shared Variance " + " and ".join(names)

    def evaluate(self, average_models=False, n_permutations=1000, seed=0):
        """Function wrapper for the entire VPA evaluation

        For every subject (and timepoint) the brain RDM is regressed on every subset of the independent variables
        (OLS with intercept, in-sample R2). R<vars> are these R2 values and y<vars> the unique (one variable) and
        shared (several variables) components obtained from them.

        Significance comes from a permutation test: the stimulus labels of the predictor RDMs are permuted (rows and
        columns together, the same permutation for every predictor), and the group mean over subjects is compared
        with the permutation null. For time-resolved data the maximum over timepoints is used as null (corrected
        for all timepoints; each tested row is corrected on its own, not across rows). p-values are given for every
        R row (R1, R12, ..., each a Mantel-type test of that combination's R2). The variance components (y1, y12,
        ...) have NaN as Significance: this permutation tests only the null that no variable relates to the brain,
        which does not hold for a component while another variable explains the brain (an unrelated variable's
        unique variance was significant in up to ~30% of simulated data sets). The test generalizes over stimuli;
        the subjects are treated as fixed.

        Args:
            average_models (bool, optional): Bool if independent variables should be averaged within. Defaults to False.
            n_permutations (int, optional): Number of permutations. 0 skips the test (Significance is NaN).
                Defaults to 1000.
            seed (int, optional): Seed of the permutations. Defaults to 0.

        Returns:
            all_variances_df: Pandas dataframe with results
        """

        self.average_models=average_models

        # Load dependent variable (subjects, time, pairs)
        dep_var, is_timeresolved = self._load_dependent(self.dependent_variable)
        num_subjects, eeg_time, n_pairs = dep_var.shape
        n_stim = int(round((1 + np.sqrt(1 + 8 * n_pairs)) / 2))

        X, columns = self._design(n_pairs)
        k = len(columns)
        subsets = self._subsets(k)
        subset_columns = [sum((columns[j] for j in c), []) for c in subsets]
        G = X.T @ X
        G_inv = [np.linalg.pinv(G[np.ix_(c, c)]) for c in subset_columns]

        R_un = np.zeros(eeg_time)
        R_ln = np.zeros(eeg_time)
        for time in range(eeg_time):
            R_un[time] = self.get_uppernoiseceiling(np.squeeze(dep_var[:,time]))
            R_ln[time] = self.get_lowernoiseceiling(np.squeeze(dep_var[:,time]))

        # Centred brain RDMs, one row per subject and timepoint (centred in place, without temporaries: dep_var is
        # loaded here and not used afterwards)
        Y = np.asarray(dep_var, dtype=float).reshape(-1, n_pairs)
        Y -= Y.mean(axis=1, keepdims=True)
        yy = np.einsum("ij,ij->i", Y, Y)
        r2 = self._subset_r2(Y, yy, X, subset_columns, G_inv).reshape(num_subjects, eeg_time, len(subsets))
        comps = self._components(r2, k)

        # Permutation null of the group mean of every R row, maximum over timepoints
        observed = r2.mean(axis=0)
        p_values = np.full(observed.shape, np.nan)
        if n_permutations > 0:
            rng = np.random.default_rng(seed)
            rows, cols = np.triu_indices(n_stim, 1)
            null_max = np.empty((n_permutations, observed.shape[1]))
            for i in range(n_permutations):
                X_perm = X[self._pair_index(rng.permutation(n_stim), n_stim, rows, cols)]
                r2_perm = self._subset_r2(Y, yy, X_perm, subset_columns, G_inv)
                null_max[i] = r2_perm.reshape(num_subjects, eeg_time, len(subsets)).mean(axis=0).max(axis=0)
            p_values = (1 + (null_max[:, None, :] >= observed[None]).sum(axis=0)) / (1 + n_permutations)
            # NaN R2 (e.g. NaN in the brain RDMs) would otherwise count as never exceeded and give the smallest p
            p_values[np.isnan(observed) | np.isnan(null_max).any(axis=0)] = np.nan
            if np.isnan(observed).any():
                warnings.warn("R2 is NaN for some brain RDMs (NaN values, or no variance), so the p-values of every "
                              "timepoint are NaN (the test is corrected over timepoints).")

        # Dataframe: R rows, then components, then the noise ceiling
        data_to_append = []
        for i, c in enumerate(subsets):
            data_to_append.append({'Variable': 'R' + ''.join(str(j + 1) for j in c),
                                   'Description': self._row_description('R', c, k),
                                   'Values': r2[..., i], 'Significance': p_values[:, i], 'Color': None})
        for i, c in enumerate(subsets):
            data_to_append.append({'Variable': 'y' + ''.join(str(j + 1) for j in c),
                                   'Description': self._row_description('y', c, k),
                                   'Values': comps[..., i], 'Significance': np.full(eeg_time, np.nan), 'Color': None})

        # Surface the noise ceiling as its own rows instead of discarding it
        data_to_append.append({'Variable': 'UNC', 'Description': 'Upper Noise Ceiling', 'Values': R_un, 'Significance': np.full(eeg_time, np.nan), 'Color': None})
        data_to_append.append({'Variable': 'LNC', 'Description': 'Lower Noise Ceiling', 'Values': R_ln, 'Significance': np.full(eeg_time, np.nan), 'Color': None})

        all_variances_df = pd.DataFrame(data_to_append, columns=['Variable', 'Description', 'Values', 'Significance', 'Color'])

        # Drop the singleton time axis from the results when the data has no time dimension
        if not is_timeresolved:
            for column in ('Values', 'Significance'):
                all_variances_df[column] = [
                    np.squeeze(np.asarray(value), axis=-1) for value in all_variances_df[column]
                ]

        return all_variances_df