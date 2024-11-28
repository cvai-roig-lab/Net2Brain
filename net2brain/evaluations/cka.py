import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .eval_helper import get_npy_files, get_layers_ncondns


class CKA:
    """
    Class for computing Centered Kernel Alignment (CKA), a similarity metric
    between two datasets based on their pairwise similarities.

    Steps:
    1. Compute Gram matrices for both datasets using the dot product (linear kernel).
    2. Center the Gram matrices to remove mean effects.
    3. Compute the Hilbert-Schmidt Independence Criterion (HSIC), which measures
       the dependence between the two datasets.
    4. Normalize the HSIC value by the self-similarities of each dataset to compute
       the final CKA score, ranging from 0 (no similarity) to 1 (perfect similarity).
    """

    def centering(self, K):
        """
        Center the Gram matrix K using the centering matrix H.
        
        Parameters:
            K (numpy.ndarray): The Gram matrix (n x n) for input features.
        
        Returns:
            numpy.ndarray: The centered Gram matrix.
        """
        n = K.shape[0]                     # Number of samples
        unit_matrix = np.ones((n, n))      # Matrix of ones
        identity_matrix = np.eye(n)        # Identity matrix
        H = identity_matrix - unit_matrix / n  # Centering matrix

        centered_K = H @ K @ H             # Apply double centering: H * K * H
        return centered_K

    def compute_gram_matrix(self, X):
        """
        Compute the Gram matrix (similarity matrix) for input features X.

        Parameters:
            X (numpy.ndarray): Input features (n x d).

        Returns:
            numpy.ndarray: The Gram matrix (n x n).
        """
        return X @ X.T

    def compute_hsic(self, centered_K1, centered_K2):
        """
        Compute the Hilbert-Schmidt Independence Criterion (HSIC), a measure
        of dependence between two datasets, using their centered Gram matrices.

        Parameters:
            centered_K1 (numpy.ndarray): Centered Gram matrix for the first dataset.
            centered_K2 (numpy.ndarray): Centered Gram matrix for the second dataset.

        Returns:
            float: The HSIC value.
        """
        hsic_value = np.trace(centered_K1 @ centered_K2)
        return hsic_value

    def linear_CKA(self, X, Y):
        """
        Compute the linear Centered Kernel Alignment (CKA) score.

        Parameters:
            X (numpy.ndarray): Input features from one dataset (n x d1).
            Y (numpy.ndarray): Input features from another dataset (n x d2).

        Returns:
            float: The CKA similarity score between X and Y.
        """
        # Standardize the data
        scaler_dnn = StandardScaler()
        scaler_fmri = StandardScaler()
        X = scaler_dnn.fit_transform(X)
        Y = scaler_fmri.fit_transform(Y)
        
        # Compute Gram matrices
        gram_X = self.compute_gram_matrix(X)
        gram_Y = self.compute_gram_matrix(Y)

        # Center the Gram matrices
        centered_gram_X = self.centering(gram_X)
        centered_gram_Y = self.centering(gram_Y)

        # Compute HSIC for X and Y, and normalization terms
        hsic_XY = self.compute_hsic(centered_gram_X, centered_gram_Y)
        hsic_XX = self.compute_hsic(centered_gram_X, centered_gram_X)
        hsic_YY = self.compute_hsic(centered_gram_Y, centered_gram_Y)

        # Normalize HSIC to compute CKA
        cka_score = hsic_XY / np.sqrt(hsic_XX * hsic_YY)
        return cka_score
    
    def loop_cka(self, feat_path, fmri_data, layer_name):
        """
        Perform CKA for a single DNN layer and corresponding fMRI data.
        
        Parameters:
            feat_path (str): Path to the directory containing feature files.
            fmri_data (numpy.ndarray): Raw fMRI data (n_images x n_voxels).
            layer_name (str): Name of the DNN layer to process.
        
        Returns:
            float: The CKA score for the given layer and fMRI data.
        """
        # Gather DNN features for the specified layer across all stimuli
        feat_files = sorted(glob.glob(os.path.join(feat_path, "*.npz")))
        dnn_features = []

        for file in feat_files:
            data = np.load(file)
            layer_features = data[layer_name]  # Extract features for the specific layer
            layer_features = layer_features.flatten() # Flatten the data
            dnn_features.append(layer_features)
        
        
        dnn_features = np.vstack(dnn_features)  # Combine into (n_images, layer_features)

        
         # Check if the number of stimuli match
        if dnn_features.shape[0] != fmri_data.shape[0]:
            raise ValueError(
                f"Mismatch in number of stimuli: "
                f"DNN features have {dnn_features.shape[0]} samples, "
                f"but fMRI data has {fmri_data.shape[0]} samples. "
                f"Number of stimuli must be equal."
            )

        # Compute CKA for current layer and roi
        print("Calulating CKA score...")
        cka_score = self.linear_CKA(dnn_features, fmri_data)
        
        return cka_score
    

    def prepare_cka(self, feat_path, brain_path, model_name="Results", n_permutations=1000):
        """
        Compute CKA scores for all layers of DNN activations with all ROIs' fMRI data
        and return a pandas DataFrame.
        
        Parameters:
            feat_path (str): Path to the directory containing DNN feature files.
            brain_path (str): Path to the directory containing ROI fMRI data files.
            model_name (str): Name of the model for result labeling.
            n_permutations (int): Number of permutations for statistical significance.

        Returns:
            pd.DataFrame: A DataFrame containing CKA scores and metadata for each ROI and DNN layer.
        """
        # Get all ROI fMRI data files
        roi_paths = get_npy_files(brain_path)  # Returns list of all .npy files in the folder

        # List to store rows for the DataFrame
        rows = []

        # Iterate through all ROI files
        for roi_file in roi_paths:
            print(f"Processing ROI file: {roi_file}")

            # Load ROI fMRI data
            fmri_data = np.load(roi_file)  # Shape: (n_images, n_voxels)

            # Get layer names and other metadata
            _, layer_list, _ = get_layers_ncondns(feat_path)  # Helper function to extract layers

            # Extract ROI name
            roi_name = os.path.basename(roi_file).split(".")[0]

            # Compute CKA for each layer
            for layer_name in tqdm(layer_list, desc=f"Processing Layers for ROI: {roi_name}"):
                # Compute CKA score for the layer
                cka_score = self.loop_cka(feat_path, fmri_data, layer_name)

                # Construct the row dictionary
                row = {
                    "ROI": roi_name,
                    "Layer": layer_name,
                    "Model": model_name,
                    "R": cka_score,
                    "%R2": np.nan,  # Placeholder for future use
                    "Significance": np.nan,
                    "SEM": np.nan,
                    "LNC": np.nan,  # Placeholder for Local Noise Ceiling
                    "UNC": np.nan   # Placeholder for Upper Noise Ceiling
                }
                rows.append(row)  # Add the row to the list

        # Create the DataFrame from the collected rows
        all_rois_df = pd.DataFrame(rows, columns=['ROI', 'Layer', 'Model', 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])
        
        return all_rois_df
    
    
    @classmethod
    def run(cls, feat_path, brain_path, model_name="Results"):
        """
        Unified entry point for running CKA evaluation.

        Parameters:
            feat_path (str): Path to the DNN features.
            brain_path (str): Path to the brain ROI data.
            model_name (str): Name of the model being evaluated.

        Returns:
            pd.DataFrame: DataFrame with CKA results.
        """
        instance = cls()
        return instance.prepare_cka(feat_path, brain_path, model_name=model_name)

        
        