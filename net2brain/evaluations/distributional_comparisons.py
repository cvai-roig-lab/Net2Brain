import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from .eval_helper import get_npy_files, get_layers_ncondns
import os
from tqdm import tqdm
import glob
import pandas as pd
from sklearn.decomposition import PCA


def reduce_features(X1, X2, print_statement):
    # Check if the number of stimuli match
    
    if print_statement:
        if X1.shape[1] != X2.shape[1]:
            print(f"There is a mismatch between the feature space of DNN data {X1.shape[1]} and fMRI data {X2.shape[1]}")
            print("This is to be expected, hence the larger dimension will be reduced via PCA to the smaller dimension.")

    # Reduce to the smaller number of features
    n_features = min(X1.shape[1], X2.shape[1], X1.shape[0] - 1)  # centred data has at most n_samples - 1 components with variance
    pca_X1 = PCA(n_components=n_features, random_state=0)
    pca_X2 = PCA(n_components=n_features, random_state=0)
    X1_reduced = pca_X1.fit_transform(X1)
    X2_reduced = pca_X2.fit_transform(X2)
    return X1_reduced, X2_reduced

class DistributionalComparison:
    """
    Class for computing distributional comparisons between two datasets
    based on their feature-wise distributions.

    Steps:
    1. Preprocess the datasets to standardize features.
    2. Estimate feature-wise distributions for both datasets.
    3. Compare the distributions using a chosen metric:
       - Jensen-Shannon Divergence (JSD)
       - Wasserstein Distance (Earth Mover's Distance)
    4. Aggregate the feature-wise comparisons into a single score.
    """

    def __init__(self):
        self.reduced_feat_information = True

    def compute_jsd(self, p, q):
        """
        Compute the Jensen-Shannon Divergence between two probability distributions.
        
        Parameters:
            p (numpy.ndarray): First histogram (counts or normalized).
            q (numpy.ndarray): Second histogram, over the same bins as p.
        
        Returns:
            float: Jensen-Shannon Divergence (base 2, between 0 and 1).
        """
        return jensenshannon(p, q, base=2) ** 2

    def compute_wasserstein(self, x1, x2):
        """
        Compute the Wasserstein Distance (Earth Mover's Distance) between two distributions.
        
        Parameters:
            x1 (numpy.ndarray): Samples of the first distribution (1-D).
            x2 (numpy.ndarray): Samples of the second distribution (1-D).
        
        Returns:
            float: Wasserstein Distance.
        """
        return wasserstein_distance(x1, x2)

    def feature_distribution(self, X, bins=50):
        """
        Compute normalized histograms for each feature in a dataset.
        
        Parameters:
            X (numpy.ndarray): Input dataset (n_samples x n_features).
            bins (int): Number of bins for the histograms.
        
        Returns:
            tuple: (histograms, bin_centers) for all features.
        """
        n_features = X.shape[1]
        histograms = []
        bin_centers = []

        for i in range(n_features):
            hist, bin_edges = np.histogram(X[:, i], bins=bins, density=True)
            histograms.append(hist)
            bin_centers.append(0.5 * (bin_edges[:-1] + bin_edges[1:]))  # Midpoints of bins

        return np.array(histograms), np.array(bin_centers)

    
    def compare_distributions(self, X1, X2, metric, bins=50):
        """
        Compare feature-wise distributions between two datasets.

        Parameters:
            X1 (numpy.ndarray): First dataset (n_samples x n_features).
            X2 (numpy.ndarray): Second dataset (n_samples x n_features).
            metric (str): Distance metric ('jsd' or 'wasserstein').
            bins (int): Number of histogram bins, shared by both datasets (only used for 'jsd').

        Returns:
            float: Average distributional distance between the two datasets (lower = more similar).
        """
        # Standardize the data
        scaler_dnn = StandardScaler()
        scaler_fmri = StandardScaler()
        X1 = scaler_dnn.fit_transform(X1)
        X2= scaler_fmri.fit_transform(X2)
        
        # Ensure the number of features matches
        assert X1.shape[1] == X2.shape[1], "Datasets must have the same number of features."

        distances = []
        for i in range(X1.shape[1]):
            if metric == 'jsd':
                edges = np.histogram_bin_edges(np.concatenate([X1[:, i], X2[:, i]]), bins=bins)  # shared bins
                dist = self.compute_jsd(np.histogram(X1[:, i], edges)[0], np.histogram(X2[:, i], edges)[0])
            elif metric == 'wasserstein':
                dist = self.compute_wasserstein(X1[:, i], X2[:, i])
            else:
                raise ValueError("Invalid metric. Choose 'jsd' or 'wasserstein'.")
            distances.append(dist)

        # Average the feature-wise distances
        return np.mean(distances)
    

    def loop_dist_comp(self, feat_path, fmri_data, layer_name, metric, bins):
        """
        Perform Distributional Comparison for a single DNN layer and corresponding fMRI data.
        
        Parameters:
            feat_path (str): Path to the directory containing feature files.
            fmri_data (numpy.ndarray): Raw fMRI data (n_images x n_voxels).
            layer_name (str): Name of the DNN layer to process.
        
        Returns:
            float: The Distributional Comparison score for the given layer and fMRI data.
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

        # Match feature dimensions
        dnn_features, fmri_data = reduce_features(dnn_features, fmri_data, self.reduced_feat_information)
        self.reduced_feat_information = False

        # Compute Distributional Comparison for current layer and roi
        print("Calulating Distributional Comparison score...")
        comp_score = self.compare_distributions(dnn_features, fmri_data, metric, bins)
        
        return comp_score
    
    def prepare_dist_comp(self, feat_path, brain_path, metric, bins, model_name):
        """
        Compute Distributional Comparison scores for all layers of DNN activations with all ROIs' fMRI data
        and return a pandas DataFrame.
        
        Parameters:
            feat_path (str): Path to the directory containing DNN feature files.
            brain_path (str): Path to the directory containing ROI fMRI data files.
            model_name (str): Name of the model for result labeling.
            n_permutations (int): Number of permutations for statistical significance.

        Returns:
            pd.DataFrame: A DataFrame containing Distributional Comparison scores and metadata for each ROI and DNN layer.
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
            roi_name = os.path.splitext(os.path.basename(roi_file))[0]
            
            # Compute Distributional Comparison for each layer
            for layer_name in tqdm(layer_list, desc=f"Processing Layers for ROI: {roi_name}"):
                # Compute Distributional Comparison score for the layer
                comp_score = self.loop_dist_comp(feat_path, fmri_data, layer_name, metric, bins)

                # Construct the row dictionary
                row = {
                    "ROI": roi_name,
                    "Layer": layer_name,
                    "Model": model_name + "_" + metric,
                    "Distance": comp_score,  # lower = more similar
                    "Significance": np.nan,
                    "SEM": np.nan,
                    "LNC": np.nan,  # Placeholder for Local Noise Ceiling
                    "UNC": np.nan   # Placeholder for Upper Noise Ceiling
                }
                rows.append(row)  # Add the row to the list

        # Create the DataFrame from the collected rows
        all_rois_df = pd.DataFrame(rows, columns=['ROI', 'Layer', 'Model', 'Distance', 'Significance', 'SEM', 'LNC', 'UNC'])
        
        return all_rois_df
    
    
    @classmethod
    def run(cls, feat_path, brain_path, metric="wasserstein", bins=50, model_name="Results"):
        """
        Unified entry point for running Distributional Comparison evaluation.

        Parameters:
            feat_path (str): Path to the DNN features.
            brain_path (str): Path to the brain ROI data.
            model_name (str): Name of the model being evaluated.

        Returns:
            pd.DataFrame: DataFrame with Distributional Comparison results.
        """
        instance = cls()
        return instance.prepare_dist_comp(feat_path, brain_path, metric, bins, model_name)



