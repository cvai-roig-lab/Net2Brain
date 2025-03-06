"""This code is adapted from https://github.com/brainML/Stacking
Ruogu Lin, Thomas Naselaris, Kendrick Kay, and Leila Wehbe (2023). Stacked regressions and structured variance partitioning for interpretable brain maps."""


import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore, ttest_1samp, sem
from cvxopt import matrix, solvers
from sklearn.decomposition import PCA, IncrementalPCA
from .stacked_ridge_tools import cross_val_ridge, R2, ridge, R2r
from .stacked_variance_partitioning import forward_variance_partitioning, backward_variance_partitioning
from .eval_helper import get_npy_files

# Set option to not show progress in CVXOPT solver
solvers.options["show_progress"] = False


def get_cv_indices(n_samples, n_folds):
    """Generate cross-validation indices.

    Args:
        n_samples (int): Number of samples to generate indices for.
        n_folds (int): Number of folds to use in cross-validation.

    Returns:
        numpy.ndarray: Array of cross-validation indices with shape (n_samples,).
    """
    cv_indices = np.zeros((n_samples))
    n_items = int(np.floor(n_samples / n_folds))  # number of items in one fold
    for i in range(0, n_folds - 1):
        cv_indices[i * n_items : (i + 1) * n_items] = i
    cv_indices[(n_folds - 1) * n_items :] = n_folds - 1
    return cv_indices


def feat_ridge_CV(
    train_features,
    train_targets,
    test_features,
    method="cross_val_ridge",
    n_folds=5,
    score_function=R2,
):
    """Train a ridge regression model with cross-validation and predict on test_features.

    Args:
        train_features (numpy.ndarray): Array of shape (n_samples, n_features) containing the training features.
        train_targets (numpy.ndarray): Array of shape (n_samples, n_targets) containing the training targets.
        test_features (numpy.ndarray): Array of shape (n_test_samples, n_features) containing the test features.
        method (str): Method to use for ridge regression. Options are "simple_ridge" and "cross_val_ridge".
            Defaults to "cross_val_ridge".
        n_folds (int): Number of folds to use in cross-validation. Defaults to 5.
        score_function (callable): Scoring function to use for cross-validation. Defaults to R2.

    Returns:
        tuple: Tuple containing:
            - preds_train (numpy.ndarray): Array of shape (n_samples, n_targets) containing the training set predictions.
            - err (numpy.ndarray): Array of shape (n_samples, n_targets) containing the training set errors.
            - preds_test (numpy.ndarray): Array of shape (n_test_samples, n_targets) containing the test set predictions.
            - r2s_train_fold (numpy.ndarray): Array of shape (n_folds,) containing the cross-validation scores.
            - var_train_fold (numpy.ndarray): Array of shape (n_targets,) containing the variances of the training set predictions.
    """

    if np.all(train_features == 0):
        # If there are no predictors, return zero weights and zero predictions
        weights = np.zeros((train_features.shape[1], train_targets.shape[1]))
        train_preds = np.zeros_like(train_targets)
    else:
        # Use cross-validation to train the model
        cv_indices = get_cv_indices(train_targets.shape[0], n_folds=n_folds)
        train_preds = np.zeros_like(train_targets)

        for i_cv in range(n_folds):
            train_targets_cv = np.nan_to_num(zscore(train_targets[cv_indices != i_cv]))
            train_features_cv = np.nan_to_num(
                zscore(train_features[cv_indices != i_cv])
            )
            test_features_cv = np.nan_to_num(zscore(train_features[cv_indices == i_cv]))

            if method == "simple_ridge":
                # Use a fixed regularization parameter to train the model
                weights = ridge(train_features, train_targets, 100)
            elif method == "cross_val_ridge":
                # Use cross-validation to select the best regularization parameter
                lambdas = np.array([10**i for i in range(-6, 10)])
                if train_features.shape[1] > train_features.shape[0]:
                    weights, __ = cross_val_ridge(
                        train_features_cv,
                        train_targets_cv,
                        n_splits=5,
                        lambdas=lambdas,
                        do_plot=False,
                        method="plain",
                    )
                else:
                    weights, __ = cross_val_ridge(
                        train_features_cv,
                        train_targets_cv,
                        n_splits=5,
                        lambdas=lambdas,
                        do_plot=False,
                        method="plain",
                    )

            # Make predictions on the current fold of the data
            train_preds[cv_indices == i_cv] = test_features_cv.dot(weights)

    # Calculate prediction error on the training set
    train_err = train_targets - train_preds

    # Retrain the model on all of the training data
    lambdas = np.array([10**i for i in range(-6, 10)])
    weights, __ = cross_val_ridge(
        train_features,
        train_targets,
        n_splits=5,
        lambdas=lambdas,
        do_plot=False,
        method="plain",
    )

    # Make predictions on the test set using the retrained model
    test_preds = np.dot(test_features, weights)

    # Calculate the score on the training set
    train_scores = score_function(train_preds, train_targets)
    train_variances = np.var(train_preds, axis=0)

    return train_preds, train_err, test_preds, train_scores, train_variances





def stacking_fmri(
    train_data,
    test_data,
    train_features,
    test_features,
    method="cross_val_ridge",
    score_f=R2,
):
    """
    Stacks predictions from different feature spaces and uses them to make final predictions.

    Args:
        train_data (ndarray): Training data of shape (n_time_train, n_voxels)
        test_data (ndarray): Testing data of shape (n_time_test, n_voxels)
        train_features (list): List of training feature spaces, each of shape (n_time_train, n_dims)
        test_features (list): List of testing feature spaces, each of shape (n_time_test, n_dims)
        method (str): Name of the method used for training. Default is 'cross_val_ridge'.
        score_f (callable): Scikit-learn scoring function to use for evaluation. Default is mean_squared_error.

    Returns:
        Tuple of ndarrays:
            - r2s: Array of shape (n_features, n_voxels) containing unweighted R2 scores for each feature space and voxel
            - stacked_r2s: Array of shape (n_voxels,) containing R2 scores for the stacked predictions of each voxel
            - r2s_weighted: Array of shape (n_features, n_voxels) containing R2 scores for each feature space weighted by stacking weights
            - r2s_train: Array of shape (n_features, n_voxels) containing R2 scores for each feature space and voxel in the training set
            - stacked_train_r2s: Array of shape (n_voxels,) containing R2 scores for the stacked predictions of each voxel in the training set
            - S: Array of shape (n_voxels, n_features) containing the stacking weights for each voxel
    """

    # Number of time points in the test set
    n_time_test = test_data.shape[0]

    # Check that the number of voxels is the same in the training and test sets
    assert train_data.shape[1] == test_data.shape[1]
    n_voxels = train_data.shape[1]

    # Check that the number of feature spaces is the same in the training and test sets
    assert len(train_features) == len(test_features)
    n_features = len(train_features)

    # Array to store R2 scores for each feature space and voxel
    r2s = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space and voxel in the training set
    r2s_train = np.zeros((n_features, n_voxels))
    # Array to store variance explained by the model for each feature space and voxel in the training set
    var_train = np.zeros((n_features, n_voxels))
    # Array to store R2 scores for each feature space weighted by stacking weights
    r2s_weighted = np.zeros((n_features, n_voxels))

    # Array to store stacked predictions for each voxel
    stacked_pred = np.zeros((n_time_test, n_voxels))
    # Dictionary to store predictions for each feature space and voxel in the training set
    preds_train = {}
    # Dictionary to store predictions for each feature space and voxel in the test set
    preds_test = np.zeros((n_features, n_time_test, n_voxels))
    # Array to store weighted predictions for each feature space and voxel in the test set
    weighted_pred = np.zeros((n_features, n_time_test, n_voxels))

    # normalize data by TRAIN/TEST
    train_data = np.nan_to_num(zscore(train_data))
    test_data = np.nan_to_num(zscore(test_data))

    train_features = [np.nan_to_num(zscore(F)) for F in train_features]
    test_features = [np.nan_to_num(zscore(F)) for F in test_features]

    # initialize an error dictionary to store errors for each feature
    err = dict()
    preds_train = dict()

    # iterate over each feature and train a model using feature ridge regression
    for FEATURE in range(n_features):
        (
            preds_train[FEATURE],
            error,
            preds_test[FEATURE, :, :],
            r2s_train[FEATURE, :],
            var_train[FEATURE, :],
        ) = feat_ridge_CV(
            train_features[FEATURE], train_data, test_features[FEATURE], method=method
        )
        err[FEATURE] = error

    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # solve the quadratic programming problem to obtain the weights for stacking
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))
    stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_features)

        # combine the predictions from the individual feature spaces for voxel i
        z_test = np.array(
            [preds_test[feature_j, :, i] for feature_j in range(n_features)]
        )
        z_train = np.array(
            [preds_train[feature_j][:, i] for feature_j in range(n_features)]
        )
        # multiply the predictions by S[i,:]
        stacked_pred[:, i] = np.dot(S[i, :], z_test)
        # combine the training predictions from the individual feature spaces for voxel i
        stacked_pred_train[:, i] = np.dot(S[i, :], z_train)

    # compute the R2 score for the stacked predictions on the training data
    stacked_train_r2s = score_f(stacked_pred_train, train_data)

    # compute the R2 scores for each individual feature and the weighted feature predictions
    for FEATURE in range(n_features):
        # weight the predictions according to S:
        # weighted single feature space predictions, computed over a fold
        weighted_pred[FEATURE, :] = preds_test[FEATURE, :] * S[:, FEATURE]

    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], test_data)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], test_data)

    # compute the R2 score for the stacked predictions on the test data
    stacked_r2s = score_f(stacked_pred, test_data)

    # return the results
    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train_r2s,
        S,
    )


def stacking_CV_fmri(data, features, method="cross_val_ridge", n_folds=5, score_f=R2):
    """
    A function that performs cross-validated feature stacking to predict fMRI
    signal from a set of predictors.

    Args:
    - data (ndarray): A matrix of fMRI signal data with dimensions n_time x n_voxels.
    - features (list): A list of length n_features containing arrays of predictors
      with dimensions n_time x n_dim.
    - method (str): A string indicating the method to use to train the model. Default is "cross_val_ridge".
    - n_folds (int): An integer indicating the number of cross-validation folds to use. Default is 5.
    - score_f (function): A function to use for scoring the model. Default is R2.

    Returns:
    - A tuple containing the following elements:
      - r2s (ndarray): An array of shape (n_features, n_voxels) containing the R2 scores
        for each feature and voxel.
      - r2s_weighted (ndarray): An array of shape (n_features, n_voxels) containing the R2 scores
        for each feature and voxel, weighted by stacking weights.
      - stacked_r2s (float): The R2 score for the stacked predictions.
      - r2s_train (ndarray): An array of shape (n_features, n_voxels) containing the R2 scores
        for each feature and voxel for the training set.
      - stacked_train (float): The R2 score for the stacked predictions for the training set.
      - S_average (ndarray): An array of shape (n_features, n_voxels) containing the stacking weights
        for each feature and voxel.

    """

    n_time, n_voxels = data.shape
    n_features = len(features)

    ind = get_cv_indices(n_time, n_folds=n_folds)

    # create arrays to store results
    r2s = np.zeros((n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    var_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))
    stacked_pred = np.zeros((n_time, n_voxels))
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_features, n_time, n_voxels))
    S_average = np.zeros((n_voxels, n_features))

    # perform cross-validation by fold
    for ind_num in tqdm(range(n_folds), desc="Processing folds"):
        # split data into training and testing sets
        train_ind = ind != ind_num
        test_ind = ind == ind_num
        train_data = data[train_ind]
        train_features = [F[train_ind] for F in features]
        test_data = data[test_ind]
        test_features = [F[test_ind] for F in features]

        # normalize data
        train_data = np.nan_to_num(zscore(train_data))
        test_data = np.nan_to_num(zscore(test_data))

        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]

        # Store prediction errors and training predictions for each feature
        err = dict()
        preds_train = dict()
        for FEATURE in range(n_features):
            (
                preds_train[FEATURE],
                error,
                preds_test[FEATURE, test_ind],
                r2s_train_folds[ind_num, FEATURE, :],
                var_train_folds[ind_num, FEATURE, :],
            ) = feat_ridge_CV(
                train_features[FEATURE],
                train_data,
                test_features[FEATURE],
                method=method,
            )
            err[FEATURE] = error

        # calculate error matrix for stacking
        P = np.zeros((n_voxels, n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                P[:, i, j] = np.mean(err[i] * err[j], axis=0)

        # Set optimization parameters for computing stacking weights
        q = matrix(np.zeros((n_features)))
        G = matrix(-np.eye(n_features, n_features))
        h = matrix(np.zeros(n_features))
        A = matrix(np.ones((1, n_features)))
        b = matrix(np.ones(1))

        S = np.zeros((n_voxels, n_features))
        stacked_pred_train = np.zeros_like(train_data)

        # Compute stacking weights and combined predictions for each voxel
        for i in range(n_voxels):
            PP = matrix(P[i])
            # solve for stacking weights for every voxel
            S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(
                n_features,
            )
            # combine the predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_test[feature_j, test_ind, i] for feature_j in range(n_features)]
            )
            # multiply the predictions by S[i,:]
            stacked_pred[test_ind, i] = np.dot(S[i, :], z)
            # combine the training predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_train[feature_j][:, i] for feature_j in range(n_features)]
            )
            stacked_pred_train[:, i] = np.dot(S[i, :], z)

        S_average += S
        stacked_train_r2s_fold[ind_num, :] = score_f(stacked_pred_train, train_data)

        # Compute weighted single feature space predictions, computed over a fold
        for FEATURE in range(n_features):
            weighted_pred[FEATURE, test_ind] = (
                preds_test[FEATURE, test_ind] * S[:, FEATURE]
            )

    # Compute overall performance metrics
    data_zscored = zscore(data)
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], data_zscored)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], data_zscored)

    stacked_r2s = score_f(stacked_pred, data_zscored)

    r2s_train = r2s_train_folds.mean(0)
    stacked_train = stacked_train_r2s_fold.mean(0)
    S_average = S_average / n_folds

    # return the results
    return (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train,
        S_average,
    )



def Stacked_Encoding(feat_path, 
                    roi_path, 
                    model_name, 
                    n_folds=3, 
                    n_components=None, 
                    vpa=True,
                    save_path='stacked_encoding_results'):
    """
    Perform stacked encoding analysis to relate model activations from multiple layers to fMRI data.
    
    Stacked encoding combines predictions from different feature spaces or model layers to predict
    brain activity patterns. This approach handles correlated feature spaces that can arise with 
    naturalistic stimuli and provides interpretable weights showing the importance of each layer.
    
    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
            If a list of folders is provided, each folder's content will be summarized into one value.
            This is useful when a folder contains data for the same ROI across different subjects.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        n_folds (int): Number of folds to use for cross-validation in the stacked encoding process.
        n_components (int, optional): Number of principal components to retain in PCA. 
            If None, PCA will not be used 
            If a specific number is provided, PCA will be used with these number of components.
        vpa (bool): If true, will computed stacked variance partitioning anaylsis and save locally
        save_path (str): Directory path to save the results.
        
    Returns:
        pd.DataFrame: DataFrame containing results for each layer and ROI, including:
            - ROI: Name of the brain region
            - Layer: Name of the model layer or "stacked_layers" for the combined model
            - Model: Name of the model
            - R: Correlation coefficient (square root of R²)
            - Significance: p-value for statistical significance
            - SEM: Standard error of the mean
            - Various additional metrics
    """
    
    print("This code is adapted from https://github.com/brainML/Stacking")
    
    # Turn the roi_path into a list of files
    roi_paths = get_npy_files(roi_path)
    
    # Prpeare results dataframe
    all_results_df = pd.DataFrame()
    
            
    # Create the output folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Iterate through all folder paths
    for counter, roi_path in enumerate(roi_paths):
        print(f"Processing file {counter}, {roi_path}")
        
        if n_components == None:
            use_pca = False
        else:
            use_pca = True
        result_dataframe = _stacked_encoding(feat_path, roi_path, model_name,  save_path, n_folds, use_pca, n_components, vpa)
        
        # Append to all dataframes that we get per ROI
        all_results_df = pd.concat([all_results_df, result_dataframe], ignore_index=True)  # Append DataFrames

        
    return all_results_df
        
    

def apply_pca_to_features(features, n_components=100):
    """
    Apply PCA dimensionality reduction to feature data.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature data with shape (n_samples, n_features)
    n_components : int
        Number of PCA components to retain
        
    Returns:
    --------
    numpy.ndarray
        PCA-transformed features with shape (n_samples, n_components)
    float
        Percentage of variance explained by the retained components
    """
    
    
    # Make sure we don't try to extract more components than possible
    max_components = min(n_components, min(features.shape[0], features.shape[1]))
    
    # Use incremental PCA for very large feature spaces
    if features.shape[1] > 10000:
        pca = IncrementalPCA(n_components=max_components)
    else:
        pca = PCA(n_components=max_components)
        
    reduced_features = pca.fit_transform(features)
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    
    return reduced_features, explained_variance        

# Calculate SEM for each layer and stacked model
def compute_sem(values):
    """Compute the standard error of the mean"""
    return np.std(values) / np.sqrt(len(values))
        
        
# Compute stacked encoding
def _stacked_encoding(feat_path, roi_path, model_name,  save_path, n_folds, use_pca=True, n_components=100, vpa=True):
    """
    Perform stacked encoding to predict brain activity from multiple feature spaces.
        
    Parameters:
    -----------
    feat_path : str
        Path to the directory containing feature files (.npz) for multiple layers
    roi_path : str
        Path to the ROI file (.npy) containing brain activity data
    model_name : str
        Name of the model, used for result labeling and saving
    save_path : str
        Directory where results will be saved
    n_folds : int
        Number of cross-validation folds for stacked encoding
    use_pca : bool, optional
        Whether to apply PCA for dimensionality reduction (default: True)
    n_components : int, optional
        Number of PCA components to keep if use_pca is True (default: 100)
    vpa: bool
        If true, will computed stacked variance partitioning anaylsis and save locally
        
    Returns:
    --------
    pandas.DataFrame
    """
    
    # Load ROI data
    print(f"Loading ROI data from {roi_path}")
    roi_data = np.load(roi_path, allow_pickle=True)
    roi_name = os.path.basename(roi_path).split(".")[0]
    print(f"ROI data shape: {roi_data.shape}")
    n_samples = roi_data.shape[0]
    
    # Get feature files
    feat_files = glob.glob(os.path.join(feat_path, '*.np[zy]'))
    feat_files.sort()
    print(f"Found {len(feat_files)} feature files")
    
    # Scan first file to get layer names
    first_file = feat_files[0]
    if first_file.endswith('.npz'):
        with np.load(first_file) as data:
            layer_names = list(data.keys())
    else:
        raise ValueError("First feature file must be .npz to identify layers")
    
    print(f"Found {len(layer_names)} layers: {layer_names}")
    
    # Initialize feature arrays for each layer
    all_features = []
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"Processing layer: {layer_name}")
        
        # For each layer, we'll create a matrix of shape (n_samples, flattened_features)
        layer_features = np.zeros((n_samples, 0))  # Will update the second dimension once we know the feature size
        
        # Process each file (each file = one image)
        for i, file in enumerate(feat_files):
            if i >= n_samples:
                raise ValueError(f"Warning: More feature files than ROI samples {i} > {n_samples}")
                
            try:
                # Load the layer data for this image
                layer_data = np.load(file)[layer_name]
                
                # Flatten the features (regardless of original shape)
                flattened = layer_data.flatten()
                
                # On first iteration, resize our array once we know the feature dimension
                if i == 0:
                    layer_features = np.zeros((n_samples, flattened.shape[0]))
                
                # Add flattened features to our array
                layer_features[i, :] = flattened
                
            except KeyError:
                raise ValueError(f"Layer {layer_name} not found in file {file}")
        
        
        # Apply PCA for dimensionality reduction if requested
        if use_pca and layer_features.shape[1] > n_components:
            original_shape = layer_features.shape
            layer_features, explained_variance = apply_pca_to_features(layer_features, n_components)
            print(f"  Original feature shape for {layer_name}: {original_shape} reshaped to {layer_features.shape} (exp. var: {explained_variance:.2f}%)")
        else:
            if not use_pca:
                print("  Skipping PCA as requested. If you want to apply PCA add n_components=x")
            elif layer_features.shape[1] <= n_components:
                print(f"  Skipping PCA as feature dimension ({layer_features.shape[1]}) is already <= {n_components}")
        
        
        all_features.append(layer_features)
    
    print(f"Running stacked encoding with {len(all_features)} feature spaces...")
    # Run stacked encoding
    r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train, S_average = stacking_CV_fmri(
        data=roi_data,
        features=all_features,
        method="cross_val_ridge",
        n_folds=n_folds,
        score_f=R2
    )
    
    vp_results = {}
    if vpa:
        # Forward variance partitioning
        print("Running variance partitioning: Forward Pass")
        vp, vp_square, r2s_array, vp_sel_layer_forward = forward_variance_partitioning(r2s, stacked_r2s)
        print(f"Forward VP shape: {vp.shape}")
        print(f"Forward VP square shape: {vp_square.shape}")


        # Backward variance partitioning
        print("Running variance partitioning: Backward Pass")
        vpr, vpr_square, r2sr, vpr_sel_layer_backward = backward_variance_partitioning(r2s)
        print(f"Backward VP shape: {vpr.shape}")
        print(f"Backward VP square shape: {vpr_square.shape}")
        print("Selected layers you will find in save-file")
        
        vp_results = {
        'vp': vp,
        'vp_square': vp_square,
        'r2s_array': r2s_array,
        'vp_sel_layer_forward': vp_sel_layer_forward,
        'vpr': vpr,
        'vpr_square': vpr_square,
        'r2sr': r2sr,
        'vpr_sel_layer_backward': vpr_sel_layer_backward}

        
    # Store results in standard format
    results_list = []
    
    # Add results for each individual layer
    for i, layer_name in enumerate(layer_names):
        # Calculate SEM across voxels
        sem_value = compute_sem(r2s[i, :])
        
        # Calculate significance (p-value) compared to zero using t-test
        t_stat, p_value = ttest_1samp(r2s[i, :], 0)
        significance = p_value
        
        # Average R values
        r_values = np.sqrt(np.abs(r2s[i, :]))
        avg_r = np.mean(r_values)
        
        # Average R2 across voxels
        avg_r2 = np.mean(r2s[i, :])
        
        # Create result dictionary for this layer
        layer_result = {
            "ROI": roi_name,
            "Layer": layer_name,
            "Model": model_name,
            "R": avg_r,
            "R2": avg_r2,
            "%R2": np.nan,
            "Significance": significance,
            "SEM": sem_value,
            "LNC": np.nan,
            "UNC": np.nan
        }
        results_list.append(layer_result)
    
    # Add result for stacked model
    sem_stacked = compute_sem(stacked_r2s)
    t_stat_stacked, p_value_stacked = ttest_1samp(stacked_r2s, 0)
    significance_stacked = p_value_stacked
    avg_stacked_r2 = np.mean(stacked_r2s)
    
    stacked_r_values = np.sqrt(np.abs(stacked_r2s))
    avg_stacked_r = np.mean(stacked_r_values)
    
    stacked_result = {
        "ROI": roi_name,
        "Layer": "stacked_layers",
        "Model": model_name,
        "R": avg_stacked_r,
        "R2": avg_stacked_r2,
        "%R2": np.nan,
        "Significance": significance_stacked,
        "SEM": sem_stacked,
        "LNC": np.nan,
        "UNC": np.nan
    }
    results_list.append(stacked_result)
    
    all_rois_df = pd.DataFrame(results_list, columns=['ROI', 'Layer', 'Model', 'R', 'R2', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])
    
    # Save detailed results
    output_dir = os.path.join(save_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{roi_name}_{model_name}_stacked_encoding.npz")
    
    np.savez(
        output_file,
        r2s=r2s,
        stacked_r2s=stacked_r2s,
        r2s_weighted=r2s_weighted,
        S_average=S_average,
        layer_names=layer_names,
        # Add each vp_results item separately
        vp=vp_results['vp'] if vpa else None,
        vp_square=vp_results['vp_square'] if vpa else None,
        r2s_array=vp_results['r2s_array'] if vpa else None,
        vp_sel_layer_forward=vp_results['vp_sel_layer_forward'] if vpa else None,
        vpr=vp_results['vpr'] if vpa else None,
        vpr_square=vp_results['vpr_square'] if vpa else None,
        r2sr=vp_results['r2sr'] if vpa else None,
        vpr_sel_layer_backward=vp_results['vpr_sel_layer_backward'] if vpa else None
    )
    
    print(f"Detailed results saved to {output_file}")
    return all_rois_df
    
