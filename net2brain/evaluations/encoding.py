import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, ttest_1samp, sem
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from net2brain.evaluations.eval_helper import sq


def get_npy_files(input_path):
    """
    Returns a list of .npy files from the given input, which can be a single file,
    a list of files, or a folder. If the input is a folder, it retrieves all .npy
    files in that folder. If the input is a list of files or folders, it filters out
    the folders and returns only .npy files. A warning is raised if folders are present
    in the input list.

    Parameters:
    -----------
    input_path : str or list
        A single file path (str), a list of file/folder paths (list), or a folder path (str).

    Returns:
    --------
    list
        A list of .npy file paths.
    """
    # Convert single string input to list for consistent processing
    if isinstance(input_path, str):
        input_path = [input_path]

    # Separate files and folders
    files = [f for f in input_path if os.path.isfile(f)]
    folders = [f for f in input_path if os.path.isdir(f)]

    # Raise warning if there are folders in the input list
    if folders:
        warnings.warn("Ignoring folders in the list.")

    # If input contains a folder, add its .npy files to the list
    for folder in folders:
        files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')])

    # Filter only .npy files
    npy_files = [f for f in files if f.endswith('.npy')]

    if not npy_files:
        raise ValueError("No valid .npy files found.")

    return npy_files


def raw2rdm(raw, dim=None):
    if dim is not None:
        raw = (raw - np.mean(raw, axis=dim, keepdims=True)) / (np.std(raw, axis=dim, keepdims=True) + 1e-7)
    rdm = 1 - np.corrcoef(raw)
    return rdm


def average_df_across_layers(dataframes):
    """Function to average correlation values across layers and recalculate significance"""

    # Concatenate all DataFrames together for averaging across the 'Layer' dimension
    combined_df = pd.concat(dataframes)

    # Ensure the 'Layer' column is treated as a categorical type with original order
    combined_df['Layer'] = pd.Categorical(combined_df['Layer'], categories=combined_df['Layer'].unique(), ordered=True)

    # Group by 'Layer' and calculate the mean for the 'R' and '%R2' columns
    averaged_df = combined_df.groupby('Layer', sort=False).agg({
        'R': 'mean',
        '%R2': 'mean'
    }).reset_index()

    # Recalculate significance across all 'R' values in the combined DataFrame
    layer_significance = combined_df.groupby('Layer', sort=False)['R'].apply(lambda x: ttest_1samp(x, 0)[1])
    layer_sem = combined_df.groupby('Layer', sort=False)['R'].apply(sem)

    # Merge significance and SEM back into the averaged DataFrame
    averaged_df['Significance'] = layer_significance.values
    averaged_df['SEM'] = layer_sem.values

    # Add placeholder values for 'LNC' and 'UNC'
    averaged_df['LNC'] = np.nan
    averaged_df['UNC'] = np.nan

    return averaged_df



def get_layers_ncondns(feat_path):
    """
    Extracts information about the number of layers, the list of layer names, and the number of conditions (images)
    from the npz files in the specified feature path.

    Parameters:
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - num_layers (int): The number of layers found in the npz files.
    - layer_list (list of str): A list containing the names of the layers.
    - num_conds (int): The number of conditions (images) based on the number of npz files in the directory.
    """
    
    # Find all npz files in the specified directory
    activations = glob.glob(feat_path + "/*.npz")
    
    # Count the number of npz files as the number of conditions (images)
    num_condns = len(activations)
    
    # Load the first npz file to extract layer information
    feat = np.load(activations[0], allow_pickle=True)

    num_layers = 0
    layer_list = []

    # Iterate through the keys in the npz file, ignoring metadata keys
    for key in feat:
        if "__" in key:  # key: __header__, __version__, __globals__
            continue
        else:
            num_layers += 1
            layer_list.append(key)  # collect all layer names ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    return num_layers, layer_list, num_condns


def encode_layer(layer_id, batch_size, trn_Idx, tst_Idx, feat_path, avg_across_feat, n_components=100,
                 mem_mode='performance'):
    """
    Encodes the layer activations using IncrementalPCA or Ridge Regression, for both training and test sets.

    Parameters:
    - layer_id (str): The layer name whose activations are to be encoded.
    - metric (str): Encoding method ('pca' or 'ridge').
    - batch_size (int): Batch size for IncrementalPCA.
    - trn_Idx (list of int): Indices of the training set files.
    - tst_Idx (list of int): Indices of the test set files.
    - feat_path (str): Path to the directory containing npz files with model features.
    - avg_across_feat (bool): Whether to average across features.
    - n_components (int): Number of components for PCA.
    - mem_mode (str): 'saver' or 'performance'; Choose 'saver' if you have large features or small RAM,
        otherwise leave 'performance' as default.
    Returns:
    - metric_trn (numpy.ndarray): Encoded features of the training set.
    - metric_tst (numpy.ndarray): Encoded features of the test set.
    """

    activations = []
    feat_files = glob.glob(feat_path+'/*.npz')
    feat_files.sort()

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Train encoding
    for jj, ii in enumerate(trn_Idx):
        feat = np.load(feat_files[ii], allow_pickle=True)  # get activations of the current layer

        if avg_across_feat:
            new_activation = np.mean(feat[layer_id], axis=1).flatten()
        else:
            new_activation = feat[layer_id].flatten()
        
        if activations and new_activation.shape != activations[-1].shape:
            raise ValueError("Elements in activations do not have the same shape. "
                             "Please set 'avg_across_feat' to True to average across features.")

        activations.append(new_activation)  # collect in a list

        if ((jj + 1) % batch_size) == 0 or (jj + 1) == len(trn_Idx):
            # last batch might not be the same size
            effective_batch_size = batch_size if jj != len(trn_Idx) - 1 else len(trn_Idx) % batch_size
            pca.partial_fit(np.stack(activations[-effective_batch_size:], axis=0))
            if mem_mode == 'saver':
                # in saver mode, only fit and don't save activations in memory
                del activations
                activations = []

    if mem_mode == 'saver':
        transformed_activations = []
        for ii in trn_Idx:
            feat = np.load(feat_files[ii], allow_pickle=True)

            if avg_across_feat:
                new_activation = np.mean(feat[layer_id], axis=1).flatten()
            else:
                new_activation = feat[layer_id].flatten()

            # transform one at a time to only have a lightweight list in memory
            transformed_activations.append(pca.transform(new_activation.reshape(1, -1)))

        metric_trn = np.concatenate(transformed_activations, axis=0)
    else:
        activations = np.stack(activations, axis=0)
        metric_trn = pca.transform(activations)

    # Encode test set
    transformed_activations = []
    for ii in tst_Idx:
        feat = np.load(feat_files[ii], allow_pickle=True)

        if avg_across_feat:
            new_activation = np.mean(feat[layer_id], axis=1).flatten()
        else:
            new_activation = feat[layer_id].flatten()

        # transform one at a time to only have a lightweight list in memory
        transformed_activations.append(pca.transform(new_activation.reshape(1, -1)))

    metric_tst = np.concatenate(transformed_activations, axis=0)

    return metric_trn, metric_tst




def train_regression_per_ROI(trn_x,tst_x,trn_y,tst_y, roi_name, save_path, model_name, layer_id, veRSA=False):
    """
    Train a linear regression model for each ROI and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.

    Returns:
        correlation_lst (numpy.ndarray): List of correlation coefficients for each ROI.
    """
    if not os.path.exists(f"{save_path}/{model_name}/{layer_id}/{roi_name}.npy"):
        if trn_x is None:
            raise ValueError("Not all ROI regressions were computed on previous run - PCA needs to be re-run.")
        reg = LinearRegression().fit(trn_x, trn_y)
        y_prd = reg.predict(tst_x)
        if not os.path.exists(f"{save_path}/{model_name}/{layer_id}"):
            os.makedirs(f"{save_path}/{model_name}/{layer_id}")
        np.save(f"{save_path}/{model_name}/{layer_id}/{roi_name}.npy", y_prd)
    else:
        y_prd = np.load(f"{save_path}/{model_name}/{layer_id}/{roi_name}.npy")
    if not veRSA:
        correlation_lst = np.zeros(y_prd.shape[1])
        for v in range(y_prd.shape[1]):
            correlation_lst[v] = pearsonr(y_prd[:,v], tst_y[:,v])[0]
        return correlation_lst
    else:
        prd_rdm = raw2rdm(y_prd)
        brain_rdm = raw2rdm(tst_y)
        corr = spearmanr(sq(brain_rdm), sq(prd_rdm))[0]
        r = np.mean(corr)
        return r


def Ridge_Encoding(feat_path,
                   roi_path,
                   model_name,
                   trn_tst_split=0.8,
                   n_folds=3,
                   n_components=100,
                   batch_size=100,
                   avg_across_feat=False,
                   return_correlations = False,
                   random_state=42,
                   save_path="Linear_Encoding_Results",
                   file_name=None,
                   average_across_layers=False):

    result = Encoding(feat_path,
             roi_path,
             model_name,
             trn_tst_split=trn_tst_split,
             n_folds=n_folds,
             n_components=n_components,
             batch_size=batch_size,
             avg_across_feat=avg_across_feat,
             return_correlations = return_correlations,
             random_state=random_state,
             save_path=save_path,
             file_name=file_name,
             average_across_layers=average_across_layers,
             metric="Ridge")

    return result


def Linear_Encoding(feat_path,
                    roi_path,
                    model_name,
                    trn_tst_split=0.8,
                    n_folds=3,
                    n_components=100,
                    batch_size=100,
                    avg_across_feat=False,
                    return_correlations = False,
                    random_state=42,
                    save_path="Linear_Encoding_Results",
                    file_name=None,
                    average_across_layers=False):

    result = Encoding(feat_path,
             roi_path,
             model_name,
             trn_tst_split=trn_tst_split,
             n_folds=n_folds,
             n_components=n_components,
             batch_size=batch_size,
             avg_across_feat=avg_across_feat,
             return_correlations = return_correlations,
             random_state=random_state,
             save_path=save_path,
             file_name=file_name,
             average_across_layers=average_across_layers,
             metric="Linear")

    return result





def Encoding(feat_path,
                    roi_path,
                    model_name,
                    trn_tst_split=0.8,
                    n_folds=3,
                    n_components=100,
                    batch_size=100,
                    avg_across_feat=False,
                    return_correlations = False,
                    random_state=42,
                    shuffle=True,
                    save_path="Linear_Encoding_Results",
                    file_name=None,
                    average_across_layers=False,
                    metric="Linear",
                    veRSA=False):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
        
            If we have a list of folders, each folders content will be summarized into one value. This is important if one folder contains data for the same ROI for different subjects
            
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        avg_across_feat (bool): If True it averages the activations across axis 1. Neccessary if different stimuli have a different size of features
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Turn the roi_path into a list of files
    roi_paths = get_npy_files(roi_path)

    list_result_dataframes = []
    list_correlations = []
    all_results_df = pd.DataFrame()
    
    # Which encoding metric are we using?
    if metric=="Linear":
        encoding_metric = _linear_encoding
    elif metric=="Ridge":
        encoding_metric = _ridge_encoding

    if avg_across_feat == True:
        print("avg_across_feat==True. This averages the activations across axis 1. Only neccessary if different stimuli have a different size of features (as with LLMs)")

    
    # Iterate through all folder paths
    for counter, roi_path in enumerate(roi_paths):
        print(f"Processing file {counter}, {roi_path}")
        result_dataframe = encoding_metric(feat_path,
                                            roi_path, 
                                            model_name, 
                                            trn_tst_split=trn_tst_split, 
                                            n_folds=n_folds,
                                            random_state=random_state,
                                            shuffle=shuffle,
                                            n_components=n_components, 
                                            batch_size=batch_size, 
                                            avg_across_feat=avg_across_feat,
                                            return_correlations=return_correlations,
                                            save_path=save_path,
                                            veRSA=veRSA)

        if average_across_layers:
            list_result_dataframes.append(result_dataframe[0])  # Collect DataFrames for averaging later
        else:
            all_results_df = pd.concat([all_results_df, result_dataframe[0]], ignore_index=True)  # Append DataFrames

        if return_correlations:
            list_correlations.append(result_dataframe[1])

    if average_across_layers:
        if len(list_result_dataframes) > 1:
            warnings.warn("Code will now average the layer values across all given brain data with average_across_layers=True")
            all_results_df = average_df_across_layers(list_result_dataframes)
        else:
            warnings.warn("Only one DataFrame available. Averaging across layers is not possible. Returning the single DataFrame.")
            all_results_df = pd.concat(list_result_dataframes, ignore_index=True)  # Convert the single DataFrame to a Pandas DataFrame

    if not veRSA:
        # Create the output folder if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Determine the file name
        if file_name is None:
            csv_file_path = f"{save_path}/{model_name}.csv"
            dataframe_path = f"{save_path}/{model_name}.npy"
            correlations_file_path = f"{save_path}/{model_name}_correlations.npy"
        else:
            csv_file_path = f"{save_path}/{file_name}.csv"
            dataframe_path = f"{save_path}/{file_name}.npy"
            correlations_file_path = f"{save_path}/{file_name}_correlations.npy"

        # Save the DataFrame as a CSV file
        all_results_df.to_csv(csv_file_path, index=False)
        np.save(dataframe_path, all_results_df)

        # If return_correlations is True, save the correlations dictionary as .npy
        if return_correlations:
            np.save(correlations_file_path, list_correlations)  # Save the list of correlations as .npy

    return all_results_df
        
        
        
        
def linear_encoding(*args, **kwargs):
    warnings.warn(
        "The 'linear_encoding' function is deprecated and has been replaced by 'Linear_Encoding'. "
        "Please update your code to use the new function name, as this alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    return Linear_Encoding(*args, **kwargs)   
        
    

def _linear_encoding(feat_path,
                     roi_path,
                     model_name,
                     trn_tst_split=0.8,
                     n_folds=3,
                     n_components=100,
                     batch_size=100,
                     avg_across_feat=False,
                     return_correlations=False,
                     random_state=42,
                     shuffle=True,
                     save_path="Linear_Encoding_Results",
                     veRSA=False):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """

    # Initialize dictionaries to store results
    fold_dict = {}  # To store fold-wise results
    corr_dict = {}  # To store correlations if requested

    # Check if roi_path is a list, if not, make it a list
    roi_file = roi_path
    roi_name = roi_file.split(os.sep)[-1].split(".")[0]

    # Load feature files and get layer information
    feat_files = glob.glob(feat_path+'/*.npz')
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)

    # Loop over each fold for cross-validation
    for fold_ii in range(n_folds):

        # Set random seeds for reproducibility
        np.random.seed(fold_ii+random_state)
        random.seed(fold_ii+random_state)

        # Split the data indices into training and testing sets
        trn_Idx,tst_Idx = train_test_split(range(len(feat_files)), train_size=trn_tst_split,
                                           random_state=fold_ii+random_state, shuffle=shuffle)

        # Process each layer of model activations
        for layer_id in tqdm(layer_list, desc=f"Layers in fold {fold_ii}"):
            if fold_ii not in fold_dict.keys():
                fold_dict[fold_ii] = {}
                corr_dict[fold_ii] = {}

            # Encode the current layer using PCA and split into training and testing sets
            if not os.path.exists(f"{save_path}/{model_name}/{layer_id}"):
                pca_trn,pca_tst = encode_layer(layer_id, batch_size, trn_Idx, tst_Idx, feat_path, avg_across_feat, n_components)
            else:
                pca_trn = None
                pca_tst = None

            fold_dict[fold_ii][layer_id] = []
            corr_dict[fold_ii][layer_id] = []

            # Load fMRI data for the current ROI and split into training and testing sets
            fmri_data = np.load(os.path.join(roi_file))
            fmri_trn,fmri_tst = fmri_data[trn_Idx],fmri_data[tst_Idx]

            if not veRSA:
                # Train a linear regression model and compute correlations for the current ROI
                r_lst = train_regression_per_ROI(pca_trn,pca_tst,fmri_trn,fmri_tst, roi_name, save_path,
                                                 model_name, layer_id)
                r = np.mean(r_lst) # Mean of all train test splits

                # Store correlation results
                if return_correlations:
                    corr_dict[layer_id][roi_name].append(r_lst)
            else:
                r = train_regression_per_ROI(pca_trn, pca_tst, fmri_trn, fmri_tst, roi_name, save_path,
                                             model_name, layer_id, veRSA=True)
            fold_dict[layer_id][roi_name].append(r)


    # Compile all results into a DataFrame for easy analysis
    layers = list(fold_dict[0].keys())  # Get the layer list from the first fold
    rows = []

    # Iterate through each layer
    for layer_id in layers:

        # If we have more than one fold, collect the R values across folds from fold_dict
        if n_folds > 1:

            # find r_values per layer across folds
            r_values_across_folds = [fold_dict[fold_ii][layer_id][0] for fold_ii in range(n_folds)]

            # Get average R value across folds
            R = np.mean(r_values_across_folds)

            # Perform t-test on the R values across folds
            _, significance = ttest_1samp(r_values_across_folds, 0)

            # Compute the Standard Error of the Mean (SEM)
            sem_value = sem(r_values_across_folds)

        # If there is only one fold, use the r_lst from the fold directly for testing
        else:
            # Get R Value
            R = fold_dict[0][layer_id][0]

            # Perform t-test on the r_lst values since they are also correlation values
            _, significance = ttest_1samp(r_lst, 0)

            # Compute the Standard Error of the Mean (SEM)
            sem_value = sem(r_lst)

        # Construct the row dictionary for the DataFrame
        output_dict = {
            "ROI": roi_name,
            "Layer": layer_id,
            "Model": model_name,
            "R": R,
            "%R2": np.nan,
            "Significance": significance,
            "SEM": sem_value,
            "LNC": np.nan,
            "UNC": np.nan
        }

        # Append the row to the rows list
        rows.append(output_dict)

    # Create the DataFrame from the collected rows
    all_rois_df = pd.DataFrame(rows, columns=['ROI', 'Layer', 'Model', 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])

    if return_correlations:
        return all_rois_df, corr_dict  # Return both the DataFrame and correlation dictionary as-is
    else:
        return all_rois_df, None  # Only return the DataFrame






def train_Ridgeregression_per_ROI(trn_x,tst_x,trn_y,tst_y, roi_name, save_path, model_name, layer_id, veRSA=False):
    """
    Train a linear regression model for each ROI and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.

    Returns:
        correlation_lst (numpy.ndarray): List of correlation coefficients for each ROI.
    """
    if not os.path.exists(f"{save_path}/{model_name}/{layer_id}/{roi_name}.npy"):
        if trn_x is None:
            raise ValueError("Not all ROI regressions were computed on previous run - PCA needs to be re-run.")
        # Standardize the features
        scaler = StandardScaler()
        trn_x = scaler.fit_transform(trn_x)
        tst_x = scaler.transform(tst_x)

        reg = Ridge()
        param_grid = {'alpha': np.logspace(-2, 2, 5)}

        # Define cross-validation strategies
        inner_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
        outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

        # Inner loop: hyperparameter tuning
        grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=inner_cv, n_jobs=-1)

        X_sample, _, y_sample, _ = train_test_split(trn_x, trn_y, test_size=0.5, random_state=1)

        # Outer loop: model evaluation
        nested_cv_scores = cross_val_score(grid_search, X=X_sample, y=y_sample, cv=outer_cv, n_jobs=-1)

        # # Results
        # print(f"Nested CV Mean R^2: {nested_cv_scores.mean():.3f} ± {nested_cv_scores.std():.3f}")


        # print('for training:', trn_x.shape)
        # print('for training:', trn_y.shape)

        # Train the best model on the full dataset
        grid_search.fit(X_sample, y_sample)
        best_params = grid_search.best_params_

        best_model = Ridge(**best_params)
        best_model.fit(trn_x, trn_y)

        y_prd = best_model.predict(tst_x)

        if not os.path.exists(f"{save_path}/{model_name}/{layer_id}"):
            os.makedirs(f"{save_path}/{model_name}/{layer_id}")
        np.save(f"{save_path}/{model_name}/{layer_id}/{roi_name}.npy", y_prd)
    else:
        y_prd = np.load(f"{save_path}/{model_name}/{layer_id}/{roi_name}.npy")

    if not veRSA:
        correlation_lst = np.zeros(y_prd.shape[1])
        for v in range(y_prd.shape[1]):
            correlation_lst[v] = pearsonr(y_prd[:,v], tst_y[:,v])[0]
        return correlation_lst
    else:
        prd_rdm = raw2rdm(y_prd)
        brain_rdm = raw2rdm(tst_y)
        corr = spearmanr(sq(brain_rdm), sq(prd_rdm))[0]
        r = np.mean(corr)
        return r


def encode_layer_ridge(layer_id, trn_Idx, tst_Idx, feat_path, avg_across_feat):
    """
    Encodes the layer activations using IncrementalPCA, for both training and test sets.

    Parameters:
    - layer_id (str): The layer name whose activations are to be encoded.
    - trn_Idx (list of int): Indices of the training set files.
    - tst_Idx (list of int): Indices of the test set files.
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - trn (numpy.ndarray): features of the training set.
    - tst (numpy.ndarray): features of the test set.
    """
    feat_files = glob.glob(feat_path + '/*.npz')
    feat_files.sort()  # Ensure consistent order

    trn = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in trn_Idx])
    tst = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in tst_Idx])
    return trn, tst



def _ridge_encoding(feat_path,
                    roi_path,
                    model_name,
                    trn_tst_split=0.8,
                    n_folds=3,
                    n_components=100,
                    batch_size=100,
                    avg_across_feat=False,
                    return_correlations=False,
                    random_state=14,
                    shuffle=True,
                    save_path="Ridge_Encoding_Results",
                    veRSA=False):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Initialize dictionaries to store results
    fold_dict = {}  # To store fold-wise results
    corr_dict = {}  # To store correlations if requested
    
    # Check if roi_path is a list, if not, make it a list
    roi_file = roi_path
    roi_name = roi_file.split(os.sep)[-1].split(".")[0]
    
    # Load feature files and get layer information
    feat_files = glob.glob(feat_path+'/*.npz')
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)

    
    # Loop over each fold for cross-validation
    for fold_ii in range(n_folds):
        
        # Set random seeds for reproducibility
        np.random.seed(fold_ii+random_state)
        random.seed(fold_ii+random_state)
        
        # Split the data indices into training and testing sets
        trn_Idx,tst_Idx = train_test_split(range(len(feat_files)), train_size=trn_tst_split,
                                           random_state=fold_ii+random_state, shuffle=shuffle)

        # Process each layer of model activations
        for layer_id in tqdm(layer_list, desc=f"Layers in fold {fold_ii}"):
            if fold_ii not in fold_dict.keys():
                fold_dict[fold_ii] = {}
                corr_dict[fold_ii] = {}
            
            # Encode the current layer using PCA and split into training and testing sets
            if not os.path.exists(f"{save_path}/{model_name}/{layer_id}"):
                pca_trn,pca_tst = encode_layer_ridge(layer_id, trn_Idx, tst_Idx, feat_path, avg_across_feat)
            else:
                pca_trn = None
                pca_tst = None

            fold_dict[fold_ii][layer_id] = []
            corr_dict[fold_ii][layer_id] = []

            # Load fMRI data for the current ROI and split into training and testing sets
            fmri_data = np.load(os.path.join(roi_file))
            fmri_trn,fmri_tst = fmri_data[trn_Idx],fmri_data[tst_Idx]

            if not veRSA:
                # Train a regression model and compute correlations for the current ROI
                r_lst = train_Ridgeregression_per_ROI(pca_trn,pca_tst,fmri_trn,fmri_tst, roi_name, save_path,
                                                      model_name, layer_id, veRSA)
                r = np.mean(r_lst) # Mean of all train test splits

                # Store correlation results
                if return_correlations:
                    corr_dict[fold_ii][layer_id].append(r_lst)
            else:
                r = train_regression_per_ROI(pca_trn, pca_tst, fmri_trn, fmri_tst, roi_name, save_path,
                                             model_name, layer_id, veRSA=True)
            fold_dict[fold_ii][layer_id].append(r)
                
    # Compile all results into a DataFrame for easy analysis
    layers = list(fold_dict[0].keys())  # Get the layer list from the first fold
    rows = []

    # Iterate through each layer
    for layer_id in layers:

        # If we have more than one fold, collect the R values across folds from fold_dict
        if n_folds > 1:
            
            # find r_values per layer across folds
            r_values_across_folds = [fold_dict[fold_ii][layer_id][0] for fold_ii in range(n_folds)]

            # Get average R value across folds
            R = np.mean(r_values_across_folds)

            # Perform t-test on the R values across folds
            _, significance = ttest_1samp(r_values_across_folds, 0)

            # Compute the Standard Error of the Mean (SEM)
            sem_value = sem(r_values_across_folds)

        # If there is only one fold, use the r_lst from the fold directly for testing
        else:
            # Get R Value
            R = fold_dict[0][layer_id][0]

            # Perform t-test on the r_lst values since they are also correlation values
            _, significance = ttest_1samp(r_lst, 0)

            # Compute the Standard Error of the Mean (SEM)
            sem_value = sem(r_lst)

        # Construct the row dictionary for the DataFrame
        output_dict = {
            "ROI": roi_name,
            "Layer": layer_id,
            "Model": model_name,
            "R": R,
            "%R2": np.nan,
            "Significance": significance,
            "SEM": sem_value,
            "LNC": np.nan,
            "UNC": np.nan
        }

        # Append the row to the rows list
        rows.append(output_dict)

    # Create the DataFrame from the collected rows
    all_rois_df = pd.DataFrame(rows, columns=['ROI', 'Layer', 'Model', 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])

    if return_correlations:
        return all_rois_df, corr_dict  # Return both the DataFrame and correlation dictionary as-is
    else:
        return all_rois_df, None  # Only return the DataFrame