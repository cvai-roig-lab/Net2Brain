import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from .eval_helper import get_npy_files, get_layers_ncondns
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, ttest_1samp, sem
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


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


def encode_layer(layer_id, batch_size, trn_Idx, tst_Idx, feat_path, avg_across_feat, n_components=100):
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

    Returns:
    - metric_trn (numpy.ndarray): Encoded features of the training set.
    - metric_tst (numpy.ndarray): Encoded features of the test set.
    """
    
    activations = []
    feat_files = glob.glob(feat_path + '/*.np[zy]')
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
            
        if ((jj + 1) % batch_size) == 0:
            pca.partial_fit(np.stack(activations[-batch_size:], axis=0))

    activations = np.stack(activations, axis=0)

    metric_trn = pca.transform(activations)

    # Encode test set
    activations = []
    for ii in tst_Idx:
        feat = np.load(feat_files[ii], allow_pickle=True)
        
        if avg_across_feat:
            activations.append(np.mean(feat[layer_id], axis=1).flatten())
        else:
            activations.append(feat[layer_id].flatten())
        
    activations = np.stack(activations, axis=0)

    metric_tst = pca.transform(activations)
    
    return metric_trn, metric_tst




def train_regression_per_ROI(trn_x,tst_x,trn_y,tst_y):
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
    reg = LinearRegression().fit(trn_x, trn_y)
    y_prd = reg.predict(tst_x)
    correlation_lst = np.zeros(y_prd.shape[1])
    for v in range(y_prd.shape[1]):
        correlation_lst[v] = pearsonr(y_prd[:,v], tst_y[:,v])[0]
    return correlation_lst





def Stacked_Ridge_Encoding(feat_path,
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
    
    
def Stacked_Linear_Encoding(feat_path, 
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
                    save_path="Linear_Encoding_Results", 
                    file_name=None,
                    average_across_layers=False,
                    metric="Linear"):
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
        encoding_metric = _stacked_linear_encoding
    elif metric=="Ridge":
        encoding_metric = _stacked_ridge_encoding
        
    if avg_across_feat == True:
        print("avg_across_feat==True. This averages the activations across axis 1. Only neccessary if different stimuli have a different size of features (as with LLMs)")
        
    print("Since Stacked Regression is computationally very expensive we apply PCA. Adjust n_components to your needs")
    
    
    # Iterate through all folder paths
    for counter, roi_path in enumerate(roi_paths):
        print(f"Processing file {counter}, {roi_path}")
        result_dataframe = encoding_metric(feat_path, 
                                            roi_path, 
                                            model_name, 
                                            trn_tst_split=trn_tst_split, 
                                            n_folds=n_folds, 
                                            n_components=n_components, 
                                            batch_size=batch_size, 
                                            avg_across_feat=avg_across_feat, 
                                            return_correlations=return_correlations,
                                            random_state=random_state)
        
    
        
        
        
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
    
    
def train_linearregression_base(layer_train,layer_predict,test_predict, brain_train, n_components, batch_size):
    """
    Train a base linear regression model for each ROI using Incremental PCA to reduce dimensionality.

    This function performs the following:
    1. Standardizes input DNN activations (features).
    2. Reduces dimensionality using Incremental PCA for efficient processing of large feature matrices.
    3. Trains a Ridge regression model with hyperparameter tuning to predict fMRI responses.
    4. Returns predictions for both Train 2 (level-one feature generation) and test datasets.

    Args:
        layer_train (numpy.ndarray): Training set activations (DNN features) corresponding to Train 1.
        layer_predict (numpy.ndarray): Test set activations (DNN features) corresponding to Train 2.
        test_predict (numpy.ndarray): Test set activations for final evaluation.
        brain_train (numpy.ndarray): fMRI data corresponding to Train 1.
        n_components (int): Number of principal components to retain in Incremental PCA.
        batch_size (int): Batch size for Incremental PCA to process data in smaller chunks.

    Returns:
        base_prd (numpy.ndarray): Predictions for Train 2 data (used as level-one features).
        test_prd (numpy.ndarray): Predictions for the final test data.
    """
    
    # Standardize the features
    scaler = StandardScaler()
    layer_train = scaler.fit_transform(layer_train)
    layer_predict = scaler.transform(layer_predict)
    test_predict = scaler.transform(test_predict)
    
    # Reduce dimensionality with PCA if needed
    pca = PCA(n_components=min(layer_train.shape[0], layer_train.shape[1], n_components))
    layer_train = pca.fit_transform(layer_train)
    layer_predict = pca.transform(layer_predict)
    test_predict = pca.transform(test_predict)
    
    best_model = LinearRegression().fit(layer_train, brain_train)
    best_model.fit(layer_train, brain_train)
    base_prd = best_model.predict(layer_predict)

    # Predict on the test set to tranform it in the right space for the head regression
    test_prd = best_model.predict(test_predict)
    return base_prd, test_prd   
    
    
    
    
def train_linearregression_head(level1_predictions, level1_test_preds,fmri_train2,fmri_test, n_components, batch_size):
    """
    Train the head (stacking) linear regression model to combine predictions from base models
    and compute correlation coefficients for each ROI.

    This function performs the following:
    1. Standardizes level-one predictions (features) generated by base models.
    2. Reduces dimensionality of level-one features using Incremental PCA for efficient processing.
    3. Trains a linear regression model with hyperparameter tuning to predict fMRI responses.
    4. Computes correlations between predicted and actual fMRI responses for each ROI.

    Args:
        level1_predictions (numpy.ndarray): Combined predictions from base models (level-one features) for Train 2.
        level1_test_preds (numpy.ndarray): Combined predictions from base models (level-one features) for the test set.
        fmri_train2 (numpy.ndarray): fMRI data corresponding to Train 2.
        fmri_test (numpy.ndarray): fMRI data corresponding to the test set.
        n_components (int): Number of principal components to retain in Incremental PCA.
        batch_size (int): Batch size for Incremental PCA to process data in smaller chunks.

    Returns:
        correlation_lst (numpy.ndarray): Correlation coefficients for each ROI, representing
                                         the alignment between predicted and actual fMRI responses.
    """

    
    # Standardize the features
    scaler = StandardScaler()
    level1_predictions = scaler.fit_transform(level1_predictions)
    level1_test_preds = scaler.transform(level1_test_preds)
    
    # Reduce dimensionality with PCA if needed
    pca = PCA(n_components=min(level1_predictions.shape[0], level1_predictions.shape[1], n_components))  # Keep max components specified
    level1_predictions = pca.fit_transform(level1_predictions)  # Fit and transform training data
    level1_test_preds = pca.transform(level1_test_preds)  # Transform test data
    
    best_model = LinearRegression().fit(level1_predictions, fmri_train2)
    best_model.fit(level1_predictions, fmri_train2)
    y_prd = best_model.predict(level1_test_preds)
    
    correlation_lst = np.zeros(y_prd.shape[1])
    for v in range(y_prd.shape[1]):
        correlation_lst[v] = pearsonr(y_prd[:,v], fmri_test[:,v])[0]
    return correlation_lst
    

        
def linear_encoding(*args, **kwargs):
    warnings.warn(
        "The 'linear_encoding' function is deprecated and has been replaced by 'Linear_Encoding'. "
        "Please update your code to use the new function name, as this alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    return Linear_Encoding(*args, **kwargs)   




def _stacked_linear_encoding(feat_path, 
                    roi_path, 
                    model_name, 
                    trn_tst_split=0.8, 
                    n_folds=3, 
                    n_components=100, 
                    batch_size=100, 
                    avg_across_feat=False,
                    return_correlations=False,
                    random_state=14):
    """
    Perform stacked linear regression to relate model activations to fMRI data across multiple folds.

    This function implements a stacked regression approach where:
    1. Base models are trained on individual layers of DNN activations to predict fMRI responses.
    2. Predictions from these base models are combined into a "level-one" feature matrix.
    3. A second-level model is trained on the combined predictions to improve prediction accuracy.

    Incremental PCA is used to reduce the dimensionality of DNN activations before regression,
    ensuring efficient computation on large datasets.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in Incremental PCA.
        batch_size (int): Batch size for Incremental PCA to process data in smaller chunks.
        avg_across_feat (bool): Whether to average activations across spatial dimensions before flattening.
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results, including:
            - Correlations for each layer.
            - Statistical significance (p-values).
            - Standard error of the mean (SEM).
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI
                          (only returned if `return_correlations` is True).
    """
    
    # Initialize dictionaries to store results
    fold_dict = {}  # To store fold-wise results
    corr_dict = {}  # To store correlations if requested
    
    # Check if roi_path is a list, if not, make it a list
    roi_file = roi_path 
    roi_name = roi_file.split(os.sep)[-1].split(".")[0]
    
    # Load feature files and get layer information
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)
    
    train_test_split_ratio = trn_tst_split  # 80% for training, 10% for testing
    
    
    
    # Cross-validation loop
    for fold_ii in range(n_folds):
        # Set random seeds for reproducibility
        np.random.seed(fold_ii + random_state)
        random.seed(fold_ii + random_state)
        
        all_layer_preds = []
        all_layer_test_preds = []
        
        # Process each layer of model activations
        for layer_id in tqdm(layer_list, desc=f"Layers in fold {fold_ii}"):
            if fold_ii not in fold_dict.keys():
                fold_dict[fold_ii] = []
                corr_dict[fold_ii] = []
                
                
            # Split data into training and testing sets (90/10)
            trn_Idx, tst_Idx = train_test_split(
                range(num_condns),  # number samples
                test_size=(1 - train_test_split_ratio),  # 10%
                train_size=train_test_split_ratio,  # 90%
                random_state=fold_ii + random_state
            )
            
            # Shuffle training indices before splitting into Train 1 and Train 2
            shuffled_trn_Idx = np.random.permutation(trn_Idx)  # Randomly shuffle the training indices
            train1_size = int(0.67 * len(shuffled_trn_Idx))  # 67% of shuffled training data for Train 1
            train1_Idx, train2_Idx = shuffled_trn_Idx[:train1_size], shuffled_trn_Idx[train1_size:]
                      
            
            # Extract Train 1 and Train 2 data using indices
            layer_train1, layer_train2 = extract_layer_activations(layer_id, train1_Idx, train2_Idx, feat_path, avg_across_feat)
            
            # Extract Test data to stack it for Head Regression
            _, layer_test = extract_layer_activations(layer_id, train2_Idx, tst_Idx, feat_path, avg_across_feat)
            
            # Get fMRI data:
            fmri_data = np.load(os.path.join(roi_file))
            fmri_train1, fmri_train2 = fmri_data[train1_Idx],fmri_data[train2_Idx]
            
            # Train a linear regression model on train1 and predict train 2
            base_pred, test_pred = train_linearregression_base(layer_train1, layer_train2, test_predict=layer_test, brain_train=fmri_train1, n_components=n_components, batch_size=batch_size)
            all_layer_preds.append(base_pred)
            all_layer_test_preds.append(test_pred)
            
        level1_predictions = np.hstack(all_layer_preds)
        level1_test_preds = np.hstack(all_layer_test_preds)
        fmri_test = fmri_data[tst_Idx]
        
        ### Head Layer ###
        layer_train1, layer_test = extract_layer_activations(layer_id, train1_Idx, train2_Idx, feat_path, avg_across_feat)
        r_lst = train_Ridgeregression_head(level1_predictions, level1_test_preds,fmri_train2,fmri_test, n_components=n_components, batch_size=batch_size)
        
        
        if return_correlations:
            corr_dict[fold_ii].append(r_lst)
        
        r = np.mean(r_lst) # Mean of all train test splits
        fold_dict[fold_ii].append(r)
        
    
    
    # Compile all results into a DataFrame for easy analysis
    rows = []
    layer_id = "stacked_layers"

    # If we have more than one fold, collect the R values across folds from fold_dict
    if n_folds > 1:
        
        # find r_values per layer across folds
        r_values_across_folds = [value[0] for value in fold_dict.values()]
        
        # Get average R value across folds
        R = np.mean(r_values_across_folds)
        
        # Perform t-test on the R values across folds
        _, significance = ttest_1samp(r_values_across_folds, 0)

        # Compute the Standard Error of the Mean (SEM)
        sem_value = sem(r_values_across_folds)

    # If there is only one fold, use the r_lst from the fold directly for testing
    else:
        # Get R Value
        R = fold_dict[0][0]
        
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
    
    # Create the DataFrame from the collected rows
    all_rois_df = pd.DataFrame([output_dict], columns=['ROI', 'Layer', 'Model', 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])
    
    if return_correlations:
        return all_rois_df, corr_dict  # Return both the DataFrame and correlation dictionary as-is
    else:
        return all_rois_df, None  # Only return the DataFrame
        
    
 
    
def train_Ridgeregression_base(layer_train, layer_predict, test_predict, brain_train, n_components, batch_size):
    """
    Train a base ridge regression model for each ROI using Incremental PCA to reduce dimensionality.

    This function performs the following:
    1. Standardizes input DNN activations (features).
    2. Reduces dimensionality using Incremental PCA for efficient processing of large feature matrices.
    3. Trains a Ridge regression model with hyperparameter tuning to predict fMRI responses.
    4. Returns predictions for both Train 2 (level-one feature generation) and test datasets.

    Args:
        layer_train (numpy.ndarray): Training set activations (DNN features) corresponding to Train 1.
        layer_predict (numpy.ndarray): Test set activations (DNN features) corresponding to Train 2.
        test_predict (numpy.ndarray): Test set activations for final evaluation.
        brain_train (numpy.ndarray): fMRI data corresponding to Train 1.
        n_components (int): Number of principal components to retain in Incremental PCA.
        batch_size (int): Batch size for Incremental PCA to process data in smaller chunks.

    Returns:
        base_prd (numpy.ndarray): Predictions for Train 2 data (used as level-one features).
        test_prd (numpy.ndarray): Predictions for the final test data.
    """

    
    # Standardize the features
    scaler = StandardScaler()
    layer_train = scaler.fit_transform(layer_train)
    layer_predict = scaler.transform(layer_predict)
    test_predict = scaler.transform(test_predict)
    
    # Reduce dimensionality with PCA if needed
    pca = PCA(n_components=min(layer_train.shape[0], layer_train.shape[1], n_components))
    layer_train = pca.fit_transform(layer_train)
    layer_predict = pca.transform(layer_predict)
    test_predict = pca.transform(test_predict)

    reg = Ridge()
    param_grid = {'alpha': np.logspace(-1, 4, 10)}  
    
    # Define cross-validation strategies
    inner_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
    outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

    # Inner loop: hyperparameter tuning
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=inner_cv, n_jobs=-1)

    X_sample, _, y_sample, _ = train_test_split(layer_train, brain_train, test_size=0.5, random_state=1)

    # Outer loop: model evaluation
    nested_cv_scores = cross_val_score(grid_search, X=X_sample, y=y_sample, cv=outer_cv, n_jobs=-1)
    
    # Train the best model on the full dataset
    grid_search.fit(X_sample, y_sample)
    best_params = grid_search.best_params_

    best_model = Ridge(**best_params)
    best_model.fit(layer_train, brain_train)

    base_prd = best_model.predict(layer_predict)
    
    # Predict on the test set to tranform it in the right space for the head regression
    test_prd = best_model.predict(test_predict)
    return base_prd, test_prd



def train_Ridgeregression_head(level1_predictions, level1_test_preds, fmri_train2, fmri_test, n_components, batch_size):
    """
    Train the head (stacking) Ridge regression model to combine predictions from base models
    and compute correlation coefficients for each ROI.

    This function performs the following:
    1. Standardizes level-one predictions (features) generated by base models.
    2. Reduces dimensionality of level-one features using Incremental PCA for efficient processing.
    3. Trains a Ridge regression model with hyperparameter tuning to predict fMRI responses.
    4. Computes correlations between predicted and actual fMRI responses for each ROI.

    Args:
        level1_predictions (numpy.ndarray): Combined predictions from base models (level-one features) for Train 2.
        level1_test_preds (numpy.ndarray): Combined predictions from base models (level-one features) for the test set.
        fmri_train2 (numpy.ndarray): fMRI data corresponding to Train 2.
        fmri_test (numpy.ndarray): fMRI data corresponding to the test set.
        n_components (int): Number of principal components to retain in Incremental PCA.
        batch_size (int): Batch size for Incremental PCA to process data in smaller chunks.

    Returns:
        correlation_lst (numpy.ndarray): Correlation coefficients for each ROI, representing
                                         the alignment between predicted and actual fMRI responses.
    """

    
    # Standardize the features
    scaler = StandardScaler()
    level1_predictions = scaler.fit_transform(level1_predictions)
    level1_test_preds = scaler.transform(level1_test_preds)
    
    # Reduce dimensionality with PCA if needed
    pca = PCA(n_components=min(level1_predictions.shape[0], level1_predictions.shape[1], n_components))  # Keep max components specified
    level1_predictions = pca.fit_transform(level1_predictions)  # Fit and transform training data
    level1_test_preds = pca.transform(level1_test_preds)  # Transform test data

    reg = Ridge()
    param_grid = {'alpha': np.logspace(-1, 4, 10)}  

    # Define cross-validation strategies
    inner_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
    outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)

    # Inner loop: hyperparameter tuning
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=inner_cv, n_jobs=-1)

    X_sample, _, y_sample, _ = train_test_split(level1_predictions, fmri_train2, test_size=0.5, random_state=1)

    # Outer loop: model evaluation
    nested_cv_scores = cross_val_score(grid_search, X=X_sample, y=y_sample, cv=outer_cv, n_jobs=-1)
    
    # Train the best model on the full dataset
    grid_search.fit(X_sample, y_sample)
    best_params = grid_search.best_params_

    best_model = Ridge(**best_params)
    best_model.fit(level1_predictions, fmri_train2)

    y_prd = best_model.predict(level1_test_preds)
    
    correlation_lst = np.zeros(y_prd.shape[1])
    for v in range(y_prd.shape[1]):
        correlation_lst[v] = pearsonr(y_prd[:,v], fmri_test[:,v])[0]
    return correlation_lst
    




def extract_layer_activations(layer_id, trn_Idx, tst_Idx, feat_path, avg_across_feat):
    """
    Extracts the layer activations.

    Parameters:
    - layer_id (str): The layer name whose activations are to be encoded.
    - trn_Idx (list of int): Indices of the training set files.
    - tst_Idx (list of int): Indices of the test set files.
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - trn (numpy.ndarray): features of the training set.
    - tst (numpy.ndarray): features of the test set.
    """
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()  # Ensure consistent order
    
    if avg_across_feat:
        trn = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in trn_Idx])
        tst = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in tst_Idx])
    else:
        trn = np.array([np.load(feat_files[ii], allow_pickle=True)[layer_id].flatten() for ii in trn_Idx])
        tst = np.array([np.load(feat_files[ii], allow_pickle=True)[layer_id].flatten() for ii in tst_Idx])
    return trn, tst


    
def _stacked_ridge_encoding(feat_path, 
                    roi_path, 
                    model_name, 
                    trn_tst_split=0.8, 
                    n_folds=3, 
                    n_components=100, 
                    batch_size=100, 
                    avg_across_feat=False,
                    return_correlations=False,
                    random_state=14):
    """
    Perform stacked ridge regression to relate model activations to fMRI data across multiple folds.

    This function implements a stacked regression approach where:
    1. Base models are trained on individual layers of DNN activations to predict fMRI responses.
    2. Predictions from these base models are combined into a "level-one" feature matrix.
    3. A second-level model is trained on the combined predictions to improve prediction accuracy.

    Incremental PCA is used to reduce the dimensionality of DNN activations before regression,
    ensuring efficient computation on large datasets.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in Incremental PCA.
        batch_size (int): Batch size for Incremental PCA to process data in smaller chunks.
        avg_across_feat (bool): Whether to average activations across spatial dimensions before flattening.
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results, including:
            - Correlations for each layer.
            - Statistical significance (p-values).
            - Standard error of the mean (SEM).
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI
                          (only returned if `return_correlations` is True).
    """
    
    # Initialize dictionaries to store results
    fold_dict = {}  # To store fold-wise results
    corr_dict = {}  # To store correlations if requested
    
    # Check if roi_path is a list, if not, make it a list
    roi_file = roi_path 
    roi_name = roi_file.split(os.sep)[-1].split(".")[0]
    
    # Load feature files and get layer information
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)
    
    train_test_split_ratio = trn_tst_split  # 80% for training, 10% for testing
    
    # Cross-validation loop
    for fold_ii in range(n_folds):
        # Set random seeds for reproducibility
        np.random.seed(fold_ii + random_state)
        random.seed(fold_ii + random_state)
        
        all_layer_preds = []
        all_layer_test_preds = []
        
        # Process each layer of model activations
        for layer_id in tqdm(layer_list, desc=f"Layers in fold {fold_ii}"):
            if fold_ii not in fold_dict.keys():
                fold_dict[fold_ii] = []
                corr_dict[fold_ii] = []
                
                
            # Split data into training and testing sets (90/10)
            trn_Idx, tst_Idx = train_test_split(
                range(num_condns),  # number samples
                test_size=(1 - train_test_split_ratio),  # 10%
                train_size=train_test_split_ratio,  # 90%
                random_state=fold_ii + random_state
            )
            
            # Shuffle training indices before splitting into Train 1 and Train 2
            shuffled_trn_Idx = np.random.permutation(trn_Idx)  # Randomly shuffle the training indices
            train1_size = int(0.67 * len(shuffled_trn_Idx))  # 67% of shuffled training data for Train 1
            train1_Idx, train2_Idx = shuffled_trn_Idx[:train1_size], shuffled_trn_Idx[train1_size:]
                      
            
            # Extract Train 1 and Train 2 data using indices
            layer_train1, layer_train2 = extract_layer_activations(layer_id, train1_Idx, train2_Idx, feat_path, avg_across_feat)
            
            # Extract Test data to stack it for Head Regression
            _, layer_test = extract_layer_activations(layer_id, train2_Idx, tst_Idx, feat_path, avg_across_feat)
            
            # Get fMRI data:
            fmri_data = np.load(os.path.join(roi_file))
            fmri_train1, fmri_train2 = fmri_data[train1_Idx],fmri_data[train2_Idx]
            
            # Train a linear regression model on train1 and predict train 2
            base_pred, test_pred = train_Ridgeregression_base(layer_train1, layer_train2, test_predict=layer_test, brain_train=fmri_train1, n_components=n_components, batch_size=batch_size)
            all_layer_preds.append(base_pred)
            all_layer_test_preds.append(test_pred)
            
        level1_predictions = np.hstack(all_layer_preds)
        level1_test_preds = np.hstack(all_layer_test_preds)
        fmri_test = fmri_data[tst_Idx]
        
        ### Head Layer ###
        layer_train1, layer_test = extract_layer_activations(layer_id, train1_Idx, train2_Idx, feat_path, avg_across_feat)
        r_lst = train_Ridgeregression_head(level1_predictions, level1_test_preds,fmri_train2,fmri_test, n_components=n_components, batch_size=batch_size)
        
        
        if return_correlations:
            corr_dict[fold_ii].append(r_lst)
        
        r = np.mean(r_lst) # Mean of all train test splits
        fold_dict[fold_ii].append(r)
        
    
    # Compile all results into a DataFrame for easy analysis
    rows = []
    layer_id = "stacked_layers"

    # If we have more than one fold, collect the R values across folds from fold_dict
    if n_folds > 1:
        
        # find r_values per layer across folds
        r_values_across_folds = [value[0] for value in fold_dict.values()]
        
        # Get average R value across folds
        R = np.mean(r_values_across_folds)
        
        # Perform t-test on the R values across folds
        _, significance = ttest_1samp(r_values_across_folds, 0)

        # Compute the Standard Error of the Mean (SEM)
        sem_value = sem(r_values_across_folds)

    # If there is only one fold, use the r_lst from the fold directly for testing
    else:
        # Get R Value
        R = fold_dict[0][0]
        
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
    
    # Create the DataFrame from the collected rows
    all_rois_df = pd.DataFrame([output_dict], columns=['ROI', 'Layer', 'Model', 'R', '%R2', 'Significance', 'SEM', 'LNC', 'UNC'])
    
    if return_correlations:
        return all_rois_df, corr_dict  # Return both the DataFrame and correlation dictionary as-is
    else:
        return all_rois_df, None  # Only return the DataFrame