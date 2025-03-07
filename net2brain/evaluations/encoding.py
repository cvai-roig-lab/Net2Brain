import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.stats import pearsonr, spearmanr, ttest_1samp, sem
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from net2brain.evaluations.eval_helper import sq, get_npy_files, get_layers_ncondns


def average_df_across_layers(combined_df):
    """Function to average correlation values across layers and recalculate significance"""

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


def encode_layer(trn_Idx, tst_Idx, feat_path, layer_id, avg_across_feat, batch_size, n_components,
                 srp_before_pca, srp_on_subset, mem_mode, save_pca, save_path):
    """
    Encodes the layer activations using IncrementalPCA for both training and test sets.

    Args:
        trn_Idx (list of int): Indices of the training set files.
        tst_Idx (list of int): Indices of the test set files.
        feat_path (str): Path to the directory containing npz files with model features.
        layer_id (str): The layer name whose activations are to be encoded.
        avg_across_feat (bool): Whether to average across features.
        batch_size (int): Batch size for IncrementalPCA.
        n_components (int): Number of components for PCA.
        srp_before_pca (bool): Whether to apply Sparse Random Projection (SRP) before PCA. Use when features are so
            high-dimensional that IncrementalPCA runs out of memory after some batches. Num of dims estimated by SRP.
        srp_on_subset (int or None): Number of samples to use for SRP fitting. If None, all samples are used,
            which is recommended if you have enough memory (if `srp_before_pca` is False it has no effect).
        mem_mode (str): 'saver' or 'performance'; Choose 'saver' if you don't have enough memory to store all
            training sample features, otherwise leave 'performance' as default. If you have `srp_before_pca` enabled,
            in the first case you will also need to restrict the number of samples for SRP fitting with `srp_on_subset`.
        save_pca (bool): Whether to save the PCA transform to disk.
        save_path (str or None): The path to save the PCA transform in (if `save_pca` is True).

    Returns:
        PCA-transformed training and test set features (tuple of numpy.ndarray).
    """

    activations = []
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()

    if save_path is None or not os.path.exists(save_path):

        if srp_before_pca:
            all_data_for_estim = []
            srp_trn = trn_Idx if srp_on_subset is None else trn_Idx[:srp_on_subset]
            for jj, ii in enumerate(srp_trn):
                feat = np.load(feat_files[ii], allow_pickle=True)  # get activations of the current layer
                if avg_across_feat:
                    new_activation = np.mean(feat[layer_id], axis=1).flatten()
                else:
                    new_activation = feat[layer_id].flatten()
                if all_data_for_estim and new_activation.shape != all_data_for_estim[-1].shape:
                    raise ValueError("Elements in activations do not have the same shape. "
                                     "Please set 'avg_across_feat' to True to average across features.")
                all_data_for_estim.append(new_activation)  # collect in a list
            target_dim = min(all_data_for_estim[0].shape[0], johnson_lindenstrauss_min_dim(len(all_data_for_estim)))
            srp = SparseRandomProjection(n_components=target_dim)
            srp.fit(np.stack(all_data_for_estim, axis=0))
            if save_pca:
                with open(save_path.split('pca.pkl')[0]+'srp.pkl', "wb") as f:
                    pickle.dump(srp, f)
            del all_data_for_estim

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
                # second condition for the case of the last batch not being the same size
                effective_batch_size = batch_size if jj != len(trn_Idx) - 1 else len(trn_Idx) % batch_size
                if srp_before_pca:
                    pca.partial_fit(srp.transform(np.stack(activations[-effective_batch_size:], axis=0)))
                else:
                    pca.partial_fit(np.stack(activations[-effective_batch_size:], axis=0))
                if mem_mode == 'saver':
                    # in saver mode, only fit and don't save activations in memory
                    del activations
                    activations = []

        if save_pca:
            with open(save_path, "wb") as f:
                pickle.dump(pca, f)
    else:
        with open(save_path, "rb") as f:
            pca = pickle.load(f)
        if srp_before_pca:
            with open(save_path.split('pca.pkl')[0] + 'srp.pkl', "rb") as f:
                srp = pickle.load(f)

    if mem_mode == 'saver' or (save_path is not None and os.path.exists(save_path)):
        transformed_activations = []
        for ii in trn_Idx:
            feat = np.load(feat_files[ii], allow_pickle=True)

            if avg_across_feat:
                new_activation = np.mean(feat[layer_id], axis=1).flatten()
            else:
                new_activation = feat[layer_id].flatten()

            # transform one at a time to only have a lightweight list in memory
            if srp_before_pca:
                transformed_activations.append(pca.transform(srp.transform(new_activation.reshape(1, -1))))
            else:
                transformed_activations.append(pca.transform(new_activation.reshape(1, -1)))

        metric_trn = np.concatenate(transformed_activations, axis=0)
    else:
        activations = np.stack(activations, axis=0)
        metric_trn = pca.transform(srp.transform(activations)) if srp_before_pca else pca.transform(activations)

    # Encode test set
    transformed_activations = []
    for ii in tst_Idx:
        feat = np.load(feat_files[ii], allow_pickle=True)

        if avg_across_feat:
            new_activation = np.mean(feat[layer_id], axis=1).flatten()
        else:
            new_activation = feat[layer_id].flatten()

        # transform one at a time to only have a lightweight list in memory
        if srp_before_pca:
            transformed_activations.append(pca.transform(srp.transform(new_activation.reshape(1, -1))))
        else:
            transformed_activations.append(pca.transform(new_activation.reshape(1, -1)))

    metric_tst = np.concatenate(transformed_activations, axis=0)

    return metric_trn, metric_tst


def train_regression_per_ROI(trn_x, tst_x, trn_y, tst_y, veRSA=False, save_model=False, save_path=None):
    """
    Train a linear regression model (for one ROI) and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.
        veRSA (bool): Whether to apply RSA on top of encoding (veRSA).
        save_model (bool): Save the linear regression model to disk.
        save_path (str, optional): The path to save the model in (if save_model is True).

    Returns:
        List of correlation coefficients (numpy.ndarray) or single value output of veRSA (float).
    """
    if save_path is None or not os.path.exists(save_path):
        reg = LinearRegression().fit(trn_x, trn_y)
        y_prd = reg.predict(tst_x)
        if save_model:
            with open(save_path, "wb") as f:
                pickle.dump(reg, f)
    else:
        with open(save_path, "rb") as f:
            reg = pickle.load(f)
        y_prd = reg.predict(tst_x)
    if not veRSA:
        correlation_lst = np.zeros(y_prd.shape[1])
        for v in range(y_prd.shape[1]):
            correlation_lst[v] = pearsonr(y_prd[:, v], tst_y[:, v])[0]
        return correlation_lst
    else:
        prd_rdm = 1 - np.corrcoef(y_prd)
        brain_rdm = 1 - np.corrcoef(tst_y)
        r = spearmanr(sq(brain_rdm), sq(prd_rdm))[0]
        return r


def Ridge_Encoding(feat_path,
                   roi_path,
                   model_name,
                   trn_tst_split=0.8,
                   n_folds=3,
                   n_components=100,
                   batch_size=100,
                   srp_before_pca=False,
                   srp_on_subset=None,
                   mem_mode='performance',
                   avg_across_feat=False,
                   return_correlations=False,
                   random_state=42,
                   shuffle=True,
                   save_path="Linear_Encoding_Results",
                   file_name=None,
                   average_across_layers=False,
                   veRSA=False,
                   save_model=False,
                   save_pca=False,
                   layer_skips=()):
    """
    Perform ridge encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
            If we have a list of folders, each folder will be searched for .npy files and the analysis will be run
            for each. If folders contain different subject ROIs, make sure that the .npy file names are unique (e.g.
            V1_subj1.npy) across the folders.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float or int): Data to use for training (rest is used for testing). If int,
            it is absolute number of samples, if float, it is a fraction of the whole dataset.
        n_folds (int): Number of folds to split the data for cross-validation.
        avg_across_feat (bool): If True it averages the activations across axis 1. Necessary if different stimuli have a
            different size of features.
        return_correlations (bool): If True, return correlation values for each voxel (only with veRSA False).
        random_state (int): Seed for random operations to ensure reproducibility.
        shuffle (bool): Whether to shuffle the data before splitting into training and testing sets.
        save_path (str): Path to the directory where the results will be saved. Pick a different name for each
            different encoding set-up that you run (e.g. different trn_tst_split, n_folds, n_components,
            metric etc.). Keep the same name for different models that you want to compare, and running veRSA.
        file_name (str): (Optional) Name of the file to save the correlation results as. If None, will be the model
            name.
        average_across_layers (bool): If True, average the layer values across all given brain data.
        veRSA (bool): If True, performs RSA on top of the voxelwise encoding.
        save_model (bool): Save the linear regression model to disk.
        layer_skips (tuple, optional): Names of the model layers to skip during encoding. Use original layer names.


    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """

    result = Encoding(feat_path,
                      roi_path,
                      model_name,
                      trn_tst_split=trn_tst_split,
                      n_folds=n_folds,
                      n_components=n_components,
                      batch_size=batch_size,
                      srp_before_pca=srp_before_pca,
                      srp_on_subset=srp_on_subset,
                      mem_mode=mem_mode,
                      avg_across_feat=avg_across_feat,
                      return_correlations=return_correlations,
                      random_state=random_state,
                      shuffle=shuffle,
                      save_path=save_path,
                      file_name=file_name,
                      average_across_layers=average_across_layers,
                      metric="Ridge",
                      veRSA=veRSA,
                      save_model=save_model,
                      save_pca=save_pca,
                      layer_skips=layer_skips)

    return result


def Linear_Encoding(feat_path,
                    roi_path,
                    model_name,
                    trn_tst_split=0.8,
                    n_folds=3,
                    n_components=100,
                    batch_size=100,
                    srp_before_pca=False,
                    srp_on_subset=None,
                    mem_mode='performance',
                    avg_across_feat=False,
                    return_correlations=False,
                    random_state=42,
                    shuffle=True,
                    save_path="Linear_Encoding_Results",
                    file_name=None,
                    average_across_layers=False,
                    veRSA=False,
                    save_model=False,
                    save_pca=False,
                    layer_skips=()):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
            If we have a list of folders, each folder will be searched for .npy files and the analysis will be run
            for each. If folders contain different subject ROIs, make sure that the .npy file names are unique (e.g.
            V1_subj1.npy) across the folders.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float or int): Data to use for training (rest is used for testing). If int,
            it is absolute number of samples, if float, it is a fraction of the whole dataset.
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        srp_before_pca (bool): Whether to apply Sparse Random Projection (SRP) before PCA. Use when features are so
            high-dimensional that IncrementalPCA runs out of memory after some batches. Num of dims estimated by SRP.
        srp_on_subset (int or None): Number of samples to use for SRP fitting. If None, all samples are used,
            which is recommended if you have enough memory (if `srp_before_pca` is False it has no effect).
        mem_mode (str): 'saver' or 'performance'; Choose 'saver' if you don't have enough memory to store all
            training sample features, otherwise leave 'performance' as default. If you have `srp_before_pca` enabled,
            in the first case you will also need to restrict the number of samples for SRP fitting with `srp_on_subset`.
        avg_across_feat (bool): If True it averages the activations across axis 1. Necessary if different stimuli have a
            different size of features.
        return_correlations (bool): If True, return correlation values for each voxel (only with veRSA False).
        random_state (int): Seed for random operations to ensure reproducibility.
        shuffle (bool): Whether to shuffle the data before splitting into training and testing sets.
        save_path (str): Path to the directory where the results will be saved. Pick a different name for each
            different encoding set-up that you run (e.g. different trn_tst_split, n_folds, n_components,
            metric etc.). Keep the same name for different models that you want to compare, and running veRSA.
        file_name (str): (Optional) Name of the file to save the correlation results as. If None, will be the model
            name.
        average_across_layers (bool): If True, average the layer values across all given brain data.
        veRSA (bool): If True, performs RSA on top of the voxelwise encoding.
        save_model (bool): Save the linear regression model to disk.
        save_pca (bool): Save the PCA transform to disk.
        layer_skips (tuple, optional): Names of the model layers to skip during encoding. Use original layer names.


    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """

    result = Encoding(feat_path,
                      roi_path,
                      model_name,
                      trn_tst_split=trn_tst_split,
                      n_folds=n_folds,
                      n_components=n_components,
                      batch_size=batch_size,
                      srp_before_pca=srp_before_pca,
                      srp_on_subset=srp_on_subset,
                      mem_mode=mem_mode,
                      avg_across_feat=avg_across_feat,
                      return_correlations=return_correlations,
                      random_state=random_state,
                      shuffle=shuffle,
                      save_path=save_path,
                      file_name=file_name,
                      average_across_layers=average_across_layers,
                      metric="Linear",
                      veRSA=veRSA,
                      save_model=save_model,
                      save_pca=save_pca,
                      layer_skips=layer_skips)

    return result


def Encoding(feat_path,
             roi_path,
             model_name,
             trn_tst_split=0.8,
             n_folds=3,
             n_components=100,
             batch_size=100,
             srp_before_pca=False,
             srp_on_subset=None,
             mem_mode='performance',
             avg_across_feat=False,
             return_correlations=False,
             random_state=42,
             shuffle=True,
             save_path="Linear_Encoding_Results",
             file_name=None,
             average_across_layers=False,
             metric="Linear",
             veRSA=False,
             save_model=False,
             save_pca=False,
             layer_skips=()):

    # Which encoding metric are we using?
    if metric == "Linear":
        encoding_metric = _linear_encoding
    elif metric == "Ridge":
        encoding_metric = _ridge_encoding
    else:
        raise ValueError(f"Unknown metric '{metric}'. Please choose 'Linear' or 'Ridge'.")

    if avg_across_feat is True:
        print("avg_across_feat==True. This averages the activations across axis 1. Only neccessary if different stimuli"
              " have a different size of features (as with LLMs)")

    if return_correlations is True and veRSA is True:
        print("The option `return_correlations` is not supported with `veRSA`, because the voxel space is converted "
              "to condition space. It is now implicitly set to False.")

    # Turn the roi_path into a list of files
    roi_paths = get_npy_files(roi_path)

    result = encoding_metric(feat_path,
                             roi_paths,
                             model_name,
                             trn_tst_split=trn_tst_split,
                             n_folds=n_folds,
                             random_state=random_state,
                             shuffle=shuffle,
                             n_components=n_components,
                             batch_size=batch_size,
                             srp_before_pca=srp_before_pca,
                             srp_on_subset=srp_on_subset,
                             mem_mode=mem_mode,
                             avg_across_feat=avg_across_feat,
                             return_correlations=return_correlations,
                             save_path=save_path,
                             veRSA=veRSA,
                             save_model=save_model,
                             save_pca=save_pca,
                             layer_skips=layer_skips)
    if return_correlations:
        all_results_df, corr_dict = result
    else:
        all_results_df, _ = result

    if average_across_layers:
        warnings.warn(
            "Code will now average the layer values across all given brain data with average_across_layers=True")
        all_results_df = average_df_across_layers(all_results_df)

    # Create the output folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Determine the file name
    if file_name is None:
        file_name = model_name
        if veRSA:
            file_name += "_veRSA"

    # Save the DataFrame as a CSV file
    csv_file_path = f"{save_path}/{file_name}.csv"
    all_results_df.to_csv(csv_file_path, index=False)
    # And as npy
    dataframe_path = f"{save_path}/{file_name}.npy"
    np.save(dataframe_path, all_results_df)

    # If return_correlations is True, save the correlations dictionary as .npy
    if return_correlations:
        correlations_file_path = f"{save_path}/{file_name}_correlations.npy"
        np.save(correlations_file_path, corr_dict)

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
                     roi_paths,
                     model_name,
                     trn_tst_split=0.8,
                     n_folds=3,
                     n_components=100,
                     batch_size=100,
                     srp_before_pca=False,
                     srp_on_subset=None,
                     mem_mode='performance',
                     avg_across_feat=False,
                     return_correlations=False,
                     random_state=42,
                     shuffle=True,
                     save_path="Linear_Encoding_Results",
                     veRSA=False,
                     save_model=False,
                     save_pca=False,
                     layer_skips=()):
    # Initialize dictionaries to store results
    all_rois_dict = {}
    corr_dict = {}

    # Load feature files and get layer information
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)
    layer_list = [layer for layer in layer_list if layer not in layer_skips]
    num_layers = len(layer_list)

    # Loop over each fold for cross-validation
    for fold_ii in range(n_folds):
        all_rois_dict[fold_ii] = {}
        corr_dict[fold_ii] = {}

        # Set random seeds for reproducibility
        np.random.seed(fold_ii + random_state)
        random.seed(fold_ii + random_state)

        # Split the data indices into training and testing sets
        trn_Idx, tst_Idx = train_test_split(range(len(feat_files)), train_size=trn_tst_split,
                                            random_state=fold_ii + random_state, shuffle=shuffle)

        # Process each layer of model activations
        for layer_id in tqdm(layer_list, desc=f"Layers in fold {fold_ii}"):
            all_rois_dict[fold_ii][layer_id] = {}
            corr_dict[fold_ii][layer_id] = {}

            if save_pca or save_model:
                prediction_save_path = f"{save_path}/{model_name}/fold_{fold_ii}/{layer_id}"
                if not os.path.exists(prediction_save_path):
                    os.makedirs(prediction_save_path)

            # Encode the current layer using PCA and split into training and testing sets
            pca_trn, pca_tst = encode_layer(trn_Idx, tst_Idx, feat_path, layer_id, avg_across_feat, batch_size,
                                            n_components, srp_before_pca=srp_before_pca, srp_on_subset=srp_on_subset,
                                            mem_mode=mem_mode, save_pca=save_pca,
                                            save_path=f'{prediction_save_path}/pca.pkl' if save_pca else None)

            # Iterate through all ROI folder paths
            for counter, roi_path in enumerate(roi_paths):
                print(f"Processing ROI file {counter}, {roi_path}")
                roi_name = roi_path.split(os.sep)[-1].split(".")[0]

                # Load fMRI data for the current ROI and split into training and testing sets
                fmri_data = np.load(roi_path)
                fmri_trn, fmri_tst = fmri_data[trn_Idx], fmri_data[tst_Idx]

                # Train a linear regression model and compute correlations for the current ROI
                r_lst = train_regression_per_ROI(pca_trn, pca_tst, fmri_trn, fmri_tst, veRSA,
                                                 save_model=save_model,
                                                 save_path=f'{prediction_save_path}/{roi_name}.pkl' if save_model else None)

                # Mean of all voxel correlations (veRSA is no longer in voxel space)
                r = np.mean(r_lst) if not veRSA else r_lst

                # Store correlation results
                if return_correlations:
                    corr_dict[fold_ii][layer_id][roi_name] = r_lst

                all_rois_dict[fold_ii][layer_id][roi_name] = r

    # Compile all results into a DataFrame for easy analysis
    layers = list(all_rois_dict[0].keys())  # Get the layer list from the first fold
    rows = []
    # Iterate through each layer and ROI
    for layer_id in layers:
        for roi_path in roi_paths:
            roi_name = roi_path.split(os.sep)[-1].split(".")[0]

            # If we have more than one fold, collect the R values across folds from fold_dict
            if n_folds > 1:

                # find r_values per layer across folds
                r_values_across_folds = [all_rois_dict[fold_ii][layer_id][roi_name] for fold_ii in range(n_folds)]

                # Get average R value across folds
                R = np.mean(r_values_across_folds)

                # Perform t-test on the R values across folds
                _, significance = ttest_1samp(r_values_across_folds, 0)

                # Compute the Standard Error of the Mean (SEM)
                sem_value = sem(r_values_across_folds)

            # If there is only one fold, use the r_lst from the fold directly for testing
            else:
                # Get R Value
                R = all_rois_dict[0][layer_id][roi_name]

                if not veRSA:
                    # Perform t-test on the r_lst values since they are also correlation values
                    _, significance = ttest_1samp(r_lst, 0)
                    # Compute the Standard Error of the Mean (SEM)
                    sem_value = sem(r_lst)
                else:
                    significance = None
                    sem_value = None

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


def train_Ridgeregression_per_ROI(trn_x, tst_x, trn_y, tst_y, veRSA=False, save_model=False, save_path=None):
    """
    Train a ridge regression model (for one ROI) and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.
        veRSA (bool): Whether to apply RSA on top of encoding (veRSA).
        save_model (bool): Save the ridge regression model to disk.
        save_path (str, optional): The path to save the model in (if save_model is True).

    Returns:
        List of correlation coefficients (numpy.ndarray) or single value output of veRSA (float).
    """
    if save_path is None or not os.path.exists(save_path):
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

        if save_model:
            with open(save_path, "wb") as f:
                pickle.dump(reg, f)
    else:
        with open(save_path, "rb") as f:
            reg = pickle.load(f)
        y_prd = reg.predict(tst_x)

    if not veRSA:
        correlation_lst = np.zeros(y_prd.shape[1])
        for v in range(y_prd.shape[1]):
            correlation_lst[v] = pearsonr(y_prd[:, v], tst_y[:, v])[0]
        return correlation_lst
    else:
        prd_rdm = 1 - np.corrcoef(y_prd)
        brain_rdm = 1 - np.corrcoef(tst_y)
        r = spearmanr(sq(brain_rdm), sq(prd_rdm))[0]
        return r


def encode_layer_ridge(layer_id, trn_Idx, tst_Idx, feat_path, avg_across_feat):
    """
    Extracts and preprocesses neural network layer activations for ridge regression. 
    For each input, flattens the layer's activation vectors and optionally averages 
    across features.
    
    Parameters:
    - layer_id (str): Layer identifier
    - trn_Idx (list): Training indices
    - tst_Idx (list): Test indices 
    - feat_path (str): Path to feature files
    - avg_across_feat (bool): If True, averages activations across feature axis 1

    Returns:
    - trn, tst (numpy.ndarray): Processed training and test activations
    
    """
    if avg_across_feat:
        warnings.warn("avg_across_feat==True. This averages the activations across axis 1. Only necessary if different stimuli have a different size of features (as with LLMs)")
        
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()

    trn = []
    for ii in trn_Idx:
        feat = np.load(feat_files[ii], allow_pickle=True)
        if avg_across_feat:
            activation = np.mean(feat[layer_id], axis=1).flatten()
        else:
            activation = feat[layer_id].flatten()
            
        if trn and activation.shape != trn[-1].shape:
            raise ValueError("Elements in activations do not have the same shape. Please set 'avg_across_feat' to True to average across features.")
            
        trn.append(activation)
        
    tst = []
    for ii in tst_Idx:
        feat = np.load(feat_files[ii], allow_pickle=True)
        if avg_across_feat:
            activation = np.mean(feat[layer_id], axis=1).flatten()
        else:
            activation = feat[layer_id].flatten()
            
        if tst and activation.shape != tst[-1].shape:
            raise ValueError("Elements in activations do not have the same shape. Please set 'avg_across_feat' to True to average across features.")
            
        tst.append(activation)

    return np.array(trn), np.array(tst)


def _ridge_encoding(feat_path,
                    roi_paths,
                    model_name,
                    trn_tst_split=0.8,
                    n_folds=3,
                    n_components=100,
                    batch_size=100,
                    srp_before_pca=False,
                    srp_on_subset=None,
                    mem_mode='performance',
                    avg_across_feat=False,
                    return_correlations=False,
                    random_state=14,
                    shuffle=True,
                    save_path="Ridge_Encoding_Results",
                    veRSA=False,
                    save_model=False,
                    save_pca=False,
                    layer_skips=()):
    # Initialize dictionaries to store results
    all_rois_dict = {}
    corr_dict = {}

    # Load feature files and get layer information
    feat_files = glob.glob(feat_path + '/*.np[zy]')
    feat_files.sort()
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)
    layer_list = [layer for layer in layer_list if layer not in layer_skips]
    num_layers = len(layer_list)

    # Loop over each fold for cross-validation
    for fold_ii in range(n_folds):
        all_rois_dict[fold_ii] = {}
        corr_dict[fold_ii] = {}

        # Set random seeds for reproducibility
        np.random.seed(fold_ii + random_state)
        random.seed(fold_ii + random_state)

        # Split the data indices into training and testing sets
        trn_Idx, tst_Idx = train_test_split(range(len(feat_files)), train_size=trn_tst_split,
                                            random_state=fold_ii + random_state, shuffle=shuffle)

        # Process each layer of model activations
        for layer_id in tqdm(layer_list, desc=f"Layers in fold {fold_ii}"):
            all_rois_dict[fold_ii][layer_id] = {}
            corr_dict[fold_ii][layer_id] = {}

            if save_model:
                prediction_save_path = f"{save_path}/{model_name}/fold_{fold_ii}/{layer_id}"
                if not os.path.exists(prediction_save_path):
                    os.makedirs(prediction_save_path)

            # Encode the current layer using PCA and split into training and testing sets
            pca_trn, pca_tst = encode_layer_ridge(layer_id, trn_Idx, tst_Idx, feat_path, avg_across_feat)

            # Iterate through all ROI folder paths
            for counter, roi_path in enumerate(roi_paths):
                print(f"Processing ROI file {counter}, {roi_path}")
                roi_name = roi_path.split(os.sep)[-1].split(".")[0]

                # Load fMRI data for the current ROI and split into training and testing sets
                fmri_data = np.load(roi_path)
                fmri_trn, fmri_tst = fmri_data[trn_Idx], fmri_data[tst_Idx]

                # Train a regression model and compute correlations for the current ROI
                r_lst = train_Ridgeregression_per_ROI(pca_trn, pca_tst, fmri_trn, fmri_tst, veRSA,
                                                      save_model=save_model,
                                                      save_path=f'{prediction_save_path}/{roi_name}.pkl' if save_model else None)

                # Mean of all voxel correlations (veRSA is no longer in voxel space)
                r = np.mean(r_lst) if not veRSA else r_lst

                # Store correlation results
                if return_correlations:
                    corr_dict[fold_ii][layer_id][roi_name] = r_lst

                all_rois_dict[fold_ii][layer_id][roi_name] = r

    # Compile all results into a DataFrame for easy analysis
    layers = list(all_rois_dict[0].keys())  # Get the layer list from the first fold
    rows = []
    # Iterate through each layer and ROI
    for layer_id in layers:
        for roi_path in roi_paths:
            roi_name = roi_path.split(os.sep)[-1].split(".")[0]

            # If we have more than one fold, collect the R values across folds from fold_dict
            if n_folds > 1:

                # find r_values per layer across folds
                r_values_across_folds = [all_rois_dict[fold_ii][layer_id][roi_name] for fold_ii in range(n_folds)]

                # Get average R value across folds
                R = np.mean(r_values_across_folds)

                # Perform t-test on the R values across folds
                _, significance = ttest_1samp(r_values_across_folds, 0)

                # Compute the Standard Error of the Mean (SEM)
                sem_value = sem(r_values_across_folds)

            # If there is only one fold, use the r_lst from the fold directly for testing
            else:
                # Get R Value
                R = all_rois_dict[0][layer_id][roi_name]

                if not veRSA:
                    # Perform t-test on the r_lst values since they are also correlation values
                    _, significance = ttest_1samp(r_lst, 0)
                    # Compute the Standard Error of the Mean (SEM)
                    sem_value = sem(r_lst)
                else:
                    significance = None
                    sem_value = None

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