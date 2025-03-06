"""This code is adapted from https://github.com/brainML/Stacking
Ruogu Lin, Thomas Naselaris, Kendrick Kay, and Leila Wehbe (2023). Stacked regressions and structured variance partitioning for interpretable brain maps."""

import numpy as np



def Stacked_Variance_Partitioning(r2s, stacked_r2s, save_path="./"):
    """
    Perform both forward and backward variance partitioning on stacked encoding results.
    
    This function applies structured variance partitioning to understand which layers 
    contribute most to the prediction of brain activity. 
    
    Parameters
    ----------
    r2s : numpy.ndarray
        R-squared values for individual layer models, shape (n_layers, n_voxels)
        where n_layers is the number of layers and n_voxels is the number of voxels.
    stacked_r2s : numpy.ndarray
        R-squared values for the full stacked model, shape (n_voxels,)
    save_path : str, optional
        Path to save the results as an NPZ file
        
    Returns
    -------
    vp_results : dict
        Dictionary containing all variance partitioning results:
        - 'vp': Square root of forward variance partitioning differences (n_layers, n_voxels)
        - 'vp_square': Raw forward variance partitioning differences (n_layers, n_voxels)
        - 'r2s_array': Combined array of R-squared values used for forward partitioning (n_layers, n_voxels)
        - 'vp_sel_layer_forward': Layer assignments based on forward 95% criterion (n_voxels,)
        - 'vpr': Square root of backward variance partitioning differences (n_layers, n_voxels)
        - 'vpr_square': Raw backward variance partitioning differences (n_layers, n_voxels)
        - 'r2sr': Combined array of R-squared values used for backward partitioning (n_layers, n_voxels)
        - 'vpr_sel_layer_backward': Layer assignments based on backward 95% criterion (n_voxels,)
    """
    print("This code is adapted from https://github.com/brainML/Stacking")
    
    # Check input shapes
    if r2s.ndim != 2:
        raise ValueError(f"r2s should be a 2D array, got shape {r2s.shape}")
    
    if stacked_r2s.ndim != 1:
        raise ValueError(f"stacked_r2s should be a 1D array, got shape {stacked_r2s.shape}")
    
    if r2s.shape[1] != stacked_r2s.shape[0]:
        raise ValueError(f"Number of voxels in r2s ({r2s.shape[1]}) must match stacked_r2s ({stacked_r2s.shape[0]})")
    
    vp_results = {}
    
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
        'vpr_sel_layer_backward': vpr_sel_layer_backward
    }
    
    # Save results if save_path is provided
    np.savez(save_path, **vp_results)
    print(f"Variance partitioning results saved to {save_path}.npz")
 
    return vp_results




# ----- Forward Variance Partitioning -----
def forward_variance_partitioning(r2s, stacked_r2s):
    """
    Perform forward variance partitioning analysis.
    
    This function calculates how much variance each feature space contributes to the 
    prediction by comparing progressively simpler models. It implements the forward 
    inclusion approach where feature spaces are removed one by one.
    
    Parameters
    ----------
    r2s : numpy.ndarray
        R-squared values for individual models, shape (n_models, n_voxels)
    stacked_r2s : numpy.ndarray
        R-squared values for the stacked model, shape (n_voxels,)
        
    Returns
    -------
    vp : numpy.ndarray
        Square root of the variance partitioning differences, shape (n_models, n_voxels)
    vp_square : numpy.ndarray
        Raw variance partitioning differences, shape (n_models, n_voxels)
    r2s_array : numpy.ndarray
        Combined array of R-squared values used for partitioning
    vp_sel_layer : numpy.ndarray
        Layer assignments based on 95% performance criterion, shape (n_voxels,)
    """
   
    r2_7 = r2s[-1, :]
    num_layers = r2s.shape[0]
    
    # Prepare r2s array with stacked encoding results
    r2s_list = []
    for k in range(r2s.shape[0] - 1):
        r2s_list.append(r2s[k, :])
    r2s_list.append(stacked_r2s)
    r2s_array = np.array(r2s_list)
    
    # Calculate variance by difference of R2
    vp = np.zeros((r2s_array.shape[0], r2s_array.shape[1]))
    vp_square = np.zeros((r2s_array.shape[0], r2s_array.shape[1]))
    
    for j in range(vp.shape[0] - 1):
        diff = r2s_array[j, :] - r2s_array[j + 1, :]
        vp[j, :] = np.where(diff >= 0, np.sqrt(diff), -1) # -1 = removing a layer unexpectedly increased the R² value -> overcompensation or redundancy

        vp_square[j, :] = (r2s_array[j, :] - r2s_array[j + 1, :])
    
    vp[-1, :] = np.sqrt(r2_7)
    vp_square[-1, :] = r2_7
    
    vp_sel_layer = select_layers_forward(r2s, num_layers)
    
    return vp, vp_square, r2s_array, vp_sel_layer

# ----- Backward Variance Partitioning -----
def backward_variance_partitioning(r2s):
    """
    Perform backward variance partitioning analysis.
    
    This function calculates how much variance each feature space contributes to the
    prediction using a backward elimination approach. It starts with all features and
    progressively adds back features that were removed.
    
    Parameters
    ----------
    r2s : numpy.ndarray
        R-squared values for individual models, shape (n_models, n_voxels)
        
    Returns
    -------
    vpr : numpy.ndarray
        Square root of the variance partitioning differences, shape (n_models, n_voxels)
    vpr_square : numpy.ndarray
        Raw variance partitioning differences, shape (n_models, n_voxels)
    r2sr : numpy.ndarray
        Combined array of R-squared values used for backward partitioning
    vpr_sel_layer : numpy.ndarray
        Layer assignments based on 95% performance criterion in backward approach,
        shape (n_voxels,)
    """
    r2_1 = r2s[0, :]
    num_layers = r2s.shape[0]
    
    # Prepare r2sr array
    r2sr_list = []
    for k in range(r2s.shape[0] - 1):
        r2sr_list.append(r2s[k, :])
    r2sr_list.append(r2_1)
    r2sr = np.array(r2sr_list)
    
    # Calculate variance by difference of R2
    vpr = np.zeros((r2sr.shape[0], r2sr.shape[1]))
    vpr_square = np.zeros((r2sr.shape[0], r2sr.shape[1]))
    
    for j in range(vpr.shape[0] - 1):
        diff = r2sr[j, :] - r2sr[j + 1, :]
        vpr[j, :] = np.where(diff >= 0, np.sqrt(diff), -1)  # -1 = removing a layer unexpectedly increased the R² value -> overcompensation or redundancy
        vpr_square[j, :] = (r2sr[j, :] - r2sr[j + 1, :])
    
    vpr[-1, :] = np.sqrt(r2_1)
    vpr_square[-1, :] = r2_1
    
    vpr_sel_layer = select_layers_backward(r2sr, num_layers)
    
    return vpr, vpr_square, r2sr, vpr_sel_layer

# ----- Feature Selection Based on 95% Criteria -----
def select_layers_forward(r2s, num_layers):
    """
    Select layers based on forward 95% criterion.
    
    For each voxel, finds the most complex layer that maintains at least 95%
    of the performance of the full model. This implements the forward feature
    attribution approach.
    
    Parameters
    ----------
    r2s : numpy.ndarray
        R-squared values for individual models, shape (n_models, n_voxels)
    num_layers : int
        Total number of layers/models being compared
        
    Returns
    -------
    vp_sel_layer : numpy.ndarray
        Array of layer indices assigned to each voxel, where higher values
        indicate more complex layers, shape (n_voxels,)
    """
    vp_sel_layer = np.zeros(r2s.shape[1])
    
    for i in range(r2s.shape[1]):
        if r2s[0, i] <= 0:
            vp_sel_layer[i] = float('nan')
            continue
            
        selected_layer = num_layers  # Default to last layer
        for j in range(1, num_layers):
            if r2s[j, i] < 0.95 * r2s[0, i]:
                selected_layer = j
                break
        vp_sel_layer[i] = selected_layer
        
    return vp_sel_layer

def select_layers_backward(r2sr, num_layers):
    """
    Select layers based on backward 95% criterion.
    
    For each voxel, finds the simplest layer that maintains at least 95%
    of the performance of the full model. This implements the backward feature
    attribution approach.
    
    Parameters
    ----------
    r2sr : numpy.ndarray
        R-squared values array for backward analysis, shape (n_models, n_voxels)
    num_layers : int
        Total number of layers/models being compared
        
    Returns
    -------
    vpr_sel_layer : numpy.ndarray
        Array of layer indices assigned to each voxel, where lower values
        indicate simpler layers, shape (n_voxels,)
    """
    vpr_sel_layer = np.zeros(r2sr.shape[1])
    
    for i in range(r2sr.shape[1]):
        if r2sr[0, i] <= 0:
            vpr_sel_layer[i] = float('nan')
            continue
            
        selected_layer = 1  # Default to first layer
        for j in range(1, num_layers):
            if r2sr[j, i] < 0.95 * r2sr[0, i]:
                selected_layer = num_layers - j
                break
        vpr_sel_layer[i] = selected_layer
        
    return vpr_sel_layer