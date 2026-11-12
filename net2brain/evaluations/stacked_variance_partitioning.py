"""This code is adapted from https://github.com/brainML/Stacking
Ruogu Lin, Thomas Naselaris, Kendrick Kay, and Leila Wehbe (2023). Stacked regressions and structured variance partitioning for interpretable brain maps."""

import numpy as np



def Stacked_Variance_Partitioning(nested_r2s_forward, nested_r2s_backward, save_path=None):
    """
    Perform both forward and backward variance partitioning on nested stacked models.
    
    This function applies structured variance partitioning to understand which layers 
    contribute most to the prediction of brain activity. 
    
    Parameters
    ----------
    nested_r2s_forward : numpy.ndarray
        R-squared values of the forward nested stacked models, shape (n_layers, n_voxels): row k is the stacked
        model on layers k+1..K (row 0: all layers, last row: layer K alone). Compute them with
        stacked_encoding.nested_stacked_r2s, or load 'r2s_array' from the file saved by Stacked_Encoding(vpa=True).
    nested_r2s_backward : numpy.ndarray
        R-squared values of the backward nested stacked models, shape (n_layers, n_voxels): row k is the stacked
        model on layers 1..K-k (row 0: all layers, last row: layer 1 alone), or 'r2sr' from that file.
    save_path : str, optional
        Path to save the results as an NPZ file (".npz" is appended); None (default) does not save
        
    Returns
    -------
    vp_results : dict
        Dictionary containing all variance partitioning results:
        - 'vp': Square root of forward variance partitioning differences (n_layers, n_voxels)
        - 'vp_square': Raw forward variance partitioning differences (n_layers, n_voxels)
        - 'r2s_array': R-squared values of the forward nested models (n_layers, n_voxels)
        - 'vp_sel_layer_forward': Layer assignments based on forward 95% criterion (n_voxels,)
        - 'vpr': Square root of backward variance partitioning differences (n_layers, n_voxels)
        - 'vpr_square': Raw backward variance partitioning differences (n_layers, n_voxels)
        - 'r2sr': R-squared values of the backward nested models (n_layers, n_voxels)
        - 'vpr_sel_layer_backward': Layer assignments based on backward 95% criterion (n_voxels,)
    """
    print("This code is adapted from https://github.com/brainML/Stacking")
    
    # Check input shapes
    if nested_r2s_forward.ndim != 2 or nested_r2s_forward.shape != nested_r2s_backward.shape:
        raise ValueError(f"nested_r2s_forward and nested_r2s_backward should be 2D arrays of the same shape "
                         f"(n_layers, n_voxels), got {nested_r2s_forward.shape} and {nested_r2s_backward.shape} "
                         f"(see stacked_encoding.nested_stacked_r2s)")
    
    vp_results = {}
    
    # Forward variance partitioning
    print("Running variance partitioning: Forward Pass")
    vp, vp_square, r2s_array, vp_sel_layer_forward = forward_variance_partitioning(nested_r2s_forward)
    print(f"Forward VP shape: {vp.shape}")
    print(f"Forward VP square shape: {vp_square.shape}")
    
    
    # Backward variance partitioning
    print("Running variance partitioning: Backward Pass")
    vpr, vpr_square, r2sr, vpr_sel_layer_backward = backward_variance_partitioning(nested_r2s_backward)
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
    if save_path is not None:
        np.savez(save_path, **vp_results)
        print(f"Variance partitioning results saved to {save_path}.npz")
 
    return vp_results




# ----- Forward Variance Partitioning -----
def forward_variance_partitioning(nested_r2s_forward):
    """
    Perform forward variance partitioning analysis.
    
    This function calculates how much variance each feature space contributes to the 
    prediction by comparing progressively simpler models. It implements the forward 
    inclusion approach where feature spaces are removed one by one.
    
    Parameters
    ----------
    nested_r2s_forward : numpy.ndarray
        R-squared values of the stacked models on layers k+1..K for row k, shape (n_models, n_voxels)
        
    Returns
    -------
    vp : numpy.ndarray
        Square root of the variance partitioning differences, shape (n_models, n_voxels)
    vp_square : numpy.ndarray
        Raw variance partitioning differences, shape (n_models, n_voxels)
    r2s_array : numpy.ndarray
        The nested R-squared values used for partitioning
    vp_sel_layer : numpy.ndarray
        Layer assignments based on 95% performance criterion, shape (n_voxels,)
    """
   
    r2s_array = nested_r2s_forward
    r2_7 = r2s_array[-1, :]  # last layer alone
    num_layers = r2s_array.shape[0]
    
    # Calculate variance by difference of R2
    vp = np.zeros((r2s_array.shape[0], r2s_array.shape[1]))
    vp_square = np.zeros((r2s_array.shape[0], r2s_array.shape[1]))
    
    for j in range(vp.shape[0] - 1):
        diff = r2s_array[j, :] - r2s_array[j + 1, :]
        vp[j, :] = np.where(diff >= 0, np.sqrt(diff), -1) # -1 = removing a layer unexpectedly increased the R² value -> overcompensation or redundancy

        vp_square[j, :] = (r2s_array[j, :] - r2s_array[j + 1, :])
    
    vp[-1, :] = np.sqrt(r2_7)
    vp_square[-1, :] = r2_7
    
    vp_sel_layer = select_layers_forward(r2s_array, num_layers)
    
    return vp, vp_square, r2s_array, vp_sel_layer

# ----- Backward Variance Partitioning -----
def backward_variance_partitioning(nested_r2s_backward):
    """
    Perform backward variance partitioning analysis.
    
    This function calculates how much variance each feature space contributes to the
    prediction using a backward elimination approach. It starts with all features and
    progressively adds back features that were removed.
    
    Parameters
    ----------
    nested_r2s_backward : numpy.ndarray
        R-squared values of the stacked models on layers 1..K-k for row k, shape (n_models, n_voxels)
        
    Returns
    -------
    vpr : numpy.ndarray
        Square root of the variance partitioning differences, shape (n_models, n_voxels)
    vpr_square : numpy.ndarray
        Raw variance partitioning differences, shape (n_models, n_voxels)
    r2sr : numpy.ndarray
        The nested R-squared values used for backward partitioning
    vpr_sel_layer : numpy.ndarray
        Layer assignments based on 95% performance criterion in backward approach,
        shape (n_voxels,)
    """
    r2sr = nested_r2s_backward
    r2_1 = r2sr[-1, :]  # first layer alone
    num_layers = r2sr.shape[0]
    
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
    
    For each voxel, finds the layer whose removal (together with the layers before it)
    drops the R-squared below 95% of the full model, i.e. the lowest layer still needed.
    This implements the forward feature attribution approach.
    
    Parameters
    ----------
    r2s : numpy.ndarray
        R-squared values of the forward nested models, shape (n_models, n_voxels)
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
    
    For each voxel, finds the layer whose removal (together with the layers after it)
    drops the R-squared below 95% of the full model, i.e. the highest layer still needed.
    This implements the backward feature attribution approach.
    
    Parameters
    ----------
    r2sr : numpy.ndarray
        R-squared values of the backward nested models, shape (n_models, n_voxels)
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
                selected_layer = num_layers - j + 1  # row j lacks layer K-j+1, which row j-1 has
                break
        vpr_sel_layer[i] = selected_layer
        
    return vpr_sel_layer