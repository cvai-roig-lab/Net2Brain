from typing import Optional, Callable
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA


def estimate_from_files(file_list: list, key: int, feat_dim: tuple, open_func: Callable,
                        dim_reduction: str, n_samples_estim: int, n_components: Optional[int]):
    """
    Estimate a dimensionality reduction from a subset of the data in `file_list`.
    Performed separately per `key` (corresponding to a layer).
    Method is either Sparse Random Projection ('srp') or PCA ('pca').

    Args:
        file_list: List of file paths to the data.
        key: Key to extract from the data.
        feat_dim: Tuple of dimensions of the feature data.
        open_func: Function to open the data.
        dim_reduction: Dimensionality reduction method to use. Choose 'srp' or 'pca'.
        n_samples_estim: Number of samples to use for the estimation.
        n_components: Number of components to reduce to.

    Returns:
        Fitted transform and number of components.

    """
    # Estimate the dimensionality reduction from a subset of the data
    feats_for_estim = np.empty((n_samples_estim, *feat_dim))
    for i, file in enumerate(file_list[:n_samples_estim]):
        feats_for_estim[i, :] = open_func(file)[key].squeeze(0)
    # Choose the dimensionality reduction method
    if dim_reduction == 'srp':
        fitted_transform = SparseRandomProjection(n_components=n_components) if n_components else (
            SparseRandomProjection())
    elif dim_reduction == 'pca':
        fitted_transform = PCA(n_components=n_components) if n_components else PCA(n_components='mle')
    else:
        raise ValueError(f"Unknown dimensionality reduction method {dim_reduction} - choose 'srp' or 'pca'.")
    # Fit the dimensionality reduction
    if n_components:
        fitted_transform.fit(feats_for_estim.reshape(n_samples_estim, -1))
    else:
        sample_auto_proj = fitted_transform.fit_transform(feats_for_estim.reshape(n_samples_estim, -1))
        n_components = sample_auto_proj.shape[-1]
    # Return the fitted transform and the number of components
    return fitted_transform, n_components
