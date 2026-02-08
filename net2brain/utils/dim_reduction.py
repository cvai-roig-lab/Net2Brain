from typing import Optional, Callable
import numpy as np
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def estimate_from_files(file_list: list, key: int, feat_dim: tuple, open_func: Callable,
                        dim_reduction: str, n_samples_estim: int, n_components: Optional[int],
                        srp_before_pca: bool = False, pooling: Optional[Callable] = None,
                        clip_idx=None, time_idx=None):
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
        srp_before_pca (bool): Whether to apply Sparse Random Projection (SRP) before PCA. Use when features are so
            high-dimensional that PCA runs out of memory. Num of dims estimated by SRP.
        pooling (Callable or None): Pooling method for variable-length features. Required when
            features have variable lengths (e.g., LLM features).
        clip_idx: Optional index, used when estimating per video subclip.
        time_idx: Optional index, used when estimating per video timepoint.

    Returns:
        Fitted transform and number of components.

    """
    # Estimate the dimensionality reduction from a subset of the data
    feats_for_estim = np.empty((n_samples_estim, *feat_dim))
    for i, file in enumerate(file_list[:n_samples_estim]):
        if clip_idx is not None:
            if time_idx is not None:
                feats_for_estim[i, :] = open_func(file)[key][:, clip_idx, time_idx].squeeze()
            else:
                feats_for_estim[i, :] = open_func(file)[key][:, clip_idx].squeeze()
        else:
            feat = open_func(file)[key]
            if pooling is not None:
                feat = pooling(feat)
            feats_for_estim[i, :] = feat.squeeze()
    # Choose the dimensionality reduction method
    if dim_reduction == 'srp':
        fitted_transform = SparseRandomProjection(n_components=n_components) if n_components else (
            SparseRandomProjection())
        # Fit the dimensionality reduction
        if n_components:
            fitted_transform.fit(feats_for_estim.reshape(n_samples_estim, -1))
        else:
            sample_auto_proj = fitted_transform.fit_transform(
                feats_for_estim.reshape(n_samples_estim, -1))
            n_components = sample_auto_proj.shape[-1]
    elif dim_reduction == 'pca':
        fitted_transform = PCA(n_components=n_components) if n_components else PCA(n_components='mle')
        if srp_before_pca:
            srp = SparseRandomProjection(
                n_components=johnson_lindenstrauss_min_dim(n_samples_estim)
            )
            fitted_transform = Pipeline([('srp', srp), ('pca', fitted_transform)])
        # Fit the dimensionality reduction
        if n_components:
            fitted_transform.fit(feats_for_estim.reshape(n_samples_estim, -1))
        else:
            sample_auto_proj = fitted_transform.fit_transform(
                feats_for_estim.reshape(n_samples_estim, -1))
            n_components = sample_auto_proj.shape[-1]
    else:
        raise ValueError(f"Unknown dimensionality reduction method {dim_reduction} - choose 'srp' or 'pca'.")
    # Return the fitted transform and the number of components
    return fitted_transform, n_components
