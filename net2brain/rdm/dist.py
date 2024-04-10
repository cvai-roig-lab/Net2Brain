from typing import Optional

import torch
from torch import Tensor

from .dist_utils import register_distance_function, standardize


@register_distance_function()
@register_distance_function(name="l2")
def euclidean(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Computes the pairwise Euclidean distance between all vectors in `x`.

    Formula:
        `||x - y||_2 = sqrt(sum((x - y)^2))`

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector or
        a 3D tensor of shape `[b, n, d]` where `b` is the batch size.

    y : torch.Tensor, optional
       A 2D tensor of shape `[m, d]` where `m` is the number of vectors and `d` is the dimensionality of each vector or
       a 3D tensor of shape `[b, m, d]` where `b` is the batch size. If `y` is `None`, then `y = x`.

    Returns
    -------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` if `y` is `None` or `[n, m]` containing the pairwise distances, or
        a 3D tensor of shape `[b, n, n]` if `y` is `None` or `[b, n, m]` containing the batched pairwise distances.
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    if y is None:
        y = x
        y_norm = x_norm
    else:
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)

    op = torch.baddbmm if x.dim() == 3 else torch.addmm
    return op(
        x_norm,
        x,
        y.transpose(-2, -1),
        alpha=-2
    ).add_(y_norm.transpose(-2, -1)).clamp_min_(1e-30).sqrt_()


@register_distance_function()
@register_distance_function(name="l1")
def manhattan(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Computes the pairwise Manhattan distance between all vectors in `x` and `y`.

    Formula:
        `||x - y||_1 = sum(|x - y|)`

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector.

    y : torch.Tensor, optional
       A 2D tensor of shape `[m, d]` where `m` is the number of vectors and `d` is the dimensionality of each vector or
       a 3D tensor of shape `[b, m, d]` where `b` is the batch size. If `y` is `None`, then `y = x`.

    Returns
    -------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` if `y` is `None` or `[n, m]` containing the pairwise distances, or
        a 3D tensor of shape `[b, n, n]` if `y` is `None` or `[b, n, m]` containing the batched pairwise distances.
    """
    # dist = (x[:, None, :] - x[None, :, :]).abs().sum(dim=-1) # TOO SLOW, NOT BATCHED, TOO MUCH MEMORY, SEE Benchmarks
    if y is None:
        y = x
    return torch.cdist(x, y, p=1.0)


@register_distance_function(jit=True)
def cosine(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Computes the pairwise cosine distance between all vectors in `x` and `y`.

    Formula:
        `cosine(x, y) = 1 - (x . y) / (||x||_2 * ||y||_2)`

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector or
        a 3D tensor of shape `[b, n, d]` where `b` is the batch size.

    y : torch.Tensor, optional
       A 2D tensor of shape `[m, d]` where `m` is the number of vectors and `d` is the dimensionality of each vector or
       a 3D tensor of shape `[b, m, d]` where `b` is the batch size. If `y` is `None`, then `y = x`.

    Returns
    -------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` if `y` is `None` or `[n, m]` containing the pairwise distances, or
        a 3D tensor of shape `[b, n, n]` if `y` is `None` or `[b, n, m]` containing the batched pairwise distances.
    """
    norm_x = x / x.norm(p=2, dim=-1, keepdim=True)
    if y is None:
        norm_y = norm_x
    else:
        norm_y = y / y.norm(p=2, dim=-1, keepdim=True)
    return 1 - norm_x.matmul(norm_y.transpose(-2, -1))


@register_distance_function()
def correlation(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Computes the pairwise correlation distance between all vectors in `x`.

    Formula:
        `correlation(x, y) = 1 - (x - x.mean()) . (y - y.mean()) / (||x - x.mean()||_2 * ||y - y.mean()||_2)`

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector or
        a 3D tensor of shape `[b, n, d]` where `b` is the batch size.

    y : torch.Tensor, optional
       A 2D tensor of shape `[m, d]` where `m` is the number of vectors and `d` is the dimensionality of each vector or
       a 3D tensor of shape `[b, m, d]` where `b` is the batch size. If `y` is `None`, then `y = x`.

    Returns
    -------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` if `y` is `None` or `[n, m]` containing the pairwise distances, or
        a 3D tensor of shape `[b, n, n]` if `y` is `None` or `[b, n, m]` containing the batched pairwise distances.
    """
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    if y is None:
        y = x
    else:
        y = y - y.mean(dim=-1, keepdim=True)
        y = y / y.norm(p=2, dim=-1, keepdim=True)
    return 1 - x.matmul(y.transpose(-2, -1))


@register_distance_function(support_batch=False)
def pearson(x: Tensor) -> Tensor:
    """
    Computes the pairwise Pearson correlation distance between all vectors in `x`. The input is normalized along the
    first dimension, which normalizes for the mean level of activity and the variance of the activity pattern.

    See:
    Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis - connecting the branches
    of systems neuroscience. Frontiers in systems neuroscience, 2, 4. https://doi.org/10.3389/neuro.06.004.2008

    Parameters:
    -----------
    x : torch.Tensor
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector.

    Returns:
    --------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` containing the pairwise distances.
    """
    x = standardize(x, dim=0)
    return 1 - torch.corrcoef(x)
