"""

"""

import inspect
import math
from functools import partial
from typing import Union, Callable, List, Optional, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

DISTANCE_FUNCTIONS = {}


def _prepare_dist_func(func: Callable,
                       name: Optional[str] = None,
                       jit: bool = False,
                       support_batch: bool = True,
                       output_condensed: bool = False) -> Callable:
    args = inspect.getfullargspec(func).args

    if jit:
        func = torch.jit.script(func)

    func._support_batch = getattr(func, "_support_batch", support_batch)

    if 'x' not in args:
        raise ValueError(f"Distance function {name} must have an argument named `x`.")
    # mark if function support pairwise distance to support chunking
    func._pairwise = getattr(func, "_pairwise", 'y' in args)

    func._output_condensed = getattr(func, "_output_condensed", output_condensed)

    if func._pairwise and func._output_condensed:
        raise ValueError(f"The distance function {name} cannot output a condensed distance vector if in pairwise mode.")

    return func


def register_distance_function(name: Optional[str] = None,
                               jit: bool = False,
                               support_batch: bool = True,
                               output_condensed: bool = False) -> Callable:
    """
    Decorator to register a distance function.

    Parameters
    ----------
    name : str, optional
        The name of the distance function. If not specified, the name of the function will be used.
    jit : bool, default=False
        Whether to compile the function using `torch.jit.script`.
    support_batch : bool, default=True
        Whether the function supports batched inputs.
    output_condensed : bool, default=False
        Whether the function outputs a condensed distance vector.

    Returns
    -------
    Callable
    """

    def wrapper(func: Callable):
        func_name = name

        if func_name is None:
            func_name = func.__name__
        if func_name in DISTANCE_FUNCTIONS:
            raise ValueError(f"Distance function {name} is already registered.")

        func = _prepare_dist_func(func, name, jit, support_batch, output_condensed)
        DISTANCE_FUNCTIONS[func_name] = func
        return func

    return wrapper


def valid_distance_functions() -> List[str]:
    return list(DISTANCE_FUNCTIONS.keys())


def check_dist_input(x: Union[Tensor, np.ndarray],
                     y: Optional[Union[Tensor, np.ndarray]] = None,
                     device: Optional[Union[str, torch.device]] = None,
                     dtype: Optional[torch.dtype] = None,
                     func: Optional[Callable] = None
                     ) -> Dict[str, Tensor]:
    """
    Checks the input for the distance functions.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of the vectors.
    y : torch.Tensor or np.ndarray, optional
        A 2D tensor of shape `[m, d]` where `m` is the number of vectors and `d` is the dimensionality of the vectors.
    device : str or torch.device, optional
        The device to move the input tensor to.
    dtype : torch.dtype, optional
        The dtype to cast the input tensor to.
    func : Callable, optional
        The distance function to check the input for.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The input tensors `x` and `y` with the following modifications:
        - `x` is not a `torch.Tensor` then a `ValueError` is raised.
        - `x` is not a 2D/3D tensor then a `ValueError` is raised.
        - `x` is not contiguous then it is made contiguous.
        - `x` is converted to the specified dtype if provided.
        - `x` is not on the specified device then it is moved to the specified device.
    If `y` is `None`, then `None` is returned for `y`.
    """

    def _check_input(tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        if not isinstance(tensor, Tensor):
            raise ValueError(f"Expected argument array to be a `torch.Tensor` but got {type(x)}.")

        if func is not None:
            if not getattr(func, "_support_batch", True) and tensor.ndim == 3:
                raise ValueError(f"Function {func.__name__} does not support batched inputs.")

        if not (tensor.ndim == 2 or tensor.ndim == 3):
            raise ValueError(
                f"Expected argument array to be a 2D tensor of shape `[n, d]` or a 3D tensor of shape `[b, n, d]`"
                f" but got {tensor.shape}.")

        # disable any gradients
        tensor.requires_grad_(False)

        # make contiguous since some functions benefit from this
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if dtype is not None:
            tensor = tensor.to(dtype)

        # move to device if specified
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    if y is None:
        x = _check_input(x)
        return {"x": x}
    else:
        if func is not None and not getattr(func, "_pairwise", False):
            raise ValueError(f"Function {func.__name__} does not support pairwise inputs.")
        x = _check_input(x)
        y = _check_input(y)

        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                f"Expected argument arrays to have the same dimensionality but got {x.shape} and {y.shape}.")
        if x.ndim == 3 and y.ndim == 3 and x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Expected argument arrays to have the same batch size but got x={x.shape[0]} and y={y.shape[0]}.")
        return {"x": x, "y": y}


def to_condensed(x: torch.Tensor, check_symmetric: bool = False) -> torch.Tensor:
    """
    Converts a symmetric distance matrix to a condensed distance vector.

    Parameters
    ----------

    x : torch.Tensor
        A symmetric distance matrix of shape (n, n) or (b, n, n).

    check_symmetric : bool, default=False
        Whether to check that the input matrix is symmetric. This cost additional computation and
        is turned off by default.

    Returns
    -------
    torch.Tensor
        A condensed distance vector of shape (n * (n - 1) / 2, ) or (b, n * (n - 1) / 2).

    Examples
    --------
    >>> import torch
    >>> from dist import to_condensed
    >>> x = torch.tensor([[0, 1, 2, 3],
    ...                   [1, 0, 4, 5],
    ...                   [2, 4, 0, 6],
    ...                   [3, 5, 6, 0]])
    >>> to_condensed(x)
    tensor([1., 2., 3., 4., 5., 6.])
    >>> to_condensed(x, check_symmetric=True)
    tensor([1., 2., 3., 4., 5., 6.])
    >>> to_condensed(x[None, :, :], check_symmetric=True)
    tensor([[1., 2., 3., 4., 5., 6.]])
    """
    shape = x.shape

    if shape[-2] != shape[-1]:
        raise ValueError(f"The input matrix must be square, but got shape ({tuple(shape)}).")

    if check_symmetric:
        if not torch.allclose(x, x.transpose(-2, -1)):
            raise ValueError("The input matrix must be symmetric.")

    if len(shape) == 2:
        i = torch.triu_indices(shape[0], shape[1], offset=1, device=x.device)
        return x.flatten().index_select(0, i[0] * shape[0] + i[1])
    elif len(shape) == 3:
        i = torch.triu_indices(shape[1], shape[2], offset=1, device=x.device)
        return x.flatten(1).index_select(1, i[0] * shape[1] + i[1])
    else:
        raise ValueError(f"Input must be 2- or 3-d but has {len(shape)} dimension(s).")


def to_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Converts a condensed distance vector to a symmetric distance matrix.

    Parameters
    ----------
    x : torch.Tensor
        A condensed distance vector of shape (n * (n - 1) / 2, ) or (b, n * (n - 1) / 2).

    Returns
    -------
    torch.Tensor
        A symmetric distance matrix of shape (n, n) or (b, n, n).
    """
    shape = x.shape
    d = math.ceil(math.sqrt(shape[-1] * 2))

    if d * (d - 1) != shape[-1] * 2:
        raise ValueError("The input must be a condensed distance matrix.")

    i = torch.triu_indices(d, d, offset=1, device=x.device)
    if len(shape) == 1:
        out = torch.zeros(d, d, dtype=x.dtype, device=x.device)
        out[i[0], i[1]] = x
        out[i[1], i[0]] = x
    elif len(shape) == 2:
        out = torch.zeros(shape[0], d, d, dtype=x.dtype, device=x.device)
        out[..., i[0], i[1]] = x
        out[..., i[1], i[0]] = x
    else:
        raise ValueError(f"Input must be 2- or 3-d but has {len(shape)} dimension(s).")
    return out


def is_condensed_1d(x: torch.Tensor) -> bool:
    """
    Checks whether the input is a 1d condensed distance vector.

    Parameters
    ----------
    x : torch.Tensor
        Expects a 1D tenor of shape (n * (n - 1) / 2, ) to check whether it is a condensed distance vector.

    Returns
    -------
    bool
        Whether the input is a condensed distance vector.
    """
    shape = x.shape
    d = math.ceil(math.sqrt(shape[-1] * 2))
    return d * (d - 1) == shape[-1] * 2 and len(shape) == 1


def dist(x: Union[Tensor, np.ndarray],
         y: Optional[Union[Tensor, np.ndarray]] = None,
         chunk_size: Optional[int] = None,
         metric: Union[str, Callable] = "euclidean",
         **kwargs) -> Tensor:
    """
    Computes the pairwise distance between all vectors in `x`; or between all vectors in `x` and `y`.

    Optionally, the computation can be chunked into smaller chunks to reduce memory usage.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        A 2D tensor of shape `[n, d]` where `n` is the number of vectors and `d` is the dimensionality of each vector or
        a 3D tensor of shape `[b, n, d]` where `b` is the batch size if the metric supports batched inputs.

    y : torch.Tensor or np.ndarray, optional
       A 2D tensor of shape `[m, d]` where `m` is the number of vectors and `d` is the dimensionality of each vector or
       a 3D tensor of shape `[b, m, d]` where `b` is the batch size. If `y` is `None`, then `y = x`.

    chunk_size : int, optional
        The chunk size to use for computing the pairwise distances.

    metric : str or Callable, optional
        The distance metric to use. If a string is passed, it must be one of `valid_distance_functions()`. If a callable
        is passed, it must define a custom distance function.

    **kwargs
        Additional keyword arguments.

    Returns
    -------
    out : torch.Tensor
        A 2D tensor of shape `[n, n]` if `y` is `None` or `[n, m]` containing the pairwise distances, or
        a 3D tensor of shape `[b, n, n]` if `y` is `None` or `[b, n, m]` containing the batched pairwise distances.

    """
    if metric not in valid_distance_functions() and not callable(metric):
        raise ValueError(
            f"Unknown distance metric {metric}. Use one of {valid_distance_functions()} or a custom callable."
        )
    if callable(metric):
        func = partial(_prepare_dist_func(metric), **kwargs)
    else:
        func = DISTANCE_FUNCTIONS[metric]

    dist._output_condensed = getattr(func, "_output_condensed", False)

    inputs = check_dist_input(x, y, func=func, device=kwargs.pop("device", None), dtype=kwargs.pop("dtype", None))
    verbose = kwargs.pop("verbose", False)

    if chunk_size is None:
        return func(**inputs, **kwargs)
    else:
        return _apply_chunked(func, **inputs, chunk_size=chunk_size, verbose=verbose, **kwargs)


def _apply_chunked(func: Callable, x: Tensor, y: Optional[Tensor] = None, chunk_size: int = 1024, verbose: bool = False,
                   **kwargs) -> Tensor:
    """
    Applies a distance function to the input in chunks.

    Parameters
    ----------
    func : Callable
        The distance function to apply.
    x : torch.Tensor
        The input tensor.
    y : torch.Tensor, optional
        The second input tensor.
    chunk_size : int
        The chunk size.
    verbose : bool
        Whether to print progress information.
    **kwargs
        Additional keyword arguments for the distance function.

    Returns
    -------
    torch.Tensor
        Pairwise distances.
    """
    if x.ndim == 3:
        raise ValueError("Chunking is not supported for batched inputs.")

    if y is None:
        y = x

    num_samples = x.shape[0]
    out = torch.empty(num_samples, y.shape[0], dtype=x.dtype, device=x.device)

    slices = gen_batches(num_samples, chunk_size)

    bar = tqdm(slices, disable=not verbose, total=num_samples // chunk_size)
    for s in bar:
        if s.start == 0 and s.stop == num_samples:
            x_chunk = x
        else:
            x_chunk = x[s]
        out[s] = func(x_chunk, y, **kwargs)
    return out


def gen_batches(n, batch_size):
    """
    Generates batches of indices.

    Parameters
    ----------
    n: int
        The number of samples.
    batch_size: int
        The batch size.

    Yields
    -------
    slice
        A slice object containing the indices of the batch.
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)
