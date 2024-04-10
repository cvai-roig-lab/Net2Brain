import numpy as np
import pytest
import torch
from scipy.spatial import distance

from net2brain.rdm import to_condensed, to_distance_matrix, dist
from net2brain.rdm.dist import euclidean, manhattan, cosine, correlation


def scipy_squareform_batched(x, batched=False):
    if batched:
        return np.stack([distance.squareform(x[i], checks=False) for i in range(x.shape[0])])
    else:
        return distance.squareform(x, checks=False)


@pytest.mark.parametrize('b', (None, 2, 8, 64))
@pytest.mark.parametrize('size', (4, 32, 128, 256, 512))
@pytest.mark.parametrize('dtype', (torch.float16, torch.float32, torch.float64))
def test_squareform_to_vector(device, b, size, dtype):
    shape = [b, size, size] if b is not None else [size, size]
    x = torch.rand(*shape, device=device, dtype=dtype)
    gt = torch.from_numpy(scipy_squareform_batched(x, b is not None))  # disable checks to allow not symmetric matrices
    out = to_condensed(x).cpu()
    assert torch.allclose(out, gt)


@pytest.mark.parametrize('b', (None, 2, 8, 64))
@pytest.mark.parametrize('size', (6, 496, 8128, 32640))
@pytest.mark.parametrize('dtype', (torch.float16, torch.float32, torch.float64))
def test_squareform_to_matrix(device, b, size, dtype):
    shape = [b, size] if b is not None else [size]
    x = torch.rand(*shape, device=device, dtype=dtype)
    gt = torch.from_numpy(scipy_squareform_batched(x, b is not None))  # disable checks to allow not symmetric matrices
    out = to_distance_matrix(x).cpu()
    assert torch.allclose(out, gt)


@pytest.mark.parametrize('shape', ((5, 4), (8, 3), (4, 128, 157)))
def test_to_condensed_fail(device, shape):
    with pytest.raises(ValueError):
        to_condensed(torch.rand(*shape, device=device), check_symmetric=False)


@pytest.mark.parametrize('shape', ((8, 7), (7,)))
def test_to_distance_matrix_fail(device, shape):
    with pytest.raises(ValueError):
        to_distance_matrix(torch.rand(*shape, device=device))


########################################################################################################################

@pytest.mark.parametrize('b', (None, 2, 8, 64))
@pytest.mark.parametrize('n', (4, 32, 128, 256,))
@pytest.mark.parametrize('d', (64, 128, 256))
@pytest.mark.parametrize('dtype', (torch.float64,))
class TestBatchedDistanceFunction:
    @staticmethod
    def scipy_cdist_batched(x, metric='correlation'):
        if x.ndim == 3:
            return np.stack([distance.cdist(x[i], x[i], metric=metric) for i in range(x.shape[0])])
        else:
            return distance.cdist(x, x, metric=metric)

    @staticmethod
    def run_dist_test(func, gt_func, device, shape, dtype, **kwargs):
        shape = [x for x in shape if x is not None]
        x = torch.rand(*shape, device=device, dtype=dtype)
        x_np = x.cpu().numpy()
        gt = torch.from_numpy(gt_func(x_np)).to(dtype=x.dtype)
        out = func(x, **kwargs).cpu()
        assert torch.isnan(out).sum() == 0, f"Found NaN values in output of {func.__name__}"
        assert torch.allclose(out, gt, atol=1e-5, rtol=1e-5)

    def test_euclidean(self, device, b, n, d, dtype):
        self.run_dist_test(euclidean, lambda x: self.scipy_cdist_batched(x, metric='euclidean'), device,
                           (b, n, d), dtype)

    def test_manhatten(self, device, b, n, d, dtype):
        self.run_dist_test(manhattan, lambda x: self.scipy_cdist_batched(x, metric='cityblock'), device,
                           (b, n, d), dtype)

    def test_cosine(self, device, b, n, d, dtype):
        self.run_dist_test(cosine, lambda x: self.scipy_cdist_batched(x, metric='cosine'), device,
                           (b, n, d), dtype)

    def test_correlation(self, device, b, n, d, dtype):
        self.run_dist_test(correlation, lambda x: self.scipy_cdist_batched(x, metric='correlation'), device,
                           (b, n, d), dtype)


@pytest.mark.parametrize('metric', ('euclidean', 'correlation'))
@pytest.mark.parametrize('chunk_size', (128, 1024,))
@pytest.mark.parametrize('n', (32, 256, 2048))
@pytest.mark.parametrize('d', (64, 128))
@pytest.mark.parametrize('dtype', (torch.float64,))
def test_chunked_dist(device, metric, chunk_size, n, d, dtype):
    TestBatchedDistanceFunction.run_dist_test(dist,
                                              lambda x: TestBatchedDistanceFunction.scipy_cdist_batched(x,
                                                                                                        metric=metric),
                                              device, (None, n, d), dtype, chunk_size=chunk_size, metric=metric)
