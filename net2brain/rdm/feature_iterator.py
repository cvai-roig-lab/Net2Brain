import re
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache, cached_property
from pathlib import Path
from typing import Union, Tuple, List, Iterator, Dict, Type, Iterable, Callable, Optional

import numpy as np

from net2brain.utils.dim_reduction import estimate_from_files


def natural_keys(text: str) -> List[Union[int, str]]:
    """
    A function that sorts strings with numbers in a natural way.
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def nsorted(iterable: Iterable, key: Optional[Callable] = None, reverse: bool = False) -> List:
    """
    Sorts an iterable in a natural way by converting the strings to integers.
    """
    key = key or (lambda x: x)
    return sorted(iterable, key=lambda x: natural_keys(key(x)), reverse=reverse)


class FeatureFormat(Enum):
    NPZ_CONSOLIDATED = 1
    # - layer1.npz
    # |-- stimulus1
    # |-- stimulus2
    # - layer2.npz
    # |-- stimulus1
    # |-- stimulus2

    NPZ_SEPARATE = 2
    # - stimulus1.npz
    # |-- layer1
    # |-- layer2
    # - stimulus2.npz
    # |-- layer1
    # |-- layer2

    HDF5 = 3


@lru_cache(maxsize=16)
def open_npz(path: Path) -> Dict[str, np.ndarray]:
    return np.load(path, allow_pickle=True)


def detect_feature_format(root: Union[str, Path]) -> FeatureFormat:
    """
    Detects the format of the feature files in the given directory. It does not check if all feature files are valid.

    Args:
        root : str or Path
            Path to the directory containing the feature files.

    Returns:
        FeatureFormat
            The format of the feature files.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory {root} does not exist")
    elif root.is_dir():
        dir_iter = root.iterdir()
        file = next(dir_iter)
        if not file.is_file():
            raise ValueError(f"Directory {root} does not contain any files.")
        if file.suffix == ".npz":
            if 'consolidated' in str(file):
                return FeatureFormat.NPZ_CONSOLIDATED
            else:
                return FeatureFormat.NPZ_SEPARATE
        raise ValueError(f"Invalid file suffix {file.suffix} for directory {root}. Only .npz files are supported.")
    elif root.is_file():
        if root.suffix in (".h5", ".hdf5", ".hdf", ".h5py"):
            return FeatureFormat.HDF5
        else:
            raise ValueError(f"Invalid file suffix: {root.suffix}")
    else:
        raise ValueError(f"Invalid path: {root}")


class FeatureEngine(ABC):
    """
    Abstract class for feature engines. Each feature engine is responsible for extracting features from a specific
    format of feature files.
    Takes optional keyword arguments for dimensionality reduction at the feature loading, currently only implemented for
    the NPZSeperateEngine.
    """

    def __init__(self, root: Path,
                 dim_reduction=None,
                 n_samples_estim=100,
                 n_components=10000,
                 max_dim_allowed=None):
        self.root = Path(root)
        self.dim_reduction = dim_reduction
        self.n_samples_estim = n_samples_estim
        self.n_components = n_components
        self.max_dim_allowed = max_dim_allowed

    @abstractmethod
    def get_iterator(self) -> Iterator:
        """
        Returns the iterator object that provided the information needed to extract features from the feature files in
        every step. For example, in the NPZ_CONSOLIDATED format, the iterator object is the iterator of the directory
        containing the feature files, and in the NPZ_SEPARATE format, the iterator object is the iterator of the keys
        in the first feature file.
        """
        pass

    @abstractmethod
    def num_layers(self) -> int:
        """
        Returns the number of layers in the feature files, since the number of layers is not always equal to the number
        of files. For example, in the NPZ_CONSOLIDATED format, the number of layers is equal to the number of files,
        but in the NPZ_SEPARATE format, the number of layers is equal to the number of keys in each file. The number of
        layers is the basic iteration length of the feature engine.
        """
        pass

    @abstractmethod
    def next(self, item) -> Tuple[str, List[str], np.ndarray]:
        """
        Extracts features from the feature files. The `item` argument is the item returned by the iterator object
        returned by `get_iterator()`. For example, in the NPZ_CONSOLIDATED format, the `item` is the path to the feature
        file, and in the NPZ_SEPARATE format, the `item` is the key in the feature file. The return value is a tuple
        containing the stimuli names and the stacked features extracted from the feature files.

        Args:
            item : any
                The item returned by the iterator object returned by `get_iterator()`.

        Returns:
            Tuple[str, List[str], np.ndarray]
                The first element is the layer name, the second element is the list of stimuli names, and the third
                element is the stacked features extracted from the feature files.
        """
        pass


class EngineRegistry:
    """
    Registry for feature engines. It is used to register new feature engines and get the feature engine for a specific
    feature format.
    """
    ENGINE_REGISTER: Dict[FeatureFormat, Type[FeatureEngine]] = {}

    def register(self, engine) -> None:
        if engine.feature_format in self.ENGINE_REGISTER:
            raise ValueError(f"Engine for {engine.feature_format} already registered")
        self.ENGINE_REGISTER[engine.feature_format] = engine

    def get_engine(self, feature_format: FeatureFormat) -> Type[FeatureEngine]:
        return self.ENGINE_REGISTER[feature_format]


engine_registry = EngineRegistry()


@engine_registry.register
class NPZConsolidateEngine(FeatureEngine):
    feature_format = FeatureFormat.NPZ_CONSOLIDATED

    def get_iterator(self) -> Iterator:
        return filter(lambda x: x.suffix == '.npz', nsorted(self.root.iterdir(), key=lambda x: x.stem))

    @cached_property
    def num_layers(self) -> int:
        return len(list(self.get_iterator()))

    def next(self, item) -> Tuple[str, List[str], np.ndarray]:
        feat_npz = open_npz(item)
        stimuli, feats = zip(*nsorted(feat_npz.items(), key=lambda x: x[0]))
        layer = item.stem
        return layer, stimuli, np.stack(feats)


@engine_registry.register
class NPZSeparateEngine(FeatureEngine):
    feature_format = FeatureFormat.NPZ_SEPARATE

    def get_iterator(self) -> Iterator:
        return iter(nsorted(open_npz(self._stimuli[0])))

    @cached_property
    def num_layers(self) -> int:
        return len(open_npz(self._stimuli[0]))

    @cached_property
    def _stimuli(self) -> List[Path]:
        return nsorted(self.root.iterdir(), key=lambda x: x.stem)

    def next(self, item) -> Tuple[str, List[str], np.ndarray]:
        stimuli = []
        sample = open_npz(self._stimuli[0])[item]
        feat_dim = sample.shape[1:]
        # Check if dimensionality reduction is needed
        if self.dim_reduction and (not self.max_dim_allowed or len(sample.flatten()) > self.max_dim_allowed):
            # Estimate the dimensionality reduction from a subset of the data
            fitted_transform, self.n_components = estimate_from_files(self._stimuli, item, feat_dim, open_npz,
                                                         self.dim_reduction, self.n_samples_estim, self.n_components)
            feats = np.empty((len(self._stimuli), self.n_components))
            for i, file in enumerate(self._stimuli):
                if not file.suffix == ".npz":
                    warnings.warn(f"File {file} is not a valid feature file. Skipping...")
                feats[i, :] = fitted_transform.transform(open_npz(file)[item].reshape(1, -1)).squeeze(0)
                stimuli.append(file.stem)
        # Otherwise load features without dimensionality reduction
        else:
            feats = np.empty((len(self._stimuli), *feat_dim))
            for i, file in enumerate(self._stimuli):
                if not file.suffix == ".npz":
                    warnings.warn(f"File {file} is not a valid feature file. Skipping...")
                feats[i, :] = open_npz(file)[item].squeeze(0)
                stimuli.append(file.stem)
        return item, stimuli, feats


@engine_registry.register
class HDF5Engine(FeatureEngine):
    feature_format = FeatureFormat.HDF5

    def get_iterator(self) -> Iterator:
        pass

    def num_layers(self) -> int:
        pass

    def next(self, item) -> Tuple[str, List[str], np.ndarray]:
        pass


class FeatureIterator:
    def __init__(self, root: Path, **kwargs):
        """
        Iterator for extracting features from feature files. It automatically detects the format of the feature files
        and uses the corresponding feature engine to extract features.

        Args:
            root: str or Path
                Path to the directory containing the feature files.
            **kwargs: dict
                Keyword arguments for dimensionality reduction at the feature loading, currently only implemented for
                the NPZSeperateEngine.
        """
        if not isinstance(root, (Path, str)):
            raise TypeError(f"Expected Path or str for feature path, got {type(root)}")
        self.root = Path(root)

        self.format: FeatureFormat = detect_feature_format(self.root)

        if self.format == FeatureFormat.NPZ_SEPARATE:
            warnings.warn("FeatureIterator is not optimized for NPZ_SEPARATE format. "
                          "Consider using NPZ_CONSOLIDATED format instead.")

        if kwargs and self.format != FeatureFormat.NPZ_SEPARATE:
            warnings.warn(f"Keyword arguments provided for dimensionality reduction only supported for NPZ_SEPARATE, "
                          f"and will have no effect for the provided format {self.format}.")

        self.engine: FeatureEngine = engine_registry.get_engine(self.format)(self.root, **kwargs)

        self._iter = None

    def __len__(self) -> int:
        return self.engine.num_layers

    def __iter__(self):
        self._iter = self.engine.get_iterator()
        return self

    def __next__(self) -> Tuple[str, List[str], np.ndarray]:
        try:
            item = next(self._iter)
        except StopIteration:
            raise StopIteration

        return self.engine.next(item)
