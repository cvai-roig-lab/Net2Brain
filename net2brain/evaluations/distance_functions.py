import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean, cityblock, cosine, squareform

# Flatten the RDM while excluding the diagonal
def sq(x):
    """Converts a square-form distance matrix from a vector-form distance vector

    Args:
        x (numpy array): numpy array that should be vector

    Returns:
        numpy array: numpy array as vector
    """
    if x.ndim == 2:  # Only apply to 2D inputs
        return squareform(x, force='tovector', checks=False)
    return x  # If already 1D, return as-is



# List to hold all registered distance functions
registered_distance_functions = {}

# Decorator to register distance functions
def register_distance_function(func):
    """Decorator to register distance functions."""
    registered_distance_functions[func.__name__.replace("model_", "")] = func
    return func


# List to hold all registered noise ceiling distance functions
registered_nc_distance_functions = {}

# Decorator to register noise ceiling distance functions
def register_nc_distance_function(func):
    """Decorator to register noise ceiling distance functions."""
    registered_nc_distance_functions[func.__name__.replace("nc_", "")] = func
    return func




# 1. Spearman correlation
@register_distance_function
def model_spearman(model_rdm, rdms):
    """Calculate Spearman correlation."""
    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]

# 2. Euclidean distance
@register_distance_function
def model_euclidean(model_rdm, rdms):
    """Calculate Euclidean distance."""
    model_rdm_sq = sq(model_rdm)
    return [euclidean(sq(rdm), model_rdm_sq) for rdm in rdms]

# 3. Manhattan distance
@register_distance_function
def model_manhattan(model_rdm, rdms):
    """Calculate Manhattan (L1) distance."""
    model_rdm_sq = sq(model_rdm)
    return [cityblock(sq(rdm), model_rdm_sq) for rdm in rdms]

# 4. Cosine similarity
@register_distance_function
def model_cosine(model_rdm, rdms):
    """Calculate Cosine similarity."""
    model_rdm_sq = sq(model_rdm)
    return [1 - cosine(sq(rdm), model_rdm_sq) for rdm in rdms]

# 5. Pearson correlation
@register_distance_function
def model_pearson(model_rdm, rdms):
    """Calculate Pearson correlation."""
    model_rdm_sq = sq(model_rdm)
    return [stats.pearsonr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]




# Flatten the RDM while excluding the diagonal (for noise ceiling)
def get_uppertriangular(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions, 1)]



# 1. Spearman correlation for noise ceiling
@register_nc_distance_function
def nc_spearman(rdm1, rdm2):
    """Calculate Spearman for noise ceiling data.

    Args:
        rdm1 (numpy array): RDM of subject.
        rdm2 (numpy array): mean RDM of all subjects.

    Returns:
        float: squared correlation between both RDMs.
    """
    return (stats.spearmanr(rdm1, rdm2)[0]) ** 2

# 2. Euclidean distance for noise ceiling
@register_nc_distance_function
def nc_euclidean(rdm1, rdm2):
    """Calculate Euclidean distance for noise ceiling data.

    Args:
        rdm1 (numpy array): RDM of subject.
        rdm2 (numpy array): mean RDM of all subjects.

    Returns:
        float: Euclidean distance between both RDMs.
    """
    return euclidean(rdm1, rdm2)

# 3. Manhattan distance for noise ceiling
@register_nc_distance_function
def nc_manhattan(rdm1, rdm2):
    """Calculate Manhattan (L1) distance for noise ceiling data.

    Args:
        rdm1 (numpy array): RDM of subject.
        rdm2 (numpy array): mean RDM of all subjects.

    Returns:
        float: Manhattan distance between both RDMs.
    """
    return cityblock(rdm1, rdm2)

# 4. Cosine similarity for noise ceiling
@register_nc_distance_function
def nc_cosine(rdm1, rdm2):
    """Calculate Cosine similarity for noise ceiling data.

    Args:
        rdm1 (numpy array): RDM of subject.
        rdm2 (numpy array): mean RDM of all subjects.

    Returns:
        float: Cosine similarity between both RDMs.
    """
    return 1 - cosine(rdm1, rdm2)

# 5. Pearson correlation for noise ceiling
@register_nc_distance_function
def nc_pearson(rdm1, rdm2):
    """Calculate Pearson correlation for noise ceiling data.

    Args:
        rdm1 (numpy array): RDM of subject.
        rdm2 (numpy array): mean RDM of all subjects.

    Returns:
        float: squared Pearson correlation between both RDMs.
    """
    return (stats.pearsonr(rdm1, rdm2)[0]) ** 2