Loading Datasets
----------------

Before diving into feature extraction, we need to select an appropriate dataset. Net2Brain conveniently offers access to several datasets, each comprising a collection of stimulus images along with corresponding brain data. This data includes Representational Dissimilarity Matrices (RDMs) derived from fMRI and MEG measurements, providing a rich source for analysis.

Currently, you can choose from the following datasets:

- ``"78images"``
- ``"92images"``
- ``"bonner_pnas2017"``

Each dataset is uniquely identified by a name that can be used to load it with the `load_dataset` function. For example, to load the dataset from the study by Bonner et al. (2017), use the following code:

.. code-block:: python

    from net2brain.utils.download_datasets import load_dataset
    # Load stimuli and brain data from the specified dataset
    stimuli_path, roi_path = load_dataset("bonner_pnas2017")

This function returns the paths to the stimuli images and the region of interest (ROI) data, setting the stage for subsequent feature extraction and analysis.

