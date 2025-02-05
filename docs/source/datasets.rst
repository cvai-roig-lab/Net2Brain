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




Loading Datasets
----------------

Before diving into feature extraction, selecting an appropriate dataset is crucial. Net2Brain facilitates this by offering access to several datasets, each comprising a rich collection of stimulus images and corresponding brain data. This includes not only images but also Representational Dissimilarity Matrices (RDMs) derived from fMRI and MEG measurements.

Available datasets include:
- ``"78images"`` from the Algonauts2019 Challenge Training Set A
- ``"92images"`` from the Algonauts2019 Challenge Test Set
- ``"bonner_pnas2017"`` from the study by Micheal F. Bonner et al.
- ``"Algonauts"`` from the Algonauts Challenge, by EJ Allen et al.
- ``"NSD_872"`` a subset of the NSD Dataset with 872 images viewed by all participants
- ``"Things-Fmri Test"`` : ---


To list all available datasets you can use:

.. code-block:: python
        
    from net2brain.utils.download_datasets import list_available_datasets
    list_available_datasets()



These datasets can be loaded using specific classes in the Net2Brain toolkit:

.. code-block:: python

    from net2brain.utils.download_datasets import Dataset78images, Dataset92images, DatasetBonnerPNAS2017, DatasetAlgonauts_NSD, DatasetNSD_872, DatasetThings_fMRI
    paths_78 = Dataset78images().load_dataset()
    paths_92 = Dataset92images().load_dataset()
    paths_bonner = DatasetBonnerPNAS2017().load_dataset()
    paths_Algonauts = DatasetAlgonauts_NSD().load_dataset()
    paths_NSD_872 = DatasetNSD_872().load_dataset()
    paths_things = DatasetThings_fMRI.load_dataset()

    # Example to access stimuli and ROI data for the `78images` Dataset:
    stimuli_path = paths_78["stimuli_path"]
    roi_path = paths_78["roi_path"]

**Special Functions for NSD and Algonauts Datasets**

The DatasetAlgonauts_NSD and NSD_872 dataset, sharing roots with the COCO dataset, is enriched by functions that facilitate seamless interactions:

- **ID Conversion:** Switch between NSD and COCO identifiers.
- **Image Downloads:** Access original COCO images directly from NSD.
- **Segmentation Masks:** Download COCO segmentation masks for NSD images.
- **Caption Downloads:** Retrieve original COCO captions for downloaded images.
- **Image Manipulation:** Crop and rename COCO images to fit NSD conventions.
- **Visualization:** Display images alongside their segmentation masks.

.. code-block:: python

    from net2brain.utils.download_datasets import DatasetNSD_872
    nsd_dataset = DatasetNSD_872() 
    paths = nsd_dataset.load_dataset()

    # Convert NSD ID to COCO ID and vice versa
    coco_id = nsd_dataset.NSDtoCOCO("02950")
    nsd_id = nsd_dataset.COCOtoNSD("262145")

    # Downloading and visualizing functions
    nsd_dataset.Download_COCO_Images("NSD Dataset/NSD_872_images", "NSD Dataset/coco_images")
    nsd_dataset.Download_COCO_Segmentation_Masks("NSD Dataset/NSD_872_images", "NSD Dataset/coco_masks")
    nsd_dataset.Download_COCO_Captions("NSD Dataset/NSD_872_images", "NSD Dataset/coco_captions")
    nsd_dataset.Visualize("NSD Dataset/coco_images", "NSD Dataset/coco_masks", "03171")

    # Cropping and renaming for compatibility
    nsd_dataset.Crop_COCO_to_NSD("NSD Dataset/coco_images", "NSD Dataset/coco_images")
    nsd_dataset.RenameToNSD("NSD Dataset/coco_images")

    # Additional renaming functionality for datasets using Algonauts naming conventions
    nsd_dataset.RenameAlgonautsToNSD("path/to/Algonauts")
