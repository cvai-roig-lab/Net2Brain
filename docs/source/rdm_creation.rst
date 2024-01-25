Creating Representational Dissimilarity Matrices (RDMs)
=======================================================

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/2_RDM_Creation.ipynb>`_!



Representational Dissimilarity Matrices (RDMs) are powerful tools in understanding the complex processing patterns of neural network models. The RDMCreator class is designed to streamline the creation of RDMs by quantifying the distinctiveness of neural responses to different stimuli.

Generating RDMs from Deep Neural Network Features
-------------------------------------------------

The `RDMCreator` transforms the high-dimensional activations of neural networks into a two-dimensional space, representing the dissimilarity between the responses to different inputs. This is crucial for comparing neural network processing to human brain activity.

Updated functionality for `RDMCreator`:

- **device**: The device on which the RDMs will be generated, either 'cpu' or 'cuda'.
- **verbose**: Whether to print the progress of the RDM generation process.
- **feature_path**: Path to `.npz` files containing layer features for each image, structured as *[Batch x Channels x Height x Width]*.
- **save_path** (Optional): The directory to save the generated RDMs.
- **save_format** (Optional): The format in which to save the RDMs, either 'npz' or 'pt'.
- **distance** (Optional): The distance metric to use. Defaults to correlation distance.
- **chunk_size** (Optional): The number of images processed at a time to manage memory usage. Defaults to processing all images at once.

The RDM Creator outputs an RDM for each specified neural network layer, with dimensions *(#Images x #Images)*, providing a matrix of pairwise dissimilarity scores.

.. code-block:: python

    from net2brain.rdm_creation import RDMCreator

    # Example using AlexNet features
    feat_path = "path/to/AlexNet_Feat"
    save_path = "path/to/AlexNet_RDM"

    creator = RDMCreator(verbose=True, device='cpu')
    save_path = creator.create_rdms(feature_path=feat_path, save_path=save_path, save_format='npz')


The default distance function is the correlation distance. To use a different distance function, we can specify the distance function in the **distance** parameter. The available distance functions can been seen by calling the **distance_functions** method of the RDMCreator. We created synonyms for the distance functions to make it easier to use them (i.e. l2 == euclidean). The available distance functions are:
.. code-block:: python
    creator.distance_functions()


You can also use custom distance functions by passing a function to the **distance** parameter. The function should take one argument `x` of shape `(N, D)`, which represents the features (dimension `D`) of the `N` images and return a pairwise distance matrix of shape `(N, N)`. For example, we can use the cosine distance function as follows:
.. code-block:: python
    
    import torch.nn.functional as F

    def custom_cosine(x):
        x_norm = F.normalize(x, p=2, dim=1)
        return 1 - (x_norm @ x_norm.T)

    creator.create_rdms(feature_path=feat_path, save_path='AlexNet_RDM_custom', save_format='npz', distance=custom_cosine)

To visualize the RDM of a single layer:

.. code-block:: python

    from net2brain.rdm_creation import LayerRDM

    rdm = LayerRDM.from_file("path/to/RDM_features.npz")
    rdm.plot(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


Constructing RDMs from EEG Data
----------------

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/Workshops/Net2Brain_EEG_Cutting_Edge_Workshop.ipynb>`_!


The creation of RDMs from EEG data involves comparing neural responses to pairs of stimuli.
A classifier is trained using cross-validation to determine the distinctiveness of EEG responses, which is then used to populate the RDM at every timepoint.

Pseudo code:

.. code-block:: python

    for t in timepoints
        for i,j in image pairs
            accuracy_ij = 0
            for k in number of instances
                trn_I = all EEG instances for image i except k
                trn_J = all EEG instances for image j except k
                tst_I = EEG instance k for image i
                tst_J = EEG instance k for image j
                LDA.fit([trn_I,trn_J])
                accuracy_ij += LDA.predict([tst_I,tst_j])
            RDM[i,j,tt] = accuracy_ij

To use this approach, **eeg_rdm** function is provided, which requires:

- **eeg**: EEG-Data
- **labels**: Labels for EEG-Data

.. code-block:: python

    from net2brain.preprocess.rdm import eeg_rdm
    rdm = eeg_rdm(subj_data['dat'],subj_data['lbl'])