Creating Representational Dissimilarity Matrices (RDMs)
=======================================================

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/2_RDM_Creation.ipynb>`_!



Representational Dissimilarity Matrices (RDMs) are powerful tools in understanding the complex processing patterns of neural network models. The RDMCreator class is designed to streamline the creation of RDMs by quantifying the distinctiveness of neural responses to different stimuli.

Generating RDMs from Deep Neural Network Features
-------------------------------------------------

The `RDMCreator` transforms the high-dimensional activations of neural networks into a two-dimensional space, representing the dissimilarity between the responses to different inputs. This is crucial for comparing neural network processing to human brain activity.

Prerequisites for `RDMCreator`:


- **feat_path**: A path to `.npz` files containing neural features for each stimulus, structured as *[Batch x Channels x Height x Width]*.
- **save_path**: (Optional) The target directory to save the generated RDMs.
- **distance**: (Optional) Which distance metric to use. Defaults to Pearson

Functionality of `RDMCreator`:

- It outputs an RDM for each neural network layer, with dimensions *(#Stimuli x #Stimuli)*, providing a matrix of pairwise dissimilarity scores.


.. code-block:: python

    # AlexNet Example
    from net2brain.rdm_creation import RDMCreator
    
    feat_path = "path/to/AlexNet_Feat"
    save_path = "path/to/AlexNet_RDM"
    
    creator = RDMCreator(feat_path, save_path)
    creator.create_rdms()  # Creates and saves RDMs

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