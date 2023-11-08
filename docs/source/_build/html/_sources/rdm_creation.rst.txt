=======================================================
Creating Representational Dissimilarity Matrices (RDMs)
=======================================================

After extracting features from the neural network models, the next step is to compute Representational Dissimilarity Matrices (RDMs). The ``RDMCreator`` class facilitates this process.

Creating RDMs from DNN features
----------------

The ``RDMCreator`` requires:

- **Input**: Path to `.npz` files with the neural features for each image, formatted as [Batch x Channel x Height x Width].
- **Save Path** (optional): Destination directory for the generated RDMs.

Functionality:
^^^^^^

The ``RDMCreator``:

- **Outputs**: An RDM for each layer with the dimensionality (#Images, #Images).

.. code-block:: python

    # AlexNet Example
    from net2brain.rdm_creation import RDMCreator
    
    feat_path = "path/to/AlexNet_Feat"
    save_path = "path/to/AlexNet_RDM"
    
    creator = RDMCreator(feat_path, save_path)
    creator.create_rdms()  # Creates and saves RDMs

    # ResNet50 Example
    feat_path = "path/to/ResNet50_Feat"
    save_path = "path/to/ResNet50_RDM"
    
    creator = RDMCreator(feat_path, save_path)
    creator.create_rdms()  # Creates and saves RDMs



Creating RDMs from EEG Data
----------------

For each pair of images i,j find associated eeg trials X_i and X_j and using leave one out cross validation train a classifier, 
average accuracy is used for a distance measure. This is done at every timepoint.

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


**eeg_rdm** requires:

- **eeg**: EEG-Data
- **labels**: Labels for EEG-Data

.. code-block:: python

    from net2brain.preprocess.rdm import eeg_rdm
    rdm = eeg_rdm(subj_data['dat'],subj_data['lbl'])