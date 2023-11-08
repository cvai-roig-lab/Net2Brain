=======================================================
Creating Representational Dissimilarity Matrices (RDMs)
=======================================================

After extracting features from the neural network models, the next step is to compute Representational Dissimilarity Matrices (RDMs). The ``RDMCreator`` class facilitates this process.

Requirements for Generating RDMs
----------------

The ``RDMCreator`` requires:

- **Input**: Path to `.npz` files with the neural features for each image, formatted as [Batch x Channel x Height x Width].
- **Save Path** (optional): Destination directory for the generated RDMs.

Functionality:
----------------

The ``RDMCreator``:

- **Outputs**: An RDM for each layer with the dimensionality (#Images, #Images).

Example Usage
---------------

Below is an example of how to use the ``RDMCreator`` to generate RDMs using features from AlexNet and ResNet50 models.

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