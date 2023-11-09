
=================================
Feature Extraction
=================================

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/1_Feature_Extraction.ipynb>`_!


The ``FeatureExtractor`` class in Net2Brain is designed for extracting features from deep neural network models. Below are the steps and explanations for utilizing this class effectively.


Initializing the FeatureExtractor
---------------------------------

The ``FeatureExtractor`` class requires the model from which features will be extracted. You can initialize the class with the following parameters:

- ``model`` (required): The name or instance of the model.
- ``netset`` (optional): The collection of networks that the model belongs to. Not needed if you supply your own model.
- ``device`` (optional): Computation device, e.g., 'cuda' or 'cpu'. Defaults to cpu.
- ``pretrained`` (optional): Whether to use a pretrained model. Defaults to True.

Additional optional parameters to customize the FeatureExtractor, especially when using your own custom model, will be discussed further below.

- ``preprocessor`` (optional): Your custom data transform function (if not provided, standard ImageNet transformations will be used)
- ``feature_cleaner`` (optional):  Your custom feature cleaner (if not provided, no cleaning will be done)
- ``extraction_function`` (optional): Your custom extraction function (if not provided, standard Torchextractor will be used)


.. code-block:: python

    from net2brain.feature_extraction import FeatureExtractor
    fx = FeatureExtractor(model='AlexNet', netset='Standard', device='cpu')

Extracting Features
-------------------

To extract features, use the ``extract`` method with the following parameters:

- ``data_path`` (required): Path to the image (.jpg, .png)/video (.mp4, .avi), /text dataset (.txt).
- ``save_path`` (optional): Where to save the extracted features. Defaults to a folder named with the current date.
- ``layers_to_extract`` (optional): Layers from which to extract features. Defaults to the layers specified during initialization.

.. code-block:: python

    fx.extract(data_path=stimuli_path, save_path='AlexNet_Feat')

Inspecting and Modifying Layers to Extract
------------------------------------------

Inspect default layers for extraction:

.. code-block:: python

    print(fx.layers_to_extract)

List all extractable layers:

.. code-block:: python

    print(fx.get_all_layers())

To specify layers, pass them to the ``extract`` method:

.. code-block:: python

    fx.extract(data_path=stimuli_path, layers_to_extract=['layer1', 'layer2'])


.. _Custom_DNN:

Using FeatureExtractor with a Custom DNN
----------------------------------------

You can integrate a custom model by providing the model instance and optionally custom functions for preprocessing, feature extraction, and cleaning.

.. code-block:: python

    from torchvision import models
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    fx = FeatureExtractor(model=model, device='cpu')
    fx.extract(data_path=stimuli_path, save_path='ResNet50_Feat', layers_to_extract=['layer1', 'layer2'])

Custom Functions Example
------------------------

Here's how to define custom preprocessing, extraction, and cleaning functions:

.. code-block:: python

    def my_preprocessor(image, model_name, device):
        # Define custom preprocessing steps
        # Return preprocessed image tensor

    def my_extractor(preprocessed_data, layers_to_extract, model):
        # Define custom feature extraction steps
        # Return extracted features

    def my_cleaner(features):
        # Define custom feature cleaning steps
        # Return cleaned features

    # Usage with custom functions
    fx = FeatureExtractor(model=model, device='cpu', preprocessor=my_preprocessor, feature_cleaner=my_cleaner, extraction_function=my_extractor)
    fx.extract(stimuli_path, layers_to_extract=['layer1', 'layer2'])
