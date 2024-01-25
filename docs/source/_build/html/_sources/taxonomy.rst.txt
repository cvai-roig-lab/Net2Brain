
Exploring the Toolbox - Model Taxonomy
----------------------------------------------

.. note::

   Run and test this code by using `this notebook <https://github.com/cvai-roig-lab/Net2Brain/blob/main/notebooks/0_Exploring_Net2Brain.ipynb>`_!

Model Taxonomy serves as a comprehensive guide to navigate through the multitude of neural network models available in the toolbox. It provides a structured way to categorize models based on attributes like dataset, architecture, training method, and visual task. This aids users in selecting the most suitable model for their specific research requirements.

The taxonomy is designed to simplify the process of model selection by categorizing models into a searchable framework. It helps users to:

- Identify models that fit their experimental needs.
- Compare models across different attributes.
- Save time by quickly locating models trained for specific tasks or datasets.
- Make informed decisions about which models may yield the most relevant insights for their research.

By using the Model Taxonomy, researchers can efficiently scout through various models and select the one that best aligns with their study's objectives.

.. code-block:: python

    from net2brain.feature_extraction import (
        show_all_architectures,
        show_all_netsets,
        show_taxonomy,
        print_netset_models,
        find_model_like_name,
        find_model_by_dataset,
        find_model_by_training_method,
        find_model_by_visual_task,
        find_model_by_custom
    )

Viewing All Models and Architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To explore all available models and their corresponding netsets:

.. code-block:: python

    show_all_architectures()
    show_all_netsets()

For a closer look at the models within a specific netset:

.. code-block:: python

    print_netset_models('standard')

Finding a Specific Model
^^^^^^^^^^^^^^^^^^^^^^^^

To find a model by its name:

.. code-block:: python

    find_model_like_name('ResNet')

Utilizing the Model Taxonomy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The toolbox offers a detailed taxonomy of models to streamline your search:

.. code-block:: python

    show_taxonomy()  # Shows the model taxonomy

Searching Models by Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find models based on specific attributes:

.. code-block:: python

    # Find models by the dataset they were trained on
    find_model_by_dataset("Taskonomy")

    # Discover models by their training method
    find_model_by_training_method("SimCLR")

    # Search for models trained for a particular visual task
    find_model_by_visual_task("Panoptic Segmentation")

Custom Searches
^^^^^^^^^^^^^^^

For tailored searches combining various attributes or focusing on a specific model:

.. code-block:: python

    find_model_by_custom(["COCO", "Object Detection"], model_name="fpn")
