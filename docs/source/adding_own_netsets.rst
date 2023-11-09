Creating Your Own NetSet
========================

Introduction
------------

Creating a custom NetSet into **Net2Brain** is really easy and only involves a few simple steps, including copying a template, adding data types, and configuring model settings. This guide will walk you through each step with examples to help you create your own NetSet.

Step 1: Copying the Empty NetSet
---------------------------------

Start by copying our template-file ``empty_netset.py``. This file contains a template class `YOURNETSET` which inherits from `NetSetBase`. 

.. code-block:: python

    class YOURNETSET(NetSetBase):  # Rename to your desired netset name
        def __init__(self, model_name, device):
            # Your code here

Step 2: Customizing the NetSet
------------------------------

Rename the `YOURNETSET` class to the name of your netset. Define the supported data types and the netset name and the path to your config-file (step 3).

.. code-block:: python

    class MyCustomNetSet(NetSetBase):
        self.supported_data_types = ['image', 'audio']  # Example data types
        self.netset_name = "MyCustomNetSet"
        self.config_path = os.path.join(directory_path, "./") # Path to configuration file that lists all models & functions to access it (see other configs)

Step 3: Creating a Configuration File
-------------------------------------

Create a JSON configuration file that lists all the models and their functions. The configuration files for the other architectures lie under *"/net2brain/architectures/configs"*. Feel free to take a look at them for inspiration.

.. code-block:: json

    {
        "AlexNet": {
            "model_function": "torchvision.models.alexnet",
            "nodes": ["features.0", "..."]
        }
    }

Step 4: Optional Modifications
-------------------------------

You may wish to add custom preprocessing or feature cleaning methods. These can be specified within the class methods. For example:

.. code-block:: python

    def get_preprocessing_function(self, data_type):
        # Custom preprocessing steps

    def get_feature_cleaner(self, data_type):
        # Custom cleaning steps

Step 5: Importing Your NetSet
-----------------------------

Finally, import your new netset into `feature_extractor.py`.

.. code-block:: python

    from my_custom_netset import MyCustomNetSet

Conclusion
----------

You now have a custom NetSet ready for use with your feature extraction pipeline. Remember to test your NetSet thoroughly to ensure it functions as expected.
