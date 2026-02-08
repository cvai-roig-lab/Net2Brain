Model Zoo
================

**Net2Brain** is an all-in-one solution that offers access to approximately 600 pre-trained neural networks across various visual tasks. These networks enable comprehensive experimental comparisons between human brain activity and artificial neural network activations. Moreover, users have the flexibility to incorporate their own models, making the toolkit highly adaptable and limitless in terms of architecture types.


What is a Netset?
-----------------
A Netset is a curated collection of neural network models from the same family but with potentially different training protocols or datasets. This organization allows for clear differentiation when multiple versions of a model, such as ResNet50, are available from various sources, ensuring clarity and ease of use.


Available Netsets
-----------------

**Net2Brain** facilitates the exploration and utilization of a vast range of Deep Neural Networks (DNNs) through its diverse netsets, which are libraries of different pre-trained models:

1. **Standard torchvision** (`Pytorch`)
   A compendium of torchvision models catering to a spectrum of tasks from image classification to video classification. Detailed information can be found on the `torchvision models page <https://pytorch.org/vision/stable/models.html>`_.

2. **Timm** (`Timm`)
   A library containing a rich array of advanced computer vision models developed by Ross Wightman. More details are available on the `Timm GitHub repository <https://github.com/rwightman/pytorch-image-models#models>`_.

3. **PyTorch Hub** (`Torchhub`)
   This netset includes models for a variety of visual tasks accessible through the torch.hub API, not encompassed by torchvision. For more, see the `PyTorch Hub documentation <https://pytorch.org/docs/stable/hub.html>`_.

4. **MMAction** (`MMAction`)
   Offers a wide range of video models, including more recent transformer-based architectures. Explore more on the `MMAction documentation <https://mmaction2.readthedocs.io/en/latest/get_started/overview.htmll>`_.

5. **Unet** (`Unet`)
   Unet models are specialized for abnormality segmentation in brain MRI and are accessible through torch.hub. Learn more at the `Unet hub page <https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/>`_.

6. **Taskonomy** (`Taskonomy`)
   Networks trained for distinct visual tasks, instrumental in discerning relationships between various tasks. Insights can be gained from the `Taskonomy GitHub page <https://github.com/StanfordVL/taskonomy>`_.

7. **CLIP** (`Clip`)
   Multimodal neural networks combining vision and language, trained on diverse (image, text) pairs. Explore more at the `CLIP GitHub repository <https://github.com/openai/CLIP>`_.

8. **CorNet** (`Cornet`)
   Networks that emulate the ventral visual pathway's structure, incorporating recurrent connections. Discover more at the `CORnet GitHub repository <https://github.com/dicarlolab/CORnet>`_.

9.  **Huggingface** (`Huggingface`)
    Features a broad range of advanced language models that deal with text-input. Additional details can be found on the `Huggingface homepage <https://huggingface.co/>`_.

10. **Yolo** (`Yolo`)
    Includes fast, accurate YOLOv5 models for real-time object detection in images and video streams. Further information is on the `YOLO GitHub repository <https://github.com/ultralytics/yolov5>`_.

11. **Toolbox** (`Toolbox`)
    A set of networks that are implemented within Net2Brain itself, providing immediate access to specialized neural network functionalities.


Adding Your Own Models
----------------------
In addition to the provided netsets, **Net2Brain** supports the integration of custom models. For guidance on incorporating your own neural networks, please refer to :ref:`Creating Your Own NetSet <ownnetset>` or :ref:`Using FeatureExtractor with a Custom DNN <customdnn>`.

Discovering Models
------------------
An extensive catalog of models is available for exploration within these netsets. To delve into the full array of models, consult the `Model Taxonomy documentation <https://net2brain.readthedocs.io/en/latest/taxonomy.html>`_, which details the functions to view all available models.

.. _taxonomy.rst: taxonomy.rst
.. _Adding Custom Models documentation: your-link-placeholder
