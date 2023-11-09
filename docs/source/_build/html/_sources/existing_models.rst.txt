Model Zoo
================

**Net2Brain** is an all-in-one solution that offers access to approximately 600 pre-trained neural networks across various visual tasks. These networks enable comprehensive experimental comparisons between human brain activity and artificial neural network activations. Moreover, users have the flexibility to incorporate their own models, making the toolkit highly adaptable and limitless in terms of architecture types.


What is a Netset?
-----------------
A Netset is a curated collection of neural network models from the same family but with potentially different training protocols or datasets. This organization allows for clear differentiation when multiple versions of a model, such as ResNet50, are available from various sources, ensuring clarity and ease of use.


Available Netsets
-----------------
**Net2Brain** facilitates exploration and utilization of a vast range of Deep Neural Networks (DNNs) through its diverse netsets, which are libraries of different pre-trained models:

1. **Standard torchvision** (`standard`)
   A compendium of torchvision models catering to a spectrum of tasks from image classification to video classification. Detailed information can be found on the `torchvision models page <https://pytorch.org/vision/stable/models.html>`_.

2. **Timm** (`timm`)
   A library containing a rich array of advanced computer vision models developed by Ross Wightman. More details are available on the `Timm GitHub repository <https://github.com/rwightman/pytorch-image-models#models>`_.

3. **PyTorch Hub** (`pytorch`)
   This netset includes models for a variety of visual tasks accessible through the torch.hub API, not encompassed by torchvision. For more, see the `PyTorch Hub documentation <https://pytorch.org/docs/stable/hub.html>`_.

4. **Unet** (`unet`)
   Unet models are specialized for abnormality segmentation in brain MRI and are accessible through torch.hub. Learn more at the `Unet hub page <https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/>`_.

5. **Taskonomy** (`taskonomy`)
   Networks trained for distinct visual tasks, instrumental in discerning relationships between various tasks. Insights can be gained from the `Taskonomy GitHub page <https://github.com/StanfordVL/taskonomy>`_.

6. **Slowfast** (`pyvideo`)
   These are top-tier video classification models from the Kinetics 400 dataset, available through torch.hub. Visit the `PyTorchVideo GitHub repository <https://github.com/facebookresearch/pytorchvideo>`_ for more.

7. **YOLO** (`yolo`)
   YOLO models are renowned for their speed and accuracy in real-time object detection. Further information is on the `YOLO GitHub repository <https://github.com/ultralytics/yolov5>`_.

8. **CLIP** (`clip`)
   Multimodal neural networks combining vision and language, trained on diverse (image, text) pairs. Explore more at the `CLIP GitHub repository <https://github.com/openai/CLIP>`_.

9. **CorNet** (`cornet`)
   Networks that emulate the ventral visual pathway's structure, incorporating recurrent connections. Discover more at the `CORnet GitHub repository <https://github.com/dicarlolab/CORnet>`_.

10. **Detectron2** (`detectron2`)
    A system from Facebook AI Research that implements leading object detection algorithms. Explore the `Detectron2 GitHub page <https://github.com/facebookresearch/Detectron>`_ for more.

11. **VISSL** (`vissl`)
    A collection of self-supervision approaches with reference implementations. Delve into the `VISSL GitHub repository <https://github.com/facebookresearch/vissl>`_.

Adding Your Own Models
----------------------
In addition to the provided netsets, **Net2Brain** supports the integration of custom models. For guidance on incorporating your own neural networks, please refer to `Creating Your Own NetSet <https://net2brain.readthedocs.io/en/latest/adding_own_netsets.html>`_ or `Using FeatureExtractor with a Custom DNN <https://net2brain.readthedocs.io/en/latest/adding_own_netsets.html>`_.

Discovering Models
------------------
An extensive catalog of models is available for exploration within these netsets. To delve into the full array of models, consult the `Model Taxonomy documentation <taxonomy.rst>`_, which details the functions to view all available models.

.. _taxonomy.rst: taxonomy.rst
.. _Adding Custom Models documentation: your-link-placeholder
