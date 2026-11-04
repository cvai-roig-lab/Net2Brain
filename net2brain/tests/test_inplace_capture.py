"""Regression tests: saved features must be a layer's output at the moment the layer ran.

Many models overwrite a layer's output later in the forward pass with an in-place operation,
e.g. ``nn.ReLU(inplace=True)`` after a conv layer or ``out += identity`` after the last
BatchNorm of a ResNet block. Storing a reference to the output instead of a copy made
Net2Brain save the overwritten values under the earlier layer's name.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from net2brain.architectures.netsetbase import copy_module_output, create_extractor
from net2brain.feature_extraction import FeatureExtractor


class InPlaceNet(nn.Module):
    """Tiny model with both kinds of in-place overwrite."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)  # overwrites the conv output
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        identity = self.relu(self.conv(x))
        out = self.bn(identity)
        out += identity  # overwrites the bn output
        return out


def test_create_extractor_keeps_outputs_overwritten_in_place():
    torch.manual_seed(0)
    model = InPlaceNet().eval()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        conv_out = model.conv(x)
        bn_out = model.bn(torch.relu(conv_out))

    _, features = create_extractor(model, ["conv", "bn"])(x)

    assert (features["conv"] < 0).any()
    torch.testing.assert_close(features["conv"], conv_out)
    torch.testing.assert_close(features["bn"], bn_out)


def test_copy_module_output_keeps_structure_and_copies_tensors():
    Pair = namedtuple("Pair", ["tensor", "number"])
    tensor = torch.randn(3)
    expected = tensor.clone()
    output = (tensor, [tensor, {"key": tensor}], Pair(tensor, 1), None)

    copied = copy_module_output(output)
    tensor.zero_()  # an in-place change after capture must not reach the copy

    assert type(copied) is tuple and type(copied[1]) is list and type(copied[2]) is Pair
    for value in (copied[0], copied[1][0], copied[1][1]["key"], copied[2].tensor):
        torch.testing.assert_close(value, expected)
    assert copied[2].number == 1 and copied[3] is None


def test_alexnet_conv_features_are_not_relu_outputs(root_path, tmp_path):
    """The reported case: AlexNet conv layers were saved after the following in-place ReLU."""
    images_path = root_path / Path("images")
    fx = FeatureExtractor(model="AlexNet", netset="Standard", pretrained=False, device="cpu")

    # Compute the expected output before extraction attaches hooks to the model.
    image_file = sorted(p for p in images_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]
    x = fx.netset.image_preprocessing(Image.open(image_file).convert("RGB"), "AlexNet", "cpu")
    with torch.no_grad():
        conv1 = fx.model.features[0](x).numpy()

    fx.extract(images_path, save_path=tmp_path, layers_to_extract=["features.0", "features.1"],
               consolidate_per_layer=False)
    saved = np.load(tmp_path / f"{image_file.name.split('.')[0]}.npz")

    assert (saved["features.0"] < 0).any()
    np.testing.assert_allclose(saved["features.0"], conv1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(saved["features.1"], np.maximum(conv1, 0), rtol=1e-5, atol=1e-5)


def test_resnet_bn_before_residual_add():
    """`out += identity` in the ResNet block overwrites the output of bn2."""
    fx = FeatureExtractor(model="ResNet18", netset="Standard", pretrained=False, device="cpu")
    model, block = fx.model, fx.model.layer1[0]
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)

    # Compute the expected output before extraction attaches hooks to the model.
    with torch.no_grad():
        stem = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        expected = block.bn2(block.conv2(block.relu(block.bn1(block.conv1(stem)))))

    features = fx.netset.extraction_function(x, ["layer1.0.bn2"])

    torch.testing.assert_close(features["layer1.0.bn2"], expected)
