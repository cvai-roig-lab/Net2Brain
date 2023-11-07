from pathlib import Path

import pytest
from torchvision import models
from torchvision import transforms as T

from net2brain.feature_extraction import FeatureExtractor


@pytest.mark.parametrize(
    "netset,model",
    [
        ("standard", "AlexNet"),
        ("timm", "vit_base_patch32_224_in21k"),
        ("pytorch", "deeplabv3_resnet101"),
        ("unet", "unet"),
        ("taskonomy", "autoencoding"),
        ("pyvideo", "slowfast_r50"),
        ("clip", "RN50"),
        ("cornet", "cornet_z"),
    ],
)
def test_load_netset_model(netset, model):
    fx = FeatureExtractor(model, netset, pretrained=True)
    assert fx.model_name == model, "loaded model different than the one requested"
    assert fx.preprocess is not None, "preprocess not loaded"
    assert fx._extractor is not None, "extractor not loaded"
    assert fx.layers_to_extract() is not None, "No layers to extract"
    assert fx._features_cleaner is not None, "feature cleaner not loaded"
    return


@pytest.mark.parametrize(
    "netset,model",
    [
        ("Standard", "AlexNet"),
        ("Timm", "vit_base_patch32_224_in21k"),
        ("Timm", "resnet50"),
        ("Pytorch", "deeplabv3_resnet101"),
        ("Unet", "unet"),
        ("Yolo, yolo51"),
        ("Taskonomy", "autoencoding"),
        ("Taskonomy", "colorization"),
        ("Pyvideo", "slowfast_r50"),
        ("Clip", "RN50"),
        ("Cornet", "cornet_z"),
    ],
)
@pytest.mark.parametrize(
    "pretrained",
    [(True), (False)],
)

def test_extractor_outputs(
    root_path, tmp_path, netset, model, pretrained
):
    # Define paths
    imgs_path = root_path / Path("images")

    # Extract features
    fx = FeatureExtractor(model, netset, imgs_path, pretrained=pretrained, save_path=tmp_path)
    fx.extract()

    # Layer consolidation
    fx.consolidate_per_layer()

    output_files = list(tmp_path.iterdir())

    # Assert output files are as expected
    assert len(output_files) == 2

    return


# @pytest.mark.parametrize(
#     "input_layers,output_layers",
#     [(None, ["layer1", "layer2", "layer3", "layer4"]), (["layer1"], ["layer1"])],
# )
# def test_feature_extraction_layers(input_layers, output_layers):
#     fx = FeatureExtractor("ResNet50", "standard", layers_to_extract=input_layers)
#     #assert fx.layers_to_extract == output_layers
#     return


def test_feature_extraction_custom(root_path, tmp_path):
    # Define paths
    imgs_path = root_path / "images"

    # Define model and transforms
    model = models.alexnet(pretrained=True)
    # model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    transforms = T.Compose(
        [
            T.Resize((224, 224)),  # transform images if needed
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Define extractor
    layers = ["features.0", "features.1"]
    fx = FeatureExtractor(
        model, transforms=transforms, layers_to_extract=layers, device="cpu"
    )

    # Run extractor
    fx.extract(dataset_path=imgs_path, save_format="pt", save_path=tmp_path)

    # Test files got created
    output_files = list(tmp_path.iterdir())
    assert len(output_files) == 2

    return


def test_missing_netset():
    with pytest.raises(NameError):
        FeatureExtractor("alexnet")


# @pytest.mark.parametrize(
#     "netset,model", [(i, x) for i in AVAILABLE_NETWORKS for x in AVAILABLE_NETWORKS[i]]
# )
# def test_all_models(netset, model):
#     fx = FeatureExtractor(model, netset, pretrained=False, device="cpu")
#     assert fx.model_name == model, "loaded model different than the one requested"
#     assert fx.preprocess is not None, "preprocess not loaded"
#     assert fx._extractor is not None, "extractor not loaded"
#     assert fx._features_cleaner is not None, "feature cleaner not loaded"
