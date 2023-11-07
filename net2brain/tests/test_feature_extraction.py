from pathlib import Path

import pytest
from torchvision import models
from torchvision import transforms as T
from torchvision import models
from torchvision import transforms as T
from torchvision import transforms as trn
import torchextractor as tx

from net2brain.feature_extraction import FeatureExtractor


@pytest.mark.parametrize(
    "netset,model",
    [
        ("Standard", "AlexNet"),
        ("Timm", "vit_base_patch32_224_in21k"),
        ("Pytorch", "deeplabv3_resnet101"),
        ("Unet", "unet"),
        ("Taskonomy", "autoencoding"),
        ("Pyvideo", "slowfast_r50"),
        ("Clip", "RN50"),
        ("Cornet", "cornet_z"),
    ],
)
def test_load_netset_model(netset, model):
    fx = FeatureExtractor(model, netset, pretrained=True)
    assert fx.model_name == model, "loaded model different than the one requested"
    assert fx.preprocessor is not None, "preprocess not loaded"
    assert fx.extraction_function is not None, "extractor not loaded"
    assert fx.layers_to_extract() is not None, "No layers to extract"
    assert fx.feature_cleaner is not None, "feature cleaner not loaded"
    return


@pytest.mark.parametrize(
    "netset,model",
    [
        ("Standard", "AlexNet"),
        ("Timm", "vit_base_patch32_224_in21k"),
        ("Timm", "resnet50"),
        ("Pytorch", "deeplabv3_resnet101"),
        ("Unet", "unet"),
        ("Yolo", "yolov51"),
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
    fx = FeatureExtractor(model, netset, pretrained=pretrained, save_path=tmp_path)
    fx.extract(imgs_path)

    # Layer consolidation
    fx.consolidate_per_layer()

    output_files = list(tmp_path.iterdir())

    # Assert output files are as expected
    assert len(output_files) > 1

    return





def test_own_model(root_path, tmp_path):

    # Define iamge path
    image_path = root_path / Path("images")

    # Define a model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # This one exists in the toolbox as well, it is just supposed to be an example!

    ## Define extractor (Note: NO NETSET NEEDED HERE)
    fx = FeatureExtractor(model=model, device='cpu')

    # Run extractor
    fx.extract(image_path, layers_to_extract=['layer1', 'layer2', 'layer3', 'layer4'])

    return



def test_with_own_functions(root_path, tmp_path):

    def my_preprocessor(image, model_name, device):
        """
        Args:
            image (Union[Image.Image, List[Image.Image]]): A PIL Image or a list of PIL Images.
            model_name (str): The name of the model, used to determine specific preprocessing if necessary.
            device (str): The device to which the tensor should be transferred ('cuda' for GPU, 'cpu' for CPU).

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The preprocessed image(s) as PyTorch tensor(s).
        """

        print("I am using my own preprocessor")
        transforms = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transforms(image).unsqueeze(0)
        if device == 'cuda':
            img_tensor = img_tensor.cuda()

        return img_tensor


    def my_extactor(preprocessed_data, layers_to_extract, model):

        print("I am using my own extractor")

        # Create a extractor instance
        extractor_model = tx.Extractor(model, layers_to_extract)
        
        # Extract actual features
        _, features = extractor_model(preprocessed_data)

        return features


    def my_cleaner(features):
        print("I am using my own cleaner which does not do anything")
        return features
    

    # Define iamge path
    image_path = root_path / Path("images")

    # Define a model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # This one exists in the toolbox as well, it is just supposed to be an example!

    ## Define extractor (Note: NO NETSET NEEDED HERE)
    fx = FeatureExtractor(model=model, device='cpu', preprocessor=my_preprocessor, feature_cleaner=my_cleaner, extraction_function=my_extactor)

    # Run extractor
    fx.extract(image_path, layers_to_extract=['layer1', 'layer2', 'layer3', 'layer4'])

    return



def test_missing_netset():
    with pytest.raises(NameError):
        FeatureExtractor("alexnet")

