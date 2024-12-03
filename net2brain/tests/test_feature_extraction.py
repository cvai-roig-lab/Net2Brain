from pathlib import Path
import os
import urllib.request
import pytest
from torchvision import models
from torchvision import transforms as trn
import torchextractor as tx
import torch
from net2brain.feature_extraction import FeatureExtractor


# Function to download model checkpoints before the tests run
def download_checkpoints(save_directory):
    checkpoints = {
        "Unet_unet_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fi3BauasaDSSRkeCoZJV6H/Unet_unet_weights.pth",
        "Timm_resnet50_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiWpZ85sv2NeLxFBe8CoiC/Timm_resnet50_weights.pth",
        "Timm_vit_base_patch32_224_in21k_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiKErJtqPauJ5bJABULQAD/Timm_vit_base_patch32_224_in21k_weights.pth",
        "Clip_RN50_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiUX9MtT75Rrbn2VgE4itw/Clip_RN50_weights.pth",
        "Cornet_cornet_z_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fi8XLK2rbyEEELrCahNJvZ/Cornet_cornet_z_weights.pth",
        "Pytorch_deeplabv3_resnet101_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiANgiPXWswvazih59qxMs/Pytorch_deeplabv3_resnet101_weights.pth",
        "Taskonomy_colorization_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fi8vq1Wy7igLci1cfrTvs9/Taskonomy_colorization_weights.pth",
        "Taskonomy_autoencoding_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiL7MRhk4JRo12Ub2pskWX/Taskonomy_autoencoding_weights.pth",
        "Standard_AlexNet_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiPk9zRqKtq7rGfftJaaY7/Standard_AlexNet_weights.pth",
        "Yolo_yolov5l_weights.pth": "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiUWzn75C6pCEKQMXhyLNp/Yolo_yolov5l_weights.pth"
    }

    # Download each checkpoint if not already downloaded
    for filename, url in checkpoints.items():
        file_path = os.path.join(save_directory, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {filename}.")
        else:
            print(f"{filename} already exists, skipping download.")


# Run this function before tests to ensure checkpoints are downloaded
@pytest.fixture(scope="session", autouse=True)
def setup_checkpoints(tmp_path_factory):
    # Define the directory to save checkpoints
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    # Download checkpoints
    download_checkpoints(checkpoint_dir)
    # Return the path to the checkpoints
    return checkpoint_dir


# Define your tests
@pytest.mark.parametrize(
    "netset,model",
    [
        ("Standard", "AlexNet"),
        ("Timm", "vit_base_patch32_224_in21k"),
        ("Timm", "resnet50"),
        ("Unet", "unet"),
        ("Taskonomy", "autoencoding"),
        ("Taskonomy", "colorization"),
        ("Clip", "RN50"),
        ("Cornet", "cornet_z"),
    ],
)
@pytest.mark.parametrize(
    "pretrained",
    [(True), (False)],
)
def test_extractor_outputs(root_path, tmp_path, setup_checkpoints, netset, model, pretrained):
    # Define paths
    imgs_path = root_path / Path("images")

    # Check if pretrained=True, and load from checkpoint if so
    checkpoint_dir = setup_checkpoints
    if pretrained:
        checkpoint_file = os.path.join(checkpoint_dir, f"{netset}_{model}_weights.pth")
        if os.path.exists(checkpoint_file):
            print(f"Loading {netset} {model} weights from {checkpoint_file}")
            fx = FeatureExtractor(model=model, netset=netset, pretrained=False, device='cpu')
            fx.model.load_state_dict(torch.load(checkpoint_file))  # Load from checkpoint
        else:
            pytest.fail(f"Checkpoint for {netset} {model} not found.")
    else:
        fx = FeatureExtractor(model=model, netset=netset, pretrained=False, device='cpu')

    # Extract features
    fx.extract(imgs_path, save_path=tmp_path)

    # Layer consolidation
    fx.consolidate_per_layer()

    output_files = list(tmp_path.iterdir())

    # Assert output files are as expected
    assert len(output_files) > 1

    return


@pytest.mark.parametrize(
    "netset,model",
    [
        ("Audio", "PANNS_Cnn10"),
        ("Audio", "PANNS_Cnn14"),
        ("Audio", "PANNS_Cnn6"),
        ("Audio", "PANNS_DaiNet19"),
        ("Audio", "PANNS_LeeNet11"),
        ("Audio", "PANNS_MobileNetV1"),
        ("Audio", "PANNS_MobileNetV2"),
        ("Audio", "PANNS_Res1dNet51"),
        ("Audio", "PANNS_ResNet22"),
        ("Audio", "MIT/ast-finetuned-audioset-10-10-0.448-v2"),
        ("Audio", "MIT/ast-finetuned-audioset-16-16-0.442"),
    ],
)
@pytest.mark.parametrize(
    "pretrained",
    [True, ],
)
def test_extractor_outputs_audio(root_path, tmp_path, setup_checkpoints, netset, model, pretrained):
    # Define paths
    audios_path = root_path / Path("audios")

    fx = FeatureExtractor(model=model, netset=netset, pretrained=pretrained, device='cpu')

    # Extract features
    fx.extract(audios_path, save_path=tmp_path)

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
    model = models.alexnet(
        pretrained=False)  # This one exists in the toolbox as well, it is just supposed to be an example!

    ## Define extractor (Note: NO NETSET NEEDED HERE)
    fx = FeatureExtractor(model=model, device='cpu')

    # Run extractor
    fx.extract(image_path, layers_to_extract=['features.0', 'features.3', 'features.6', 'features.8', 'features.10'])

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
    model = models.alexnet(
        pretrained=False)  # This one exists in the toolbox as well, it is just supposed to be an example!

    ## Define extractor (Note: NO NETSET NEEDED HERE)
    fx = FeatureExtractor(model=model, device='cpu', preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                          extraction_function=my_extactor)

    # Run extractor
    fx.extract(image_path, layers_to_extract=['features.0', 'features.3', 'features.6', 'features.8', 'features.10'])

    return
