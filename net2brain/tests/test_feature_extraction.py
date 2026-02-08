from pathlib import Path
import pytest
from torchvision import models
from torchvision import transforms as trn
import torchextractor as tx
from net2brain.feature_extraction import FeatureExtractor


@pytest.mark.parametrize(
    "netset,model",
    [
        ("Standard", "AlexNet"),
        ("Timm", "vit_base_patch32_224_in21k"),
        ("Timm", "resnet50"),
        ("Taskonomy", "autoencoding"),
        ("Taskonomy", "colorization"),
        ("Clip", "RN50"),
        ("Cornet", "cornet_z"),
    ],
)
def test_extractor_outputs(root_path, tmp_path, netset, model):
    # Define paths
    imgs_path = root_path / Path("images")

    # Create extractor with random initialization
    fx = FeatureExtractor(model=model, netset=netset, pretrained=False, device='cpu')

    # Extract features
    fx.extract(imgs_path, save_path=tmp_path)

    # Layer consolidation
    fx.consolidate_per_layer()

    output_files = list(tmp_path.iterdir())

    # Assert output files are as expected
    assert len(output_files) > 1


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
        ("Audio", "PANNS_ResNet22")
    ],
)
def test_extractor_outputs_audio(root_path, tmp_path, netset, model):
    # Define paths
    audios_path = root_path / Path("audios")

    # Use pretrained=True for audio models (these download from libraries with caching)
    fx = FeatureExtractor(model=model, netset=netset, pretrained=True, device='cpu')

    # Extract features
    fx.extract(audios_path, save_path=tmp_path)

    # Layer consolidation
    fx.consolidate_per_layer()

    output_files = list(tmp_path.iterdir())

    # Assert output files are as expected
    assert len(output_files) > 1


def test_own_model(root_path, tmp_path):
    # Define image path
    image_path = root_path / Path("images")

    # Define a model
    model = models.alexnet(pretrained=False)

    # Define extractor (Note: NO NETSET NEEDED HERE)
    fx = FeatureExtractor(model=model, device='cpu')

    # Run extractor
    fx.extract(image_path, layers_to_extract=['features.0', 'features.3', 'features.6', 'features.8', 'features.10'])


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

    # Define image path
    image_path = root_path / Path("images")

    # Define a model
    model = models.alexnet(pretrained=False)

    # Define extractor (Note: NO NETSET NEEDED HERE)
    fx = FeatureExtractor(model=model, device='cpu', preprocessor=my_preprocessor, 
                          feature_cleaner=my_cleaner, extraction_function=my_extactor)

    # Run extractor
    fx.extract(image_path, layers_to_extract=['features.0', 'features.3', 'features.6', 'features.8', 'features.10'])