from pathlib import Path
import pytest
import shutil

from net2brain.feature_extraction import FeatureExtractor


@pytest.mark.parametrize(
    "netset,model", [
        ("standard","AlexNet"), ("timm", "vit_base_patch32_224_in21k"), 
        ("pytorch", "deeplabv3_resnet101"), ("unet", "unet"),
        ("taskonomy", "autoencoding"), ("pyvideo", "slowfast_r50"),
        ('clip', 'RN50'), ("cornet", "cornet_z")

    ]
)
def test_load_netset_model(netset, model):
    fx = FeatureExtractor(model, netset)
    assert fx.model_name == model, "loaded model different than the one requested"
    #assert fe.transforms is not None, "transforms not loaded"
    assert fx.preprocess is not None, "preprocess not loaded"
    assert fx._extractor is not None, "extractor not loaded"
    assert fx._features_cleaner is not None, "feature cleaner not loaded"
    return


@pytest.mark.parametrize(
    "netset,model", [
        ("standard","AlexNet"), ("timm", "vit_base_patch32_224_in21k"), 
        ("pytorch", "deeplabv3_resnet101"), ("unet", "unet"),
        ("taskonomy", "autoencoding"), ("pyvideo", "slowfast_r50"),
        ('clip', 'RN50'), ("cornet", "cornet_z")

    ]
)
@pytest.mark.parametrize(
    "save_format,output_type", [
        ("dataset",dict), ("pt", None), ("npz", None)
    ]
)
def test_extractor_outputs(netset, model, save_format, output_type):
    # Define paths
    imgs_path = Path('./net2brain/tests/images')
    save_path = Path(f'./net2brain/tests/images/tmp/{model}/{save_format}')
    save_path.mkdir(parents=True, exist_ok=True)

    # Extract features
    fx = FeatureExtractor(model, netset)
    feats = fx.extract(imgs_path, save_format=save_format, save_path=save_path)
    output_files = [d for d in save_path.iterdir()]

    # Assert return type is as expected
    if output_type == None:
        assert type(feats) is type(output_type)
    else:
        assert type(feats) == output_type
    
    # Assert output files are as expected
    print(fx.layers_to_extract)
    if save_format == 'dataset':
        if "slowfast" in model:
            assert len(output_files) == len(fx.layers_to_extract) * 2
        else:
            print(fx.layers_to_extract)
            assert len(output_files) == len(fx.layers_to_extract)
            assert feats[list(feats.keys())[0]].measurements.shape[0] == 2
    else:
        assert len(output_files) == 2

    # Remove temporary files
    shutil.rmtree(save_path)

    return
    
@pytest.mark.parametrize(
    "input_layers,output_layers", [
        (None, ["layer1", "layer2", "layer3", "layer4"]), 
        (["layer1"], ["layer1"])
    ]
)
def test_feature_extraction_layers(input_layers, output_layers):
    fx = FeatureExtractor('ResNet50', 'standard', layers_to_extract=input_layers)
    assert fx.layers_to_extract == output_layers
    return
