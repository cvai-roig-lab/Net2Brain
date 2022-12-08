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
    fe = FeatureExtractor(model, netset)
    assert fe.model == model, "loaded model different than the one requested"
    assert fe.transforms is not None, "transforms not loaded"
    assert fe.preprocess is not None, "preprocess not loaded"
    assert fe._extractor is not None, "extractor not loaded"
    assert fe._feature_cleaner is not None, "feature cleaner not loaded"
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
        ("dataset",dict), ("pt", None), ("np", None)
    ]
)
def test_extractor_outputs(netset, model, save_format, output_type):
    # Define paths
    imgs_path = Path('./net2brain/tests/images')
    save_path = Path('./net2brain/tests/images/tmp')
    save_path.mkdir(parents=True, exist_ok=True)

    # Extract features
    fx = FeatureExtractor(model, netset)
    feats = fx.extract(imgs_path, save_format=save_format, save_path=save_path)
    output_files = [d for d in save_path.iterdir()]

    # Assert return type is as expected
    assert type(feats) == output_type
    
    # Assert output files are as expected
    if save_format == 'dataset':
        assert len(output_files) == len(fx.layers_to_extract)
        assert feats[list(feats.keys())[0]].measurements.shape[0] == 2
    else:
        assert len(output_files) == 2

    # Remove temporary files
    shutil.rmtree(save_path)

    return
    
