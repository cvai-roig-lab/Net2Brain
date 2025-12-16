import os
import warnings
import json
from platformdirs import user_cache_dir
from pathlib import Path
from math import sqrt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import torch
import torch.nn.functional as F
import torchextractor as tx

from mmaction.utils import register_all_modules

register_all_modules()
from mmaction.registry import MODELS as MMA_MODELS
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmaction.datasets.transforms.loading import DecordInit, SampleFrames, DecordDecode
from mmaction.datasets.transforms.processing import Resize, CenterCrop
from mmaction.datasets.transforms.formatting import FormatShape, PackActionInputs
from mmengine.dataset.base_dataset import Compose as MMECompose
from mmengine.dataset.utils import pseudo_collate

from .netsetbase import NetSetBase
from .shared_functions import download_to_path, download_github_folder


class MMAction(NetSetBase):
    """
    """

    def __init__(self, model_name, device):
        self.supported_data_types = ['video']
        self.netset_name = "MMAction"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/mmaction.json")

    def get_preprocessing_function(self, data_type):
        if data_type == 'video':
            return self.video_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")

    def get_feature_cleaner(self, data_type):
        if data_type == 'video':
            return self.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")

    def load_video_data(self, data_path):
        return [data_path]

    def get_model(self, pretrained):

        # Load the JSON file
        with open(self.config_path, 'r') as file:
            data = json.load(file)

        # Check if model_name exists in the data
        if self.model_name not in data:
            raise ValueError(f"{self.model_name} not found in the configuration file.")

        # Retrieve the attributes for the given model_name
        model_entry = data[self.model_name]
        checkpoint_url = model_entry["download_links"]["url_to_checkpoint"]
        config_url = model_entry["download_links"]["url_to_config"]
        self.layers = model_entry["extractor"]["layers"]
        self.extractor_settings = model_entry["extractor"]
        self.preprocessor_settings = model_entry["preprocessor"]

        cache_dir = Path(user_cache_dir("net2brain"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = cache_dir / "mma_checkpoints" / f"{self.model_name}.pth"
        if not checkpoint_path.exists():
            print(f"Downloading checkpoint for {self.model_name} to cache ...")
            # TODO: check where other net2brain checkpoints are stored
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            download_to_path(checkpoint_url, checkpoint_path)

        config_folder = cache_dir / "mma_configs"
        if not config_folder.exists():
            print(f"Config files for MMAction models not found in cache, downloading ...")
            download_github_folder(
                owner="open-mmlab",
                repo="mmaction2",
                repo_path="configs",
                out_dir=config_folder
            )
            cfg_to_fix = config_folder / "recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb.py"
            # fix issue in one config file - missing line 7
            lines = cfg_to_fix.read_text().splitlines(keepends=True)
            lines.insert(6, "        feature_shape='NHWC',\n")
            cfg_to_fix.write_text(''.join(lines))
        config_path = config_folder / config_url

        cfg = Config.fromfile(config_path)
        self.loaded_model = MMA_MODELS.build(cfg.model)
        load_checkpoint(self.loaded_model, str(checkpoint_path))

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()

        return self.loaded_model

    def video_preprocessing(
            self,
            video_path,
            model,
            device,
    ):

        (clip_len,
         frame_interval,
         resize_size,
         crop_size,
         format_shape) = (self.preprocessor_settings["clip_len"],
                          self.preprocessor_settings["frame_interval"],
                          self.preprocessor_settings["resize_size"],
                          self.preprocessor_settings["crop_size"],
                          self.preprocessor_settings["format_shape"] if
                          "format_shape" in self.preprocessor_settings else "NCTHW")
        video = {"filename": video_path, "start_index": 0, "modality": "RGB"}
        video = DecordInit()(video)
        frame_interval *= video["avg_fps"] / 30  # 30 fps is the expected frame rate
        num_clips = round(video["total_frames"] / (clip_len * frame_interval))
        transform = MMECompose(
            [
                (
                    SampleFrames(
                        clip_len=clip_len,
                        frame_interval=round(frame_interval),
                        num_clips=num_clips,
                        out_of_bound_opt="repeat_last",
                        test_mode=True,
                    )
                ),
                DecordDecode(),
                Resize(scale=(-1, resize_size)),
                CenterCrop(crop_size=crop_size),
                FormatShape(input_format=format_shape),
                PackActionInputs(),
            ]
        )
        video = transform(video)
        # we want to loop over clips in the extraction function, so we need to reshape the input
        if format_shape == "NCTHW":
            # we're only allowing center_crop, so batch_dim (=crops_dim) = 1
            video["inputs"] = video["inputs"].unsqueeze(1)
        elif format_shape == "NCHW":
            # batch_dim will be frames per clip now (still no multi-crop)
            video["inputs"] = video["inputs"].reshape(
                (num_clips, -1) + video["inputs"].shape[1:]).float()
        return video

    def clean_extracted_features(self, features):
        # in mma slowfast has separate keys for slow and fast, so it doesn't need special handling
        clean_dict = {}
        for A_key, subtuple in features.items():
            if isinstance(subtuple, (list, tuple)):
                tensor_elements = [elem for elem in subtuple if torch.is_tensor(elem)]
                if len(tensor_elements) == 1:
                    clean_dict[A_key] = tensor_elements[0].cpu()
                else:
                    new_names = [A_key + f"_{counter}" for counter in range(len(tensor_elements))]
                    for counter, key in enumerate(new_names):
                        clean_dict[key] = tensor_elements[counter].cpu()
            else:
                clean_dict[A_key] = subtuple.cpu()
        return clean_dict

    def extraction_function(self, data, layers_to_extract=None, agg_frames='across_clips'):
        # note: this function does not support multiple spatial crops
        if layers_to_extract == "top_level" and self.layers is not None:
            layers_to_extract = "json"
        elif layers_to_extract == "json":
            raise NotImplementedError("Use the `top_level` option instead of `json` for mmaction "
                                      "models. JSON-defined layers will still be used if existing.")
        layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)
        skips = self.extractor_settings["skips"]
        layers = [layer for layer in layers if layer not in skips]
        normalizer = self.loaded_model.data_preprocessor
        data = pseudo_collate([data])
        data = normalizer(data)["inputs"].squeeze(0)  # squeeze out the fake batch
        device = data.device
        data = data.cpu()
        if data.ndim == 6:
            format_shape = "NCTHW"  # N is broken in 2: n_clips, n_crops(=1)
        elif data.ndim == 5:
            format_shape = "NCHW"  # N is broken in 2: n_clips, T
        else:
            raise ValueError(
                f"Expected 5D or 6D preprocessed input tensor, got {data.ndim}D tensor")
        n_clips = data.shape[0]
        features_all_clips = {}
        for i in range(n_clips):  # sacrifice speed to avoid increasing batch size
            extractor_model = tx.Extractor(self.loaded_model, layers)
            try:
                out, features = extractor_model(data[i].unsqueeze(0).to(device), stage='head')
                # this unsqueeze is needed because of a peculiarity that some squeezing is done
                # before passing to the model (otherwise batch dim would be just the crops/frames)
            except RuntimeError:
                # pad the input such as that data[i].shape[0] is divisible by 8
                # this is needed for models like TSM that handle "segments" internally from batchdim
                div_by = 8
                pad_size = (div_by - data[i].size(0) % div_by) % div_by
                if pad_size > div_by / 2:
                    padded_data = data[i][: -(data[i].size(0) % div_by)]
                else:
                    padded_data = F.pad(data[i], (0, 0, 0, 0, 0, 0, 0, pad_size))
                out, features = extractor_model(padded_data.unsqueeze(0).to(device), stage='head')
            del out
            features = self.clean_extracted_features(features)
            main_format = self.extractor_settings["main_format"]
            format_exceptions = self.extractor_settings["format_exceptions"]
            for key in features:
                # Convert to a *CONSISTENT* format of features for all models:
                # (1, CLIPS, TIMEPOINTS, C, H, W)
                value = features[key].detach().cpu()
                if format_shape == "NCTHW":
                    value = value.squeeze(0)
                if key in format_exceptions:
                    key_format = format_exceptions[key]
                else:
                    key_format = main_format
                if key_format == 'T,C,H,W':
                    pass
                elif key_format == 'T,H,W,C':
                    value = value.permute(0, 3, 1, 2)
                elif key_format == 'C,T,H,W':
                    value = value.transpose(0, 1)
                elif key_format == 'T,HW+1,C':
                    T = value.shape[0]
                    H = W = int(sqrt(value.shape[1] - 1))
                    C = value.shape[2]
                    value = value[:, 1:, :].reshape(T, H, W, C).permute(0, 3, 1, 2)
                elif key_format == 'HW+1,T,C':
                    T = value.shape[1]
                    H = W = int(sqrt(value.shape[0] - 1))
                    C = value.shape[2]
                    value = value[1:, :, :].reshape(H, W, T, C).permute(2, 3, 0, 1)
                elif key_format == 'THW+1,C':
                    T = self.extractor_settings["t_same"]
                    H = W = int(sqrt((value.shape[0] - 1) / T))
                    C = value.shape[1]
                    value = value[1:, :].reshape(T, H, W, C).permute(0, 3, 1, 2)
                elif key_format == 'THW,C':
                    T = self.extractor_settings["t_same"]
                    H = W = int(sqrt(value.shape[0] / T))
                    C = value.shape[1]
                    value = value.reshape(T, H, W, C).permute(0, 3, 1, 2)
                elif key_format == 'HWT+1,C':
                    T = self.extractor_settings["t_same"]
                    H = W = int(sqrt((value.shape[0] - 1) / T))
                    C = value.shape[1]
                    value = value[1:, :].reshape(H, W, T, C).permute(2, 3, 0, 1)
                elif key_format == 'T,C':
                    value = value.unsqueeze(-1).unsqueeze(-1)  # add H and W dims
                elif key_format == 'T,1,C':
                    value = value.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # add H and W dims
                elif key_format == 'C':
                    value = value.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # add T and HW dims
                value = value.unsqueeze(0)  # batch dim
                value = value.unsqueeze(1)  # clips dim (will concat on this)
                if value.ndim != 6:
                    raise ValueError(
                        f"Expected 6D tensor, got {value.ndim}D tensor for key {key}. "
                        f"Shape: {value.shape}"
                    )
                if key not in features_all_clips:
                    features_all_clips[key] = value
                else:
                    features_all_clips[key] = torch.concat([features_all_clips[key], value], dim=1)
                    if agg_frames == 'across_clips':
                        features_all_clips[key] = features_all_clips[key].mean(1, keepdim=True)
                    elif agg_frames == 'within_clips':
                        features_all_clips[key] = features_all_clips[key].mean(2, keepdim=True)
                    elif agg_frames == 'all':
                        features_all_clips[key] = features_all_clips[key].mean(1, keepdim=True)
                        features_all_clips[key] = features_all_clips[key].mean(2, keepdim=True)
        return features_all_clips
