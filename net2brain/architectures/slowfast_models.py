import torch
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torchvision.transforms._transforms_video import (
        CenterCropVideo,
        NormalizeVideo,
    )

MODELS = {'slow_r50': torch.hub.load,
          'slowfast_r101': torch.hub.load,
          'slowfast_r50': torch.hub.load,
          'x3d_m': torch.hub.load,
          'x3d_s': torch.hub.load,
          'x3d_xs': torch.hub.load}


MODEL_NODES = {'slow_r50': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5'],
               'slowfast_r101': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6'],
               'slowfast_r50': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6'],
               'x3d_l': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5'],
               'x3d_m': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5'],
               'x3d_s': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5'],
               'x3d_xs': ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5']}





class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list



def preprocess_slowfast(video_path):
    """Preprocessing according to slowfast video model

    Args:
        video_path (str): Link to video file

    Returns:
        inputs: tensor in slowfast format
    """
    
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    num_clips = 10
    num_crops = 3
    
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )
    
    clip_duration = (num_frames * sampling_rate)/frames_per_second
    
    start_sec = 0

    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)
    
    # Is Cuda available?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]
    
    return inputs
    

def preprocess_slow(video_path):
    """Preprocessing according to slow video model

    Args:
        video_path (str): Link to video file

    Returns:
        inputs: tensor in slow format
    """
    
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    # Note that this transform is specific to the slow_R50 model.
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)
    
    # Is Cuda Available?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = inputs.to(device)

    return inputs[None, ...]


def preprocess_x3d(video_path, model_name):
    """Preprocessing according to x3d video model

    Args:
        video_path (str): Link to video file
        model_name (str): Name of model

    Returns:
        inputs: tensor in x3d format
    """
    
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    frames_per_second = 30
    model_transform_params  = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }

    # Get transform parameters based on model
    transform_params = model_transform_params[model_name]

    # Note that this transform is specific to the slow_R50 model.
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                )
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
    
    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)
    
    # Is Cuda Available?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = inputs.to(device)

    return inputs[None, ...]


def preprocess(video_path, model_name):
    """Preprocesses image according to the networks needs

    Args:
        image (str/path): path to image
        model_name (str): name of the model (sometimes needes to differenciate between model settings)

    Returns:
        inputs: Tensors of video data
    """
    
    slowfast_models = ['slowfast_16x8_r101_50_50',
                       'slowfast_r101', 'slowfast_r50', 'slowfast_r50_detection']
    
    slow_models = ['slow_r50', 'slow_r50_detection']
    
    x3d_models = ['x3d_m', 'x3d_s', 'x3d_xs']
    
    
    if model_name in slowfast_models:
        inputs = preprocess_slowfast(video_path)
        
    elif model_name in slow_models:
        inputs = preprocess_slow(video_path)
        
    elif model_name in x3d_models:
        inputs = preprocess_x3d(video_path, model_name)
    
    return inputs



