import warnings, numbers, os
from .netsetbase import NetSetBase
from .shared_functions import load_from_json
import torchextractor as tx
import torch
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from net2brain.utils.torchvideo import CenterCropVideo, NormalizeVideo

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



class Pyvideo(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['video']
        self.netset_name = "Pyvideo"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/pyvideo.json")


    def get_preprocessing_function(self, data_type):
        if data_type == 'video':
            return self.video_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_feature_cleaner(self, data_type):
        if data_type == 'video':
            return Pyvideo.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

        

    def get_model(self, pretrained):

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.loaded_model = model_attributes["model_function"]('facebookresearch/pytorchvideo', 
                                                               self.model_name, 
                                                               pretrained=pretrained)
        
        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()

        return self.loaded_model


    def preprocess_slowfast(self, video_path, device):
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

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to(device)[None, ...] for i in inputs]
        
        return inputs
        

    def preprocess_slow(self, video_path, device):
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
        

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = inputs.to(device)

        return inputs[None, ...]


    def preprocess_x3d(self, video_path, model_name, device):
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
        

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = inputs.to(device)

        return inputs[None, ...]




    def video_preprocessing(self, video_path, model_name, device):
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
            inputs = self.preprocess_slowfast(video_path, device)
            
        elif model_name in slow_models:
            inputs = self.preprocess_slow(video_path, device)
            
        elif model_name in x3d_models:
            inputs = self.preprocess_x3d(video_path, model_name, device)
        
        return inputs
    




    def clean_extracted_features(self, features):
        
        clean_dict = {}
        for A_key, subtuple in features.items():
            keys = [A_key + "_slow", A_key + "_fast"]

            try:  # if subdict is a list of two values
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
            except:
                clean_dict.update({A_key: subtuple.cpu()})

        return clean_dict
    

    def load_video_data(self, data_path):
        return [data_path]
    
    
    def extraction_function(self, data, layers_to_extract=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        # Extract actual features
        _, features = self.extractor_model(data)

        return features

         
