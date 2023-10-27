from architectures.netsetbase import NetSetBase
import os
import numpy as np
from tqdm import tqdm
from torchvision.models.feature_extraction import get_graph_node_names
from architectures.pytorch_models import Standard
from architectures.timm_models import Timm
from architectures.taskonomy_models import Taskonomy
import torchextractor as tx

from PIL import Image
import cv2
import librosa

import warnings
warnings.filterwarnings("ignore")

# FeatureExtractor class
class FeatureExtractor:
    def __init__(self, model_name, netset, data_path, device, pretrained=True, save_path=None):
        # Parameters
        self.model_name = model_name
        self.netset_name = netset
        self.netset = NetSetBase.initialize_netset(self.model_name, netset)
        self.data_path = data_path
        self.device = device
        self.pretrained = pretrained
        self.save_path = save_path or os.getcwd()

        # Initiate netset-based functions
        self.model = self.netset.get_model(self.pretrained)


    def extract(self, layers_to_extract=None):
        # Iterate over all files in the given data_path

        for data_file in tqdm(os.listdir(self.data_path)):

            full_path = os.path.join(self.data_path, data_file)
            
            # Detect data type
            data_loader = self._get_dataloader(full_path)
            self.data_type = data_loader.TYPE
            if self.data_type not in self.netset.supported_data_types:
                raise ValueError(f"{self.netset_name} does not support data type: {self.data_type}")
            
            # Load data
            data = data_loader.load_data(full_path)

            # Select preprocessor
            self.preprocessor = self.netset.get_preprocessing_function(self.data_type)

            # Preprocess data
            preprocessed_data = self.preprocessor(self.netset, data, self.model_name, self.device)

            # Extract features
            features = self.netset.extraction_function(preprocessed_data, layers_to_extract)

            # Select Feature Cleaner
            self.feature_cleaner = self.netset.get_feature_cleaner(self.data_type)

            # Clean features
            features = self.feature_cleaner(self.netset, features)

            # Write the features directly to individual files named after the input image
            for layer, data in features.items():
                file_path = os.path.join(self.save_path, f"{layer}_{data_file}.npz")
                np.savez_compressed(file_path, **{data_file: data.detach().numpy()})



    def consolidate_per_layer(self):
        # Identify unique layers from filenames
        all_files = os.listdir(self.save_path)
        unique_layers = set(file.split('_')[0] for file in all_files if '.npz' in file)

        for layer in tqdm(unique_layers):
            combined_data = {}
            # Gather all files for this layer
            layer_files = [file for file in all_files if file.startswith(layer)]
            for file in layer_files:
                file_path = os.path.join(self.save_path, file)
                with np.load(file_path, allow_pickle=True) as loaded_data:
                    combined_data.update(dict(loaded_data))
                os.remove(file_path)  # Optionally remove the individual file after reading
            # Save the combined data for this layer
            np.savez_compressed(os.path.join(self.save_path, f"{layer}.npz"), **combined_data)

    def consolidate_per_image(self):
        # Identify unique data files (images) from filenames
        all_files = os.listdir(self.save_path)
        unique_data_files = set(file.split('_')[-1].split('.npz')[0] for file in all_files if '.npz' in file)

        for data_file in tqdm(unique_data_files):
            combined_data = {}
            # Gather all files for this data_file (image)
            data_file_feature_files = [file for file in all_files if data_file in file]
            for file in data_file_feature_files:
                file_path = os.path.join(self.save_path, file)
                with np.load(file_path, allow_pickle=True) as loaded_data:
                    # Here, we use the prefix of the file (which indicates the layer) as the key in our combined_data dictionary.
                    combined_data[file.split('_')[0]] = dict(loaded_data)[data_file]
                os.remove(file_path)  # Optionally remove the individual file after reading
            # Save the combined data for this data_file (image)
            np.savez_compressed(os.path.join(self.save_path, f"{data_file}.npz"), **combined_data)


    def layers_to_extract(self):
        """Returns all possible layers for extraction."""

        return get_graph_node_names(self.netset.loaded_model)[0]


    def _initialize_netset(self, netset_name):
        # Use the dynamic loading and registration mechanism
        return NetSetBase._registry.get(netset_name, None)

    def _get_dataloader(self, data_path):
        # Logic to detect and return the correct DataType derived class
        file_extension = os.path.splitext(data_path)[1].lower()
    
        if file_extension in ['.jpg', '.jpeg', '.png']:
            return ImageData
        elif file_extension in ['.mp4', '.avi']:
            return VideoData
        elif file_extension in ['.wav', '.mp3']:
            return AudioData
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")











class DataTypeBase:
    TYPE = None

    @staticmethod
    def load_data(data_path):
        raise NotImplementedError

    @staticmethod
    def transform_data(data):
        """Transform raw data if needed (e.g., video to frames)."""
        return data  # Default behavior is to not transform
    

    

class ImageData(DataTypeBase):
    TYPE = 'image'
    @staticmethod
    def load_data(data_path):
        # Logic to load image data using PIL
        return Image.open(data_path).convert('RGB')


class VideoData(DataTypeBase):
    TYPE = 'video'
    @staticmethod
    def load_data(data_path):
        # Logic to load video data using cv2
        # This will return a list of frames. Each frame is a numpy array.
        cap = cv2.VideoCapture(data_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def transform_data(data):
        # If you want to convert frames to grayscale or resize them, etc.
        # You can add the logic here. 
        # For now, just returning the data as-is.
        return data


class AudioData(DataTypeBase):
    TYPE = 'audio'
    @staticmethod
    def load_data(data_path):
        # Logic to load audio data using librosa
        # This returns a numpy array representing the audio and its sample rate
        y, sr = librosa.load(data_path, sr=None)
        return y, sr