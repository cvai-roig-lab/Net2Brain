from .architectures.netsetbase import NetSetBase
import os
import numpy as np
from tqdm import tqdm
from .architectures.pytorch_models import Standard
from .architectures.timm_models import Timm
from .architectures.taskonomy_models import Taskonomy
from .architectures.toolbox_models import Toolbox
from .architectures.torchhub_models import Pytorch
from .architectures.cornet_models import Cornet
from .architectures.unet_models import Unet
from .architectures.yolo_models import Yolo
from .architectures.pyvideo_models import Pyvideo
from datetime import datetime
import torchextractor as tx
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

try:
    from .architectures.clip_models import Clip
except ModuleNotFoundError:
    warnings.warn("Clip not installed")



# FeatureExtractor class
class FeatureExtractor:
    def __init__(self, model_name, netset, data_path, device, pretrained=True, save_path=None):
        # Parameters
        self.model_name = model_name
        self.netset_name = netset
        self.device = device
        self.netset = NetSetBase.initialize_netset(self.model_name, netset, device)
        self.data_path = data_path
        self.pretrained = pretrained


        # Create save_path:
        now = datetime.now()
        now_formatted = f'{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}'
        self.save_path = save_path or os.path.join(os.getcwd(),"results", now_formatted)
        print(self.save_path)

        # Initiate netset-based functions
        self.model = self.netset.get_model(self.pretrained)


    def extract(self, layers_to_extract=None):
        # Iterate over all files in the given data_path

        for data_file in tqdm(os.listdir(self.data_path)):

            full_path = os.path.join(self.data_path, data_file)
            
            # Detect data type
            data_loader, self.data_type, self.data_combiner = self._get_dataloader(full_path)

            if self.data_type not in self.netset.supported_data_types:
                raise ValueError(f"{self.netset_name} does not support data type: {self.data_type}")
            
            # Load data
            data_from_file = data_loader(full_path)

            data_from_file_list = []

            for data in data_from_file:

                # Select preprocessor
                self.preprocessor = self.netset.get_preprocessing_function(self.data_type)

                # Preprocess data
                preprocessed_data = self.preprocessor(data, self.model_name, self.device)

                # Extract features
                features = self.netset.extraction_function(preprocessed_data, layers_to_extract)

                # Select Feature Cleaner
                self.feature_cleaner = self.netset.get_feature_cleaner(self.data_type)

                # Clean features
                features = self.feature_cleaner(self.netset, features)

                # Append to list of data
                data_from_file_list.append(features)

            # Combine Data from list into single dictionary depending on input type
            final_features = self.data_combiner(data_from_file_list)

            # Write the features directly to individual files named after the input image
            for layer, data in final_features.items():
                file_path = os.path.join(self.save_path, f"{layer}_{data_file}.npz")
                self._ensure_dir_exists(file_path)
                np.savez_compressed(file_path, **{data_file: data.detach().cpu().numpy()})


    def _ensure_dir_exists(self, file_path):
        """
        Ensure the directory of the given file path exists.
        If not, it will be created.
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


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

        return tx.list_module_names(self.netset.loaded_model)


    def _initialize_netset(self, netset_name):
        # Use the dynamic loading and registration mechanism
        return NetSetBase._registry.get(netset_name, None)

    def _get_dataloader(self, data_path):
        # Logic to detect and return the correct DataType derived class
        file_extension = os.path.splitext(data_path)[1].lower()
    
        if file_extension in ['.jpg', '.jpeg', '.png']:
            data_loader = self.netset.load_image_data
            data_type = "image"
            data_combiner = self.netset.combine_image_data
            return data_loader, data_type, data_combiner
        
        elif file_extension in ['.mp4', '.avi']:
            data_loader = self.netset.load_video_data
            data_type = "video"
            data_combiner = self.netset.combine_video_data
            return data_loader, data_type, data_combiner
        
        elif file_extension in ['.wav', '.mp3']:
            data_loader = self.netset.load_audio_data
            data_type = "audio"
            data_combiner = self.netset.combine_audio_data
            return data_loader, data_type, data_combiner
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        



# Create Taxonomy

def get_netset_model_dict():
    # Define a Function to Load JSON Configs
    def load_json_config(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data

    # Initialize Your Dictionary
    netset_models_dict = {}

    # Iterate Over the NetSets in the Registry
    for netset_name, netset_class in NetSetBase._registry.items():
        try:
            # Provide placeholder values for model_name and device
            netset_instance = netset_class(model_name='placeholder', device='cpu')
            
            # Access the config path directly from the instance
            config_path = netset_instance.config_path
            
            # Load the config file
            models_data = load_json_config(config_path)
            
            # Extract the model names and add them to the dictionary
            model_names = list(models_data.keys())
            netset_models_dict[netset_name] = model_names
        
        except AttributeError:
            # Handle the case where config_path is not defined in the instance
            warnings.warn(f"{netset_name} does not have a config_path attribute")
        except Exception as e:
            print(f"An error occurred while processing {netset_name}: {str(e)}")

    # Return the Dictionary
    return netset_models_dict

# Have this variable for the taxonomy function
all_networks = get_netset_model_dict()


