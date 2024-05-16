from .architectures.netsetbase import NetSetBase
import os
from collections import defaultdict
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
from .architectures.huggingface_llm import Huggingface
from datetime import datetime
import torchextractor as tx
import warnings
import json
import torch
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

from pathlib import Path


try:
    from .architectures.clip_models import Clip
except ModuleNotFoundError:
    warnings.warn("Clip not installed")



# FeatureExtractor class
class FeatureExtractor:
    def __init__(self,
                 model,
                 netset=None,
                 netset_fallback="Standard",
                 device="cpu",
                 pretrained=True,
                 preprocessor=None,
                 extraction_function=None,
                 feature_cleaner=None):
        # Parameters
        self.model_name = model
        self.device = device
        self.pretrained = pretrained

        # Get values for editable functions
        self.preprocessor = preprocessor
        self.extraction_function = extraction_function
        self.feature_cleaner = feature_cleaner




        if netset is not None:
            self.netset_name = netset
            self.netset = NetSetBase.initialize_netset(self.model_name, netset, device)


            # Initiate netset-based functions
            self.model = self.netset.get_model(self.pretrained)
            self.layers_to_extract = self.netset.layers

        else:
            if isinstance(model, str):
                raise ValueError("If no netset is given, the model_name parameter needs to be a ready model")
            else:
                # Initiate as Standard Netset structure in case user does not select preprocessing, extractor, etc.
                self.netset = NetSetBase.initialize_netset(
                    model_name=None, netset_name=netset_fallback, device=self.device
                )
                self.model = model
                self.model.eval()
                self.netset.loaded_model = self.model

                if None in (preprocessor, extraction_function, feature_cleaner):
                    warnings.warn("If you add your own model you can also select our own: \nPreprocessing Function (preprocessor) \nExtraction Function (extraction_function) \nFeature Cleaner (feature_cleaner) ")





    def extract(self, data_path, save_path=None, layers_to_extract=None, consolidate_per_layer=True, dim_reduction=None, n_components=50):

        # Create save_path:
        now = datetime.now()
        now_formatted = f'{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}'
        self.save_path = save_path or os.path.join(os.getcwd(),"results", now_formatted)
        self._ensure_dir_exists(self.save_path)

        # Specify new attributes:
        self.dim_reduction = dim_reduction
        self.n_components = n_components

        # Iterate over all files in the given data_path
        self.data_path = data_path

        # Flatten the list of supported extensions from DataTypeLoader
        DataWrapper = DataTypeLoader(self.netset)
        all_supported_extensions = [ext for extensions in DataWrapper.supported_extensions.values() for ext in extensions]

        # Filter data_files to include only files with supported extensions
        data_files = [i for i in Path(data_path).iterdir() if i.suffix.lower() in all_supported_extensions]
        data_files.sort()

        # Detect data type for the current file
        data_loader, self.data_type, self.data_combiner = DataWrapper._get_dataloader(data_path)

        if self.data_type == "multimodal":
            data_files = self._pair_modalities(data_files)

        if self.data_type not in self.netset.supported_data_types:
            raise ValueError(f"Datatype {self.data_type} not supported by current model")


        for data_file in tqdm(data_files):

            # Get datapath
            file_name = str(data_file).split(os.sep)[-1].split(".")[0]

            # Load data
            data_from_file = data_loader(data_file)

            # Create empty list for data accumulation
            data_from_file_list = []

            for data in data_from_file:

                # Preprocess data
                # Select preprocessor
                if self.preprocessor == None:
                    preprocessed_data = self.netset.get_preprocessing_function(self.data_type)(
                        data, self.model_name, self.device
                    )
                else:
                    preprocessed_data = self.preprocessor(data, self.device)

                # Extract features
                if self.extraction_function == None:
                    features = self.netset.extraction_function(preprocessed_data, layers_to_extract)
                else:
                    features = self.extraction_function(preprocessed_data, layers_to_extract, model=self.model)

                # Select Feature Cleaner
                if self.feature_cleaner == None:
                    feature_cleaner = self.netset.get_feature_cleaner(self.data_type)
                    features = feature_cleaner(self.netset, features)
                else:
                    features = self.feature_cleaner(features)

                # Append to list of data
                data_from_file_list.append(features)

            # Combine Data from list into single dictionary depending on input type
            final_features = self.data_combiner(data_from_file_list)

            # Reduce dimension of feautres
            # print("final_feats", len(final_features))
            final_features = self.reduce_dimensionality(final_features)


            # Convert the final_features dictionary to one that contains detached numpy arrays
            final_features = {key: value.detach().cpu().numpy() for key, value in final_features.items()}

            # Write the features for one image to a single file
            file_path = os.path.join(self.save_path, f"{file_name}.npz")
            np.savez(file_path, **final_features)

            # Clear variables to save RAM
            del data_from_file_list, final_features

        if consolidate_per_layer:
            print("Consolidating data per layer...")
            self.consolidate_per_layer()


    def _ensure_dir_exists(self, directory_path):
        """
        Ensure the specified directory exists.
        If not, it will be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)



    def consolidate_per_layer(self):
        # List all files, ignoring ones ending with "_consolidated.npz"
        all_files = [f for f in os.listdir(self.save_path) if not f.endswith("_consolidated.npz")]
        if not all_files:
            print("No files to consolidate.")
            return

        # Assuming that each file has the same set of layers
        sample_file_path = os.path.join(self.save_path, all_files[0])
        with np.load(sample_file_path, allow_pickle=True) as data:
            layers = list(data.keys())

        # Initialize a dictionary for combined data for each layer
        combined_data = {layer: {} for layer in layers}

        # Iterate over each file and update the combined_data structure
        for file_name in tqdm(all_files):
            file_path = os.path.join(self.save_path, file_name)
            with np.load(file_path, allow_pickle=True) as data:
                if not data.keys():
                    #print(f"Error: The file {file_name} is empty.")
                    continue

                for layer in layers:
                    if layer not in data or data[layer].size == 0:
                        #print(f"Error: The layer {layer} in file {file_name} is empty.")
                        continue

                    image_key = file_name.replace('.npz', '')
                    combined_data[layer][image_key] = data[layer]

            # Remove the file after its data has been added to combined_data
            os.remove(file_path)

        # Save the consolidated data for each layer
        for layer, data in combined_data.items():
            if not data:
                print(f"Error: No data found to consolidate for layer {layer}.")
                continue

            output_file_path = os.path.join(self.save_path, f"{layer}.npz")
            np.savez_compressed(output_file_path, **data)



    def consolidate_per_txt_file(self):
        """
        Consolidate features from multiple sentences in a single file into separate files.

        Parameters:
        - save_path (str): Path to the folder containing .npz files.

        Returns:
        - None
        """

        # Create new dir
        if not os.path.exists(os.path.join(self.save_path, "sentences")):
            os.mkdir(os.path.join(self.save_path, "sentences"))

        # List all files in the given directory
        files = [f for f in os.listdir(self.save_path) if f.endswith('.npz')]

        for file_name in tqdm(files):
            # Load the features from the original file
            data = np.load(os.path.join(self.save_path, file_name))

            # Extract original file name without extension
            original_file_name = os.path.splitext(file_name)[0]

            # Get the number of sentences in the file (assuming all layers have the same number of sentences)
            num_sentences = len(list(data.values())[0][0])

            for i in range(num_sentences):
                # Create a new file name with index
                new_file_name = f"{original_file_name}_sentence_{i}.npz"

                # Extract features for the current sentence across all layers
                sentence_features = {key: values[0][i] for key, values in data.items()}

                # Save features for the current sentence to a new file
                np.savez(os.path.join(self.save_path, "sentences", new_file_name), **sentence_features)



    def get_all_layers(self):
        """Returns all possible layers for extraction."""

        return tx.list_module_names(self.netset.loaded_model)


    def _pair_modalities(self, files):
        # Dictionary to hold base names and their associated files
        base_names = defaultdict(list)

        # Iterate over files to group by base name
        for file in files:
            base_name = os.path.splitext(file)[0]
            base_names[base_name].append(file)

        # Create tuples from the grouped files
        paired_files = [tuple(files) for files in base_names.values() if len(files) > 1]

        # Check if all groups have more than one modality, otherwise raise an error
        single_modality_bases = [base for base, files in base_names.items() if len(files) == 1]
        if single_modality_bases:
            raise ValueError(f"Missing modalities for: {', '.join(single_modality_bases)}")

        return paired_files


    def _initialize_netset(self, netset_name):
        # Use the dynamic loading and registration mechanism
        return NetSetBase._registry.get(netset_name, None)


    def reduce_dimensionality(self, features):
        if self.dim_reduction == None:
            return features
        elif self.dim_reduction == "srp":
            return self.reduce_dimensionality_sparse_random(features)
        else:
            warnings.warn(f"{self.dim_reduction} does not exist as form of dimensionality reduction. Choose between 'srp'")




    def reduce_dimensionality_sparse_random(self, features):
        """
        Perform dimensionality reduction using Sparse Random Projection.

        Parameters:
        - features (dict): Dictionary of layers with corresponding torch tensors.
        - n_components (int): Number of components to keep (default is 50).

        Returns:
        - dict: Reduced features.
        """
        reduced_features = {}
        for layer, original_tensor in features.items():
            flattened_tensor = original_tensor.detach().view(original_tensor.size(0), -1).cpu().numpy()
            sparse_random_proj = SparseRandomProjection(n_components=self.n_components)
            reduced_tensor = sparse_random_proj.fit_transform(flattened_tensor)
            reduced_features[layer] = torch.tensor(reduced_tensor).view(original_tensor.size(0), -1)
        return reduced_features



class DataTypeLoader:
    def __init__(self, netset):
        self.netset = netset
        self.supported_extensions = {
            'image': ['.jpg', '.jpeg', '.png'],
            'video': ['.mp4', '.avi'],
            'audio': ['.wav', '.mp3'],
            'text': ['.txt']
        }

    def _get_modalities_in_folder(self, folder_path):
        """Check which modalities exist in the folder."""
        files = os.listdir(folder_path)
        modalities = defaultdict(list)
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            for modality, extensions in self.supported_extensions.items():
                if file_extension in extensions:
                    base_name = os.path.splitext(file)[0]
                    modalities[base_name].append(modality)
                    break
        return modalities

    def _check_modalities(self, modalities):
        """Check for multimodal files and their completeness."""
        multimodal_files = {}
        single_modal_files = {}
        for base_name, mods in modalities.items():
            if len(mods) > 1:
                multimodal_files[base_name] = mods
            else:
                single_modal_files[base_name] = mods[0]

        # Ensure that multimodal files have counterparts in each modality
        for base_name, mods in multimodal_files.items():
            if len(set(mods)) != len(mods):  # Duplicate modality found
                raise ValueError(f"Multiple files for the same modality found for {base_name}. Expected one file per modality.")

        return multimodal_files, single_modal_files

    def _get_dataloader(self, folder_path):
        modalities = self._get_modalities_in_folder(folder_path)
        multimodal_files, single_modal_files = self._check_modalities(modalities)

        if multimodal_files and not single_modal_files:
            # If all files are multimodal
            data_loader = getattr(self.netset, 'load_multimodal_data')
            data_combiner = getattr(self.netset, 'combine_multimodal_data')
            return data_loader, 'multimodal', data_combiner
        elif single_modal_files and not multimodal_files:
            # If all files are single modal
            modality = next(iter(single_modal_files.values()))
            data_loader = getattr(self.netset, f'load_{modality}_data')
            data_combiner = getattr(self.netset, f'combine_{modality}_data')
            return data_loader, modality, data_combiner
        else:
            raise ValueError("Mixed single and multimodal files found in the folder. Ensure all files are either single or multimodal.")




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


