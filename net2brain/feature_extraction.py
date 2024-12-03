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
from .architectures.audio_models import Audio
from datetime import datetime
import torchextractor as tx
import warnings
import json
import torch
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

from pathlib import Path
from .utils.dim_reduction import estimate_from_files


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
                # Initiate as the Netset structure of choice in case user does not select preprocessing, extractor, etc.
                self.netset = NetSetBase.initialize_netset(
                    model_name=None, netset_name=netset_fallback, device=self.device
                )
                self.model = model
                self.model.eval()
                self.netset.loaded_model = self.model

                if None in (preprocessor, extraction_function, feature_cleaner):
                    warnings.warn("If you add your own model you can also select our own: \nPreprocessing Function (preprocessor) \nExtraction Function (extraction_function) \nFeature Cleaner (feature_cleaner) ")
    




    def extract(self, data_path, save_path=None, layers_to_extract=None, consolidate_per_layer=True,
                dim_reduction=None, n_samples_estim=100, n_components=10000, max_dim_allowed=None):
        """
        Args:
            data_path: str
                Path to stimuli data or list of data_paths
            save_path: str
                Path to where to save the extracted features
            layers_to_extract: list of str
                List of layers to extract from model.
                User "get_all_layers" function to see all avaiable layers
            consolidate_per_layer: Bool
                Whether to consolidate of one file per image to one file per layer
            dim_reduction: str or None
                Whether to apply dimensionality reduction to the features at the feature extraction stage.
                If the original full features are needed for further processing, set this to None and apply the
                dimensionality reduction at the RDM creation stage when the features are loaded.
                Choose from `srp` (Sparse Random Projection) and `pca` (Principal Component Analysis).
                The next three parameters only apply when `dim_reduction` is not None.
            n_samples_estim: int
                The number of samples used for estimating the dimensionality reduction.
            n_components: int or None
                The number of components to reduce the features to. If None, the number of components is estimated.
                For PCA, `n_components` must be smaller than `n_samples_estim`.
            max_dim_allowed: int or None
                Optional: The threshold over which the dimensionality reduction is applied. If None, it is always
                applied.

        Returns:

        """

        # Create save_path:
        now = datetime.now()
        now_formatted = f'{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}'
        self.save_path = save_path or os.path.join(os.getcwd(),"results", now_formatted)
        self._ensure_dir_exists(self.save_path)

        # Specify new attributes for dimensionality reduction:
        self.dim_reduction = dim_reduction
        self.n_samples_estim = n_samples_estim
        self.n_components = n_components
        self.max_dim_allowed = max_dim_allowed

        # Iterate over all files in the given data_path
        self.data_path = data_path

        # Flatten the list of supported extensions from DataTypeLoader
        DataWrapper = DataTypeLoader(self.netset)
        all_supported_extensions = [ext for extensions in DataWrapper.supported_extensions.values() for ext in extensions]

        # Filter data_files to include only files with supported extensions
        if isinstance(data_path, (str, Path)):
            data_files = [i for i in Path(data_path).iterdir() if i.suffix.lower() in all_supported_extensions]
        else:
            data_files = [Path(f) for f in data_path if Path(f).suffix.lower() in all_supported_extensions]
        data_files.sort()

        # Detect data type for the current file
        if isinstance(data_path, (str, Path)):
            data_loader, self.data_type, self.data_combiner = DataWrapper._get_dataloader(data_path)
        else:
            data_loader, self.data_type, self.data_combiner = DataWrapper._get_dataloader(data_path)
        
        if self.data_type == "multimodal":
            data_files = self._pair_modalities(data_files)

        # Select preprocessor
        if self.preprocessor == None:
            self.preprocessor = self.netset.get_preprocessing_function(self.data_type)

        if self.data_type not in self.netset.supported_data_types:
            raise ValueError(f"Datatype {self.data_type} not supported by current model")
        
        progress_bar = tqdm(data_files, desc='Processing files')
        
        for data_file in progress_bar:

            # Get datapath
            file_name = str(data_file).split(os.sep)[-1].split(".")[0]

            # Load data
            data_from_file = data_loader(data_file)

            # Create empty list for data accumulation
            data_from_file_list = []
            
            # Total number of items in the inner loop
            total_inner_items = len(data_from_file)

            for idx, data in enumerate(data_from_file):
                
                # Update the progress bar description to show progress of the inner loop
                progress_bar.set_postfix(subfiles=f'{idx + 1}/{total_inner_items}')

                # Preprocess data
                preprocessed_data = self.preprocessor(data, self.model_name, self.device)

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
                data_from_file_list.append({key: value.detach().cpu() for key, value in features.items()})
                del preprocessed_data, features

            # Combine Data from list into single dictionary depending on input type
            final_features = self.data_combiner(data_from_file_list)

            # Convert the final_features dictionary to one that contains detached numpy arrays
            if self.data_type == "text":
                for idx, subfeature in enumerate(final_features):
                    subfeature = {key: value.numpy() for key, value in subfeature.items()}

                    # Write the features for one image to a single file
                    file_path = os.path.join(self.save_path, f"{file_name}_sentence{idx:04d}.npz")
                    np.savez(file_path, **subfeature)

            else:
                final_features = {key: value.numpy() for key, value in final_features.items()}

                # Write the features for one image to a single file
                file_path = os.path.join(self.save_path, f"{file_name}.npz")
                np.savez(file_path, **final_features)

                # Clear variables to save RAM
                del data_from_file_list, final_features

        if dim_reduction:
            print("Reducing dimensions...")
            self.reduce_dimensionality()

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

            output_file_path = os.path.join(self.save_path, f"consolidated_{layer}.npz")
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


    def reduce_dimensionality(self):
        # List all files, ignoring ones ending with "_consolidated.npz"
        all_files = [f for f in os.listdir(self.save_path) if (f.endswith(".npz") and not
                                                               f.endswith("_consolidated.npz"))]
        if not all_files:
            print("No feature files to reduce dimension for.")
            return

        def open_npz(file):
            return np.load(os.path.join(self.save_path, file), allow_pickle=True)

        # Assuming that each file has the same set of layers
        sample = open_npz(all_files[0])
        layers = list(sample.keys())

        for layer in layers:
            sample_feats_at_layer = sample[layer]
            feat_dim = sample_feats_at_layer.shape[1:]
            # Check if the dimensionality reduction is necessary
            if not self.max_dim_allowed or len(sample_feats_at_layer.flatten()) > self.max_dim_allowed:
                # Estimate the dimensionality reduction from a subset of the data
                fitted_transform, _ = estimate_from_files(all_files, layer, feat_dim, open_npz,
                                             self.dim_reduction, self.n_samples_estim, self.n_components)
                for file in tqdm(all_files):
                    feats = open_npz(file)
                    reduced_feats_at_layer = {}
                    # Apply the dimensionality reduction to the features at the layer
                    for key, value in feats.items():
                        if key == layer:
                            reduced_feats_at_layer[key] = fitted_transform.transform(value.reshape(1, -1))
                        else:
                            reduced_feats_at_layer[key] = value
                    # Make sure no corrupted files are saved
                    feats.close()
                    np.savez(os.path.join(self.save_path, 'tmp_'+file), **reduced_feats_at_layer)
                    os.replace(os.path.join(self.save_path, 'tmp_'+file), os.path.join(self.save_path, file))


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
    
    
    def _get_modalities_in_files(self, files):
        """Check which modalities exist in the provided list of files."""
        modalities = defaultdict(list)
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            for modality, extensions in self.supported_extensions.items():
                if file_extension in extensions:
                    base_name = os.path.splitext(os.path.basename(file))[0]
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
        if isinstance(folder_path, (str, Path)):
            modalities = self._get_modalities_in_folder(folder_path)
        else:
            if not os.path.isfile(folder_path[0]):
                raise ValueError("You entered the path to a folder in the data_path=[] of the extractor or the file you entered does not exist. Either enter the path to the folder outside the list or enter single files as a list.")
            else:
                modalities = self._get_modalities_in_files(folder_path)
                
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
            raise ValueError("You have mixed single and multimodal files found in the folder. Ensure all files are either single or multimodal.")




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


