import torch
import numpy as np
import torchextractor as tx
from tqdm import tqdm
import glob
import os.path as op
import os
from datetime import datetime
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as T

import net2brain.architectures.pytorch_models as pymodule
import net2brain.architectures.torchhub_models as torchmodule
import net2brain.architectures.taskonomy_models as taskonomy
import net2brain.architectures.unet_models as unet
import net2brain.architectures.yolo_models as yolo
import net2brain.architectures.timm_models as timm
import net2brain.architectures.slowfast_models as pyvideo

AVAILABLE_NETWORKS = {'standard': list(pymodule.MODELS.keys()),
                      'timm': list(timm.MODELS.keys()),
                      'pytorch': list(torchmodule.MODELS.keys()),
                      'unet': list(unet.MODELS.keys()),
                      'taskonomy': list(taskonomy.MODELS.keys()),
                      'pyvideo': list(pyvideo.MODELS.keys())}

try:
    import clip
    import architectures.clip_models as clip_models
    AVAILABLE_NETWORKS.update({'clip': list(clip_models.MODELS.keys())})
except ModuleNotFoundError:
    print("Clip models are not installed.")
    clip_exist = False

try:
    import cornet
    import architectures.cornet_models as cornet_models
    AVAILABLE_NETWORKS.update({'cornet': list(cornet_models.MODELS.keys())})
except ModuleNotFoundError:
    print("CORnet models are not installed.")
    cornet_exist = False

try:
    import vissl
    import architectures.vissl_models as vissl_models
    AVAILABLE_NETWORKS.update({'vissl': list(vissl_models.MODELS.keys())})
except ModuleNotFoundError:
    print("Vissl models are not installed")
    vissl_exist = False

try:
    import detectron2
    import architectures.detectron2_models as detectron2_models
    AVAILABLE_NETWORKS.update({'detectron2': list(detectron2_models.MODELS.keys())})
except ModuleNotFoundError:
    print("Detectron2 is not installed.")
    detectron_exist = False

"""Write down all relevant paths"""
CURRENT_DIR = op.abspath(os.curdir)
BASE_DIR = op.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = op.dirname(BASE_DIR)  # path to parent folder
FEATS_DIR = op.join(PARENT_DIR, 'feats')
GUI_DIR = op.join(BASE_DIR, 'helper', 'gui')
INPUTS_DIR = op.join(PARENT_DIR, 'input_data')
STIMULI_DIR = op.join(INPUTS_DIR, 'stimuli_data')
RDMS_DIR = op.join(PARENT_DIR, 'rdms')
BRAIN_DIR = op.join(INPUTS_DIR, 'brain_data')


def ensure_directory(path):
    """Method to ensure directory exists

    Args:
        path (str): path to folder to create
    """
    if not os.path.exists(path):
        os.mkdir(path)


def create_save_folder():
    """Creates folder to save the features in. They are structured after daytime

    Returns:
        save_path(str): Path to save folder
    """
    # Get current time
    now = datetime.now()
    now_formatted = now.strftime("%d.%m.%y %H:%M:%S")

    # Replace : through -
    log_time = now_formatted.replace(":", "-")

    # Combine to path
    save_path = f"feats/{log_time}"

    # Create directory
    ensure_directory(f"feats/{log_time}")

    return save_path



def return_all_models():
    """Returns all available models
    """
    return AVAILABLE_NETWORKS

def return_all_netsets():
    """Returns all available netsets
    """
    return list(AVAILABLE_NETWORKS.keys())

def return_models(netset):
    """Returns all models within a set netset
    Args:
        netset (str): The name of the netset
    """
    if netset in list(AVAILABLE_NETWORKS.keys()):
        return AVAILABLE_NETWORKS[netset]
    else:
        raise KeyError(f"This netset '{netset}' is not available. Available are", list(AVAILABLE_NETWORKS.keys()))

def find_like(name):
    """Finds networks which have the given name within the model name.
    Way to find models within the model zoo

    Args:
        name (str): Name of model
    """
    for key, values in AVAILABLE_NETWORKS.items():
        for model_names in values:
            if name.lower() in model_names.lower():
                print(f'{key}: {model_names}')


class FeatureExtractor:
    """ This class is for generating features.  In the init function we select
    the relevant parameters as they are all different for each netset.
    The relevant ones are:

    self.model = The actual model
    self.model_name = Model name as string
    self.device = GPU or CUDA
    self.save_path = Location to save features
    self.module = Where is our network-data located?
    self.layers_to_extract= The layers we want to extract
    self.extractor = If we want to use torchextractor or anything else
    self.feature_cleaner = Some extractions return the arrays in a weird format,
                           which is why some networks require a cleanup
    self.transforms = Some images may need to be transformed/preprocessed before entering the network
    self.preprocess = Function for preprocessing the images
    """

    def __init__(self, model, device, netset=None, transforms=None):
        """Initiating FeatureExtractor Class
        No parameters needed as they will be set depending if the model is imported or loaded from the zoo
        """

        if type(model) == str:
            self.load_model_netset(model, netset, device)
        else: 
            self.load_model(model, device, transforms)

        # self.model = None
        # self.model_name = None
        # self.device = None
        # self.module = None
        # self.layers_to_extract= None
        # self.extractor = None
        # self.feature_cleaner = None
        # self.transforms = None
        # self.preprocess = None
        # self.save_path = None

        pass

    def load_model(self, model, device, transforms=None):
        """Load model into the extractor not from the model zoo

        Args:
            model (model): The actual model
            device (torch): GPU or CUDA
            transforms (Torch transforms, optional): Possible transformation to the input images Defaults to None.
        """
        # Save inputs
        self.model = model
        self.model_name = "Inserted model"
        self.device = device
        self.feats_path = None
        self.model.to(self.device)

        self.extractor = self.extract_features_tx
        self.feature_cleaner = self.no_clean

        self.transforms = transforms
        self.preprocess = self.model_preprocess
        self.layers_to_extract = None


    def model_preprocess(self, image, model_name):
        """Default preprocessing function based on the ImageNet values

        Args:
            image (path): path to image
            model_name (str): model name, not needed in this case, only needed with models from model zoo

        Returns:
            _type_: _description_
        """
        if self.transforms is None:
            self.transforms = T.Compose([T.Resize((224, 224)),
                                         T.ToTensor(),
                                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = Image.open(image)  # Open image

        image = V(self.transforms(image).unsqueeze(0))  # Apply transformation

        image = image.to(self.device)  # To Device

        return image

    def load_model_netset(self, model_name, netset, device):
        """Function to load a model from our modelzoo

        Args:
            model_name (str): Name of model
            netset (str): Name of netset
            device (torch): CPU or CUDA

        Returns:
            model (torch): The actual model
            layers (list): List of proposed layers to extract
        """

        self.model_name = model_name

        if netset == "standard":

            # select module
            self.module = pymodule

            # retrieve model data
            self.model = self.module.MODELS[model_name](pretrained=True)
            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'pytorch':

            self.module = torchmodule

            # retrieve model data
            self.model = self.module.MODELS[model_name]('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
            self.model.eval()
            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.torch_clean

        elif netset == 'taskonomy':

            self.module = taskonomy

            # retrieve model data
            self.model = self.module.MODELS[model_name](eval_only=True)
            # Load Weights
            checkpoint = torch.utils.model_zoo.load_url(self.module.MODEL_WEIGHTS[model_name])
            self.model.load_state_dict(checkpoint['state_dict'])

            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'unet':

            self.module = unet

            # retrieve model data
            self.model = self.module.MODELS[model_name]('mateuszbuda/brain-segmentation-pytorch', self.model_name, in_channels=3, out_channels=1, init_features=32, pretrained=True)
            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'clip':

            self.module = clip_models

            correct_model_name = self.model_name.replace("_-_", "/")

            # retrieve model data
            self.model = self.module.MODELS[model_name](correct_model_name, device=self.device)[0]
            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx_clip

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'cornet':

            self.module = cornet_models

            # retrieve model data
            self.model = self.module.MODELS[model_name]()
            self.model = torch.nn.DataParallel(self.model)  # turn into DataParallel

            # Load Weights
            ckpt_data = torch.utils.model_zoo.load_url(
                self.module.MODEL_WEIGHTS[model_name], map_location=self.device)
            self.model.load_state_dict(ckpt_data['state_dict'])

            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.CORnet_RT_clean

        elif netset == 'yolo':

            # TODO: ONLY WORKS ON CUDA YET - NEEDS CLEANUP

            self.module = yolo

            # retrieve model data
            self.model = self.module.MODELS[model_name](
                'ultralytics/yolov5', 'yolov5l', pretrained=True, device=self.device)

            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'detectron2':

            self.module = detectron2_models

            # retrieve model data
            config = self.module.configurator(self.model_name)  # d2 works with configs
            self.model = self.module.MODELS[model_name](config)
            self.model.eval()  # needs to be put into eval mode

            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.detectron_clean

        elif netset == 'vissl':

            self.module = vissl_models

            # retrieve model data
            config = self.module.configurator(self.model_name)  # d2 works with configs
            self.model = self.module.MODELS[model_name](config.MODEL, config.OPTIMIZER)

            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == "timm":

            self.module = timm

            # retrieve model data
            try:
                self.model = self.module.MODELS[model_name](
                    model_name, pretrained=True, features_only=True)
            except:
                self.model = self.module.MODELS[model_name](
                    model_name, pretrained=True)
            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to extract features
            if self.layers_to_extract == []:
                self.extractor = self.extract_features_timm
            else:
                self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'pyvideo':

            self.module = pyvideo

            # retrieve model data
            self.model = self.module.MODELS[model_name]('facebookresearch/pytorchvideo', self.model_name, pretrained=True)
            self.model.eval()
            self.layers_to_extract = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.slowfast_clean

        self.device = device
        self.model.to(self.device)
        self.preprocess = self.module.preprocess

        return self.model, self.layers_to_extract

    def no_clean(self, features):
        """Cleanup function after feature extraction: This one requires no cleanup.
        Just put it on cpu in case it isn't yet!

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """

        return {key: value.data.cpu() for key, value in features.items()}

    def torch_clean(self, features):
        """Cleanup function after feature extraction: This one contains subdictionaries which need to be eliminated

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """

        new_features = {}
        for key, value in features.items():
            try:
                new_features[key] = value["out"].data.cpu()
            except:
                new_features[key] = value.data.cpu()

        return new_features

    def detectron_clean(self, features):
        """Cleanup function after feature extraction: This one contains subdictionaries which need to be eliminated

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """
        clean_dict = {}
        for key, subdict in features.items():
            keys = list(subdict.keys())
            for key in keys:
                clean_dict.update({key: subdict[key].cpu()})
        return clean_dict

    def CORnet_RT_clean(self, features):
        """Cleanup function after feature extraction: The RT-Model contains subdirectories

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """

        if self.model_name == "cornet_rt":
            clean_dict = {}
            for A_key, subtuple in features.items():
                keys = [A_key + "_A", A_key + "_B"]
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
                    break  # we actually only want one key
            return clean_dict
        else:
            return {key: value.cpu() for key, value in features.items()}

    def slowfast_clean(self, features):
        """Cleanup function after feature extraction: Some features have two values (list)

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """

        clean_dict = {}
        for A_key, subtuple in features.items():
            keys = [A_key + "_slow", A_key + "_fast"]

            try:  # if subdict is a list of two values
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
            except:
                clean_dict.update({A_key: subtuple.cpu()})

        return clean_dict

    def extract_features_tx(self, image):
        """Function to extract features with torchextractor

        Args:
            image (PIL): image in PIL format

        Returns:
            (dict:tensors): Features in form of tensors
        """

        extrator = tx.Extractor(self.model, self.layers_to_extract)  # load model to extractor

        _, features = extrator(image)  # extract layers with image

        features = self.feature_cleaner(features)

        return features

    def extract_features_tx_clip(self, image):
        """Function to extract features with torchextractor. CLIP needs text input which we chose to be random.

        Args:
            image (PIL): image in PIL format

        Returns:
            (dict:tensors): Features in form of tensors
        """

        # load model to extractor
        extrator = tx.Extractor(self.model, self.layers_to_extract)

        image_data = image[0]
        tokenized_data = image[1]

        _, features = extrator(image_data, tokenized_data)  # extract layers with image, and tokenized text

        features = self.feature_cleaner(features)

        return features

    def extract_features_timm(self, image):
        """Function to extract features with timm

        Args:
            image (PIL): image in PIL format

        Returns:
            (dict:tensors): Features in form of tensors
        """

        features = self.model(image)

        converted_features = {}

        # We need to convert the features into a dict, because timm returns a list of tensors1

        for counter, feature in enumerate(features):
            converted_features["feature " + str(counter + 1)] = feature.data.cpu()

        features = self.feature_cleaner(converted_features)

        return converted_features

    def extract_from_images(self, image_list):
        """Function to loop over all our images, extract features and save them as .npz

        Args:
            image_list (list:str): List of paths to images
        """

        for image in tqdm(image_list):

            filename = op.split(image)[-1].split(".")[0]  # get filename

            # preprocess image
            processsed_image = self.preprocess(image, self.model_name)

            # extract features
            features = self.extractor(processsed_image)  # extract features

            # create save_path for file
            save_path = op.join(self.save_path, filename + ".npz")  # create safe-path

            # turn tensor into numpy array
            features = {key: value.detach().numpy() for key, value in features.items()}

            np.savez(save_path, **features)  # safe data

    def extract_feats(self, dataset_path, save_path=None, layers_to_extract=None):
        """Function to start the feature extraction

        Args:
            layers (list): list of layers to extract
            dataset_path (path): path to dataset
            save_path (path): Path where to save features. Defaults to None.
        """

        # Create save path and ensure save path exists
        if save_path is None:
            self.save_path = create_save_folder()
        else:
            ensure_directory(save_path)
            self.save_path = save_path

        # Store layers in class
        if layers_to_extract is None:
            pass
        else:
            self.layers_to_extract = layers_to_extract

        # Find all input files
        image_list = glob.glob(op.join(dataset_path, "*"))
        image_list.sort()

        # get filetype
        filetype = op.split(image_list[0])[-1].split(".")[1]

        # If images are jpg, trigger the function
        if filetype == "jpg":
            self.extract_from_images(image_list)
        else:
            raise TypeError("Can only handle .jpg images for now")  # TODO: Add .png and .mp4 video data

    def get_all_layers(self):
        """Helping function to extract all possible layers from a model

        Returns:
            list: all layers we can extract features from
        """
        layers = tx.list_module_names(self.model)
        return layers
