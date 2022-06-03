import torch
import numpy as np
import torchextractor as tx
from tqdm import tqdm
import glob
import cv2
import os.path as op
import os
from helper.helper import *


"""Write down all relevant paths"""
PATH_COLLECTION = get_paths()
CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]
BASE_DIR = PATH_COLLECTION["BASE_DIR"]
GUI_DIR = PATH_COLLECTION["GUI_DIR"]
PARENT_DIR = PATH_COLLECTION["PARENT_DIR"]
INPUTS_DIR = PATH_COLLECTION["INPUTS_DIR"]
FEATS_DIR = PATH_COLLECTION["FEATS_DIR"]
RDMS_DIR = PATH_COLLECTION["RDMS_DIR"]
STIMULI_DIR = PATH_COLLECTION["STIMULI_DIR"]
BRAIN_DIR = PATH_COLLECTION["BRAIN_DIR"]


class FeatureExtraction:
    """ This class is for generating features.  In the init function we select the relevant parameters as they are all different for each netset.
    The relevant ones are:
    
    self.module = Where is our network-data located?
    self.model = The actual model
    self.nodes = The layers we want to extract
    self.extractor = If we want to use torchextractor or anything else
    self.feature_cleaner = Some extractions return the arrays in a weird format, which is why some networks require a cleanup"""
    
    def __init__(self, model_name = "AlexNet", dataset = "78images", netset = "standard"):
        """[summary]

        Args:
            model_name (str): Which model we want to extract features from. Defaults to "AlexNet".
            dataset (str): Which dataset we choose. Defaults to "78images".
            netset (str): Which netset our model is from. Defaults to "standard".
        """
        
        # save inputs
        self.model_name = model_name
        self.dataset = dataset
        self.netset = netset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #######################################
        #######################################
        #######################################
        # Different nesets and their settings #
        #######################################
        #######################################
        #######################################
        
        if netset == "standard":
            
            import architectures.pytorch_models as pymodels
            
            # select module
            self.module = pymodels
        
            # retrieve model data
            self.model = self.module.MODELS[model_name](pretrained=True)
            self.nodes = self.module.MODEL_NODES[model_name]
            
            # select way to exract features
            self.extractor = self.extract_features_tx
            
            # select way to clean features
            self.feature_cleaner = self.no_clean
        

        elif netset == 'pytorch':
            
            import architectures.torchhub_models as torchm
            
            self.module = torchm

            # retrieve model data
            self.model = self.module.MODELS[model_name]('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
            self.model.eval()
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx
            
            # select way to clean features
            self.feature_cleaner = self.torch_clean
            
        elif netset == 'taskonomy':
            
            import architectures.taskonomy_models as taskonomy
            
            self.module = taskonomy

            # retrieve model data
            self.model = self.module.MODELS[model_name](eval_only=True)
            # Load Weights
            checkpoint = torch.utils.model_zoo.load_url(self.module.MODEL_WEIGHTS[model_name])
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean
            
        elif netset == 'unet':
            
            import architectures.unet_models as unet

            self.module = unet

            # retrieve model data
            self.model = self.module.MODELS[model_name]('mateuszbuda/brain-segmentation-pytorch', self.model_name, in_channels=3, out_channels=1, init_features=32, pretrained=True)
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean
            
        elif netset == 'clip':
            
            import architectures.clip_models as clip
            
            self.module = clip
            
            correct_model_name = self.model_name.replace("_-_", "/")
        
            # retrieve model data
            self.model = self.module.MODELS[model_name](correct_model_name, device=self.device)[0]
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx_clip

            # select way to clean features
            self.feature_cleaner = self.no_clean
            
        elif netset == 'cornet':
            
            import architectures.cornet_models as cornet

            self.module = cornet

            # retrieve model data
            self.model = self.module.MODELS[model_name]()
            self.model = torch.nn.DataParallel(self.model)  # turn into DataParallel
            
            # Load Weights
            ckpt_data = torch.utils.model_zoo.load_url(
                self.module.MODEL_WEIGHTS[model_name], map_location=self.device)
            self.model.load_state_dict(ckpt_data['state_dict'])

            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.CORnet_RT_clean

        elif netset == 'yolo':
            
            import architectures.yolo_models as yolo
            
            # TODO: ONLY WORKS ON CUDA YET - NEEDS CLEANUP
            
            self.module = yolo

            # retrieve model data
            self.model = self.module.MODELS[model_name](
                'ultralytics/yolov5', 'yolov5l', pretrained=True, device=self.device)
            
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

        elif netset == 'detectron2':
            
            import architectures.detectron2_models as detectron2
            
            self.module = detectron2

            # retrieve model data
            config = self.module.configurator(self.model_name)  # d2 works with configs
            self.model = self.module.MODELS[model_name](config)
            self.model.eval()  # needs to be put into eval mode
            
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.detectron_clean
        
            
        elif netset == 'vissl':

            import architectures.vissl_models as vissl

            self.module = vissl

            # retrieve model data
            config = self.module.configurator(self.model_name)  # d2 works with configs
            self.model = self.module.MODELS[model_name](config.MODEL, config.OPTIMIZER)

            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.no_clean

            
            
        elif netset == "timm":
            
            import architectures.timm_models as timm
            
            self.module = timm

            # retrieve model data
            try:
                self.model = self.module.MODELS[model_name](
                    model_name, pretrained=True, features_only=True)
            except:
                self.model = self.module.MODELS[model_name](
                    model_name, pretrained=True)
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to extract features
            if self.nodes == []:
                self.extractor = self.extract_features_timm
            else:
                self.extractor = self.extract_features_tx
                
            # select way to clean features
            self.feature_cleaner = self.no_clean
            
            
        elif netset == 'pyvideo':
            
            import architectures.slowfast_models as pyvideomodels

            self.module = pyvideomodels

            # retrieve model data
            self.model = self.module.MODELS[model_name]('facebookresearch/pytorchvideo', self.model_name, pretrained=True)
            self.model.eval()
            self.nodes = self.module.MODEL_NODES[model_name]

            # select way to exract features
            self.extractor = self.extract_features_tx

            # select way to clean features
            self.feature_cleaner = self.slowfast_clean
            
                
        # Paths
        self.feats_path = op.join(FEATS_DIR, self.dataset, self.netset + "_" + self.model_name)
        
        # Send to cuda
        if self.device == torch.device('cuda'):
            self.model.cuda()

        if not op.exists(self.feats_path):
            os.makedirs(self.feats_path)
    
    #####################       
    #####################  
    #####################
    # Helpful functions #
    #####################
    #####################
    #####################
            
    def sum_tensors(self, all_tensors, x):
        """Adding tensors together - this is relevant for averaging video frame results

        Args:
            all_tensors (dict:tensor): Dictionary of tensors that has been summed up over time
            x (dict:tensor): Dictionary of current tensors

        Returns:
            (dict:tensor): Current tensor added to overall tensors
        """

        if all_tensors == []:
            return x
        else:
            return {key: value + x[key] for key, value in all_tensors.items()}
        
    
    ###############################################
    ###############################################
    ###############################################
    # Measures to clean up the extracted features #
    ###############################################
    ###############################################
    ###############################################
    
    
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

               
    
    ##########################################
    ##########################################
    ##########################################
    # The different extractors we have/need! #
    ##########################################
    ##########################################
    ##########################################
    
    

    def extract_features_tx(self, image):
        """Function to extract features with torchextractor

        Args:
            image (PIL): image in PIL format

        Returns:
            (dict:tensors): Features in form of tensors
        """
          
        extrator = tx.Extractor(self.model, self.nodes)  # load model to extractor
            
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
        extrator = tx.Extractor(self.model, self.nodes)
        
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
    
    
    
    ##########################
    ##########################
    ##########################
    # The actual extraction! #
    ##########################
    ##########################
    ##########################


    
    def extract_from_images(self, image_list):
        """Function to loop over all our images, extract features and save them as .npz

        Args:
            image_list (list:str): List of paths to images
        """
        
        for image in tqdm(image_list):
            
            filename = op.split(image)[-1].split(".")[0]  # get filename
            
            image = self.module.preprocess(image, self.model_name)  # preprocess image
     
            features = self.extractor(image)  # extract features
            
            save_path = op.join(self.feats_path, filename + ".npz")  # create safe-path
            
            # turn tensor into numpy array
            features = {key: value.detach().numpy() for key, value in features.items()}
            
            np.savez(save_path, **features)  # safe data
            
        

    def extract_from_frames(self, image_list):
        """Function to loop over all our videos, turn into frames, extract features and save them as .npz

        Args:
            image_list (list:str): List of paths to images
        """
        
        for image in tqdm(image_list):
            
            filename = op.split(image)[-1].split(".")[0]  # get filename
            
            vidcap = cv2.VideoCapture(image)
            
            success, frame = vidcap.read()
            
            frame_tensor = []
            
            while success:
                
                frame = self.module.preprocess_frame(frame, self.model_name)
                
                features = self.extractor(frame)
                
                frame_tensor = self.sum_tensors(frame_tensor, features)
                
                success, frame = vidcap.read()
                
            save_path = op.join(self.feats_path, filename + ".npz")  # create safe-path
            
            # turn tensor into numpy array
            features = {key: value.detach().numpy() for key, value in features.items()}

            np.savez(save_path, **features)  # safe data
                
            
                

            
    def start_extraction(self):
        """Find all images/videos in path and start extration
        """
        
        # list all input files
        image_list = glob.glob(op.join(STIMULI_DIR, self.dataset, "*"))
        
        image_list.sort()

        filetype = op.split(image_list[0])[-1].split(".")[1]  # get filetype
        
        if filetype == "jpg":
            self.extract_from_images(image_list)
            
        elif filetype == "mp4":
            if self.netset == 'pyvideo':
                self.extract_from_images(image_list)  # pyvideo can deal with full on videos
            else:
                self.extract_from_frames(image_list)


    def get_all_nodes(self):
        """Helping function to extract all possible nodes from a model

        Returns:
            list: all nodes we can extract features from
        """
        nodes = tx.list_module_names(self.model)
        return nodes
        
    



