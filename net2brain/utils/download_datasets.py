import os
import gdown
import zipfile
import pandas as pd
import urllib
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import re
import warnings


class DatasetError(Exception):
    pass


def load_dataset(dataset_name, path=None):
    warnings.warn(
        "The 'load_dataset' function is deprecated. Please use the new class-based approach. "
        "For example, use 'DatasetBonnerPNAS2017.load_dataset()' to load datasets. "
        "You can list all available datasets using 'list_available_datasets()'.",
        DeprecationWarning,
        stacklevel=2
    )
    # Here you can still call the old function if you maintain it for backward compatibility or remove it after a transition period.

def list_available_datasets():
    def get_all_subclasses(cls):
        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            subclasses.extend(get_all_subclasses(subclass))
        return subclasses
    
    return [cls.__name__ for cls in get_all_subclasses(BaseDataset)]


class BaseDataset:
    DATASET_URLS = {}  # To be defined by subclass

    def __init__(self, path=None):
        self.path = path or os.getcwd()
        self.dataset_folder = os.path.join(self.path, self.dataset_name)

    def download_and_extract_zip(self):
        if self.dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}.")
        
        url = self.DATASET_URLS[self.dataset_name]

        if not os.path.exists(self.dataset_folder):
            print(f"Downloading dataset '{self.dataset_name}' to {self.path}...")
            zip_file_path = os.path.join(self.path, "temporary_gdrive.zip")
            gdown.download(url, zip_file_path, quiet=False)
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                zip_file.extractall(self.path)
            print(f"Files extracted to {self.path}")
            
            os.remove(zip_file_path)
    
    
    def _load(self):
        raise NotImplementedError("This method should be implemented by subclass.")
    
    @classmethod
    def load_dataset(cls, path=None):
        instance = cls(path)
        return instance._load()



class DatasetBonnerPNAS2017(BaseDataset):
    dataset_name = "bonner_pnas2017"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fi6GiRVCbBZxLQASNV1qYz/bonner_pnas2017.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        live_data = os.path.join(self.dataset_folder, "brain_data_live_study")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path, "PPA_Study": live_data}
    
    
    
class Dataset78images(BaseDataset):
    dataset_name = "78images"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiLSqoiEArM6Mi8qZDpyxr/78images.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path}
    
    
class Workhsop_Harry_Potter_Cognition(BaseDataset):
    dataset_name = "Workshop_Harry_Potter_Cognition"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fi15Hgyt64LPdiKMnNT3P4/Workshop_Harry_Potter_Cognition.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        brain_path = os.path.join(self.dataset_folder, "entire_brain_data")
        roi_path = os.path.join(self.dataset_folder, "roi_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path, "brain_path":brain_path}
    
    
class Dataset92images(BaseDataset):
    dataset_name = "92images"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiVMG4j85ZS2j8tG9cGfAj/92images.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path}
    
    

class DatasetBoldMoments(BaseDataset):
    dataset_name = "BoldMoments"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiMgXvrfGeFjNgm5wyTTSJ/BoldMoments.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        print("Beware that this is a reduced dataset for testing purposes. For the full dataset please go to https://openneuro.org/datasets/ds005165/versions/1.0.4")
        stimuli_train = os.path.join(self.dataset_folder, "stimuli_train")
        stimuli_test = os.path.join(self.dataset_folder, "stimuli_test")
        roi_path = os.path.join(self.dataset_folder, "fmri")
        
        return {"stimuli_train": stimuli_train, 
                "stimuli_test": stimuli_test, 
                "roi_path": roi_path}
    

class WorkshopCuttingGardens(BaseDataset):
    dataset_name = "cutting_gardens23"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiHrLWdMVTVJGWBmrGcYP1/cutting_gardens23.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        return {"data_dir": self.dataset_folder}






class Tutorial_LE_Results(BaseDataset):
    dataset_name = "Tutorial_LE_Results"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiRqQSBqmEDVtjJZ7TPGmM/Tutorial_LE_Results.zip"
    }

    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        # Dictionary to store folder names and their paths
        folder_paths = {}
        
        # Iterate over items in the dataset folder
        for item in os.listdir(self.dataset_folder):
            item_path = os.path.join(self.dataset_folder, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                folder_paths[item] = item_path
                
        return folder_paths



class DatasetAlgonauts_NSD_Shared(BaseDataset):
    dataset_name = "Algonauts23_shared_Net2Brain"
    DATASET_URLS = {
        dataset_name: "https://www.dropbox.com/scl/fi/y3jbpl8pixeewjjj7svbz/Algonauts23_shared_subset_Net2Brain.zip?rlkey=f2tmrxj2wapp87wb1eux5u6cc&st=gfb0pi82&dl=1"
    }



    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder
        
    def _load(self):
        self.download_and_extract_zip()
        root = self.dataset_folder
        paths = {
            "root":        root,
            "images":      os.path.join(root, "images"),
            "metadata":    os.path.join(root, "metadata_shared.csv"),
            "coco_csv":    os.path.join(root, "coco.csv"),
            "coco_images": os.path.join(root, "coco_images"),
            "coco_masks":  os.path.join(root, "coco_masks"),
        }
        for s in range(1, 9):
            subj = f"subj0{s}"
            paths[subj]           = os.path.join(root, subj)
            paths[f"{subj}_fmri"] = os.path.join(root, subj, "fmri")
            paths[f"{subj}_rois"] = os.path.join(root, subj, "rois")
            paths[f"{subj}_rdms"] = os.path.join(root, subj, "rdms")
        return paths
    

    
    
    def NSDtoCOCO(self, nsd_id, coco_path=""):
        
        if coco_path == "":
            coco_path = self.source_path + "/coco.csv"
            
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the dataset
        df = pd.read_csv(coco_path)
        
        # Ensure nsd_id is treated as a string, then strip leading zeros
        nsd_id_str = str(nsd_id).lstrip('0')

        # Convert the nsd_id in the DataFrame to string and strip leading zeros for matching
        df['nsd_id'] = df['nsd_id'].astype(str).apply(lambda x: x.lstrip('0'))

        # Find the COCO ID corresponding to the given NSD ID
        if nsd_id_str not in df['nsd_id'].values:
            raise DatasetError(f"NSD ID {nsd_id} not found in the dataset.")
        
        return int(df[df['nsd_id'] == nsd_id_str]['coco_id'].values[0])  # Assuming coco_id is numeric


    def COCOtoNSD(self, coco_id, coco_path=""):
        
        if coco_path == "":
            coco_path = self.source_path + "/coco.csv"
            
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the dataset
        df = pd.read_csv(coco_path, dtype={'coco_id': str})
        
        if coco_id not in df['coco_id'].values:
            raise DatasetError(f"COCO ID {coco_id} not found in the dataset.")
        
        return df[df['coco_id'] == coco_id]['nsd_id'].values[0]
    


    def Download_COCO_Images(self, nsd_image_folder, target_folder, NSD_path=""):
        
        if NSD_path == "":
            NSD_path = self.source_path
            
        coco_csv_path = os.path.join(NSD_path, "coco.csv")
        
        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_csv_path, dtype={'nsd_id': str, 'coco_id': str, 'coco_split': str})

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        train_annotations_path = os.path.join(NSD_path, "instances_train2017.json")
        val_annotations_path = os.path.join(NSD_path, "instances_val2017.json")

        coco_train = COCO(train_annotations_path) if os.path.exists(train_annotations_path) else None
        coco_val = COCO(val_annotations_path) if os.path.exists(val_annotations_path) else None

        for filename in tqdm(os.listdir(nsd_image_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                nsd_id_match = re.search(r'nsd-(\d+)', filename)
                nsd_id = nsd_id_match.group(1).lstrip('0') if nsd_id_match else filename.split('.')[0].lstrip('0')
                
                if nsd_id in df['nsd_id'].values:
                    row = df[df['nsd_id'] == nsd_id].iloc[0]
                    coco_id = int(row['coco_id'])
                    coco_split = row['coco_split']

                    coco = coco_train if coco_split == 'train2017' else coco_val

                    # Fetch image info and URL
                    img_info = coco.loadImgs(coco_id)[0]
                    img_url = img_info['coco_url']

                    # Download the image
                    img_filename = os.path.join(target_folder, f"{coco_id}.png")  # Replace file extension with 'png'
                    urllib.request.urlretrieve(img_url, img_filename)
                else:
                    print(f"No COCO ID found for NSD ID {nsd_id}")

        print("All specified COCO images have been downloaded to the target folder.")

    
    
        
    def Download_COCO_Segmentation_Masks(self, nsd_image_folder, target_folder, NSD_path=""):
        
        if NSD_path == "":
            NSD_path = self.source_path
            
        coco_csv_path = os.path.join(NSD_path, "coco.csv")
        
        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_csv_path, dtype={'nsd_id': str, 'coco_id': str, 'coco_split': str})

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        train_annotations_path = os.path.join(NSD_path, "instances_train2017.json")
        val_annotations_path = os.path.join(NSD_path, "instances_val2017.json")
        coco_train = COCO(train_annotations_path) if os.path.exists(train_annotations_path) else None
        coco_val = COCO(val_annotations_path) if os.path.exists(val_annotations_path) else None

        for filename in tqdm(os.listdir(nsd_image_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                nsd_id_match = re.search(r'nsd-(\d+)', filename)
                nsd_id = nsd_id_match.group(1).lstrip('0') if nsd_id_match else filename.split('.')[0].lstrip('0')

                if nsd_id in df['nsd_id'].values:
                    row = df[df['nsd_id'] == nsd_id].iloc[0]
                    coco_id = int(row['coco_id'])
                    coco_split = row['coco_split']

                    coco = coco_train if coco_split == 'train2017' else coco_val

                    annIds = coco.getAnnIds(imgIds=coco_id, iscrowd=None)
                    anns = coco.loadAnns(annIds)

                    # Initialize a mask for the whole image
                    img_info = coco.loadImgs(coco_id)[0]
                    composite_mask = np.zeros((img_info['height'], img_info['width']))

                    for ann in anns:
                        # Generate segmentation mask for the current annotation
                        mask = coco.annToMask(ann)
                        
                        # Update the composite mask
                        composite_mask = np.maximum(composite_mask, mask * ann['category_id'])

                    # Convert the composite mask to an image and save it
                    mask_img = Image.fromarray(np.uint8(composite_mask))
                    mask_filename = os.path.join(target_folder, f"{coco_id}.png")  # Replace file extension with 'png'
                    mask_img.save(mask_filename)
                else:
                    print(f"No COCO ID found for NSD ID {nsd_id}")

        print("All specified COCO segmentation masks have been downloaded to the target folder.")
        
        
        

    def Download_COCO_Captions(self, nsd_image_folder, target_folder, NSD_path=""):
        
        if NSD_path == "":
            NSD_path = self.source_path
            
        coco_csv_path = os.path.join(NSD_path, "coco.csv")
        train_captions_path = os.path.join(NSD_path, "captions_train2017.json")
        val_captions_path = os.path.join(NSD_path, "captions_val2017.json")
        
        # Check if NSD_path and required files exist
        if not os.path.exists(coco_csv_path) or not os.path.exists(train_captions_path) or not os.path.exists(val_captions_path):
            raise DatasetError(f"Required files within '{NSD_path}' were not found. Please download the dataset using 'DatasetNSD.load_dataset()' and ensure the correct path is specified with 'NSD_path = path'.")
        
        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_csv_path, dtype={'nsd_id': str, 'coco_id': str, 'coco_split': str})

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        coco_captions_train = COCO(train_captions_path)
        coco_captions_val = COCO(val_captions_path)

        for filename in tqdm(os.listdir(nsd_image_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                nsd_id_match = re.search(r'nsd-(\d+)', filename)
                if nsd_id_match:
                    nsd_id = nsd_id_match.group(1).lstrip('0')
                else:
                    nsd_id = filename.split('.')[0].lstrip('0')
                
                if nsd_id in df['nsd_id'].values:
                    row = df[df['nsd_id'] == nsd_id].iloc[0]
                    coco_id = row['coco_id']
                    coco_split = row['coco_split']

                    coco_captions = coco_captions_train if coco_split == 'train2017' else coco_captions_val

                    annIds = coco_captions.getAnnIds(imgIds=int(coco_id))
                    anns = coco_captions.loadAnns(annIds)

                    if anns:
                        caption = anns[0]['caption']
                        new_name = filename.split(".")[0]
                        with open(os.path.join(target_folder, f"{new_name}.txt"), 'w') as f:
                            f.write(f"{caption.strip()}\n")  # Use strip() to remove leading/trailing whitespace and new lines

                else:
                    print(f"No NSD ID found for {filename}")

        print("All specified COCO captions have been downloaded to the target folder.")

        
        
        
        
    def Crop_COCO_to_NSD(self, source_folder, target_folder, coco_path=""):
        
        if coco_path == "":
            coco_path = self.source_path + "/coco.csv"
            
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Create the output folder if it doesn't exist
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Read the CSV file
        df = pd.read_csv(coco_path)

        # Iterate through the images in the source folder
        for filename in tqdm(os.listdir(source_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                coco_id = int(filename.split('.')[0])  # Extracting COCO ID from filename

                # Find the corresponding row in the DataFrame
                if coco_id in df['coco_id'].values:
                    row = df[df['coco_id'] == coco_id].iloc[0]
                    crop_str = row['cropBox']
                    crop_values = eval(crop_str)  # Convert string to tuple

                    # Load the image
                    image_path = os.path.join(source_folder, filename)
                    image = Image.open(image_path)
                    img_array = np.array(image)

                    # Apply the cropping logic
                    top, bottom, left, right = crop_values  # Assuming the order is (top, left, bottom, right)
                    topCrop = int(round(img_array.shape[0] * top))
                    bottomCrop = int(round(img_array.shape[0] * bottom))
                    leftCrop = int(round(img_array.shape[1] * left))
                    rightCrop = int(round(img_array.shape[1] * right))
                    cropped_image_array = img_array[topCrop:img_array.shape[0]-bottomCrop, leftCrop:img_array.shape[1]-rightCrop]

                    # If cropping masks, ensure only original class labels are in the cropped image
                    if filename.endswith('.png'):
                        unique_classes = np.unique(img_array)
                        cropped_image = Image.fromarray(cropped_image_array)
                        # Resize the image using NEAREST to avoid introducing new labels
                        resized_image = cropped_image.resize((425, 425), Image.NEAREST)
                        resized_array = np.array(resized_image)
                        # Correct any potential new labels introduced by resizing
                        for label in np.unique(resized_array):
                            if label not in unique_classes:
                                # Find the closest existing label
                                closest_label = min(unique_classes, key=lambda x: abs(x - label))
                                resized_array[resized_array == label] = closest_label
                        final_image = Image.fromarray(resized_array)
                    else:
                        # Convert the cropped array back to an image and resize
                        cropped_image = Image.fromarray(cropped_image_array)
                        final_image = cropped_image.resize((425, 425))

                    # Save the final image
                    save_path = os.path.join(target_folder, filename)
                    final_image.save(save_path)

        print("Processing complete. Cropped images are saved in:", target_folder)


        


    def Visualize(self, image_folder, mask_folder, image_id):
        # Define COCO classes
        COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic', 'fire', 'street', 'stop', 
                        'parking', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 
                        'umbrella', 'shoe', 'eye', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'kite', 'baseball', 'baseball', 
                        'skateboard', 'surfboard', 'tennis', 'bottle', 'plate', 'wine', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted', 'bed', 'mirror', 
                        'dining', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell', 'microwave', 'oven', 
                        'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy', 'hair', 'toothbrush', 'hair']
        
         # Check for both JPG and PNG formats and load the one that exists
        jpg_image_path = os.path.join(image_folder, f"{image_id}.jpg")
        png_image_path = os.path.join(image_folder, f"{image_id}.png")

        if os.path.exists(jpg_image_path):
            original_image = Image.open(jpg_image_path)
        elif os.path.exists(png_image_path):
            original_image = Image.open(png_image_path)
        else:
            raise FileNotFoundError(f"No image found for ID {image_id} in {image_folder} in JPG or PNG format.")

        # Load the mask
        mask_path = os.path.join(mask_folder, f"{image_id}.png")
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        # Create a colored mask: background black, labels in color
        unique_labels = np.unique(mask_array)
        colored_mask_array = np.zeros((*mask_array.shape, 3), dtype=np.uint8)  # Initialize with zeros (black)
        for label in unique_labels:
            if label > 0:  # Skip background
                color = plt.cm.jet(label / np.max(unique_labels))[:3]  # Get color from colormap
                color = (np.array(color) * 255).astype(np.uint8)  # Convert color to RGB format
                colored_mask_array[mask_array == label] = color  # Apply color to label positions

        colored_mask = Image.fromarray(colored_mask_array)

        # Create an overlay of the image with the colored mask
        colored_mask = colored_mask.resize(original_image.size, Image.NEAREST)
        overlay = Image.blend(original_image.convert("RGBA"), colored_mask.convert("RGBA"), alpha=0.5)

        # Find unique classes in the mask (excluding background)
        unique_classes = unique_labels[unique_labels != 0]

        # Plotting
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original Image
        axs[0].imshow(original_image)
        axs[0].set_title(f"Original Image (ID: {image_id})")
        axs[0].axis('off')

        # Colored Original Mask with black background
        axs[1].imshow(colored_mask)
        axs[1].set_title("Colored Original Mask")
        axs[1].axis('off')

        # Overlay with original image visible only where mask is present
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay with Visible Labels")
        axs[2].axis('off')

        # Legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=COCO_CLASSES[i-1] if i-1 < len(COCO_CLASSES) else 'Undefined',
                                    markerfacecolor=plt.cm.jet(i/np.max(unique_classes)), markersize=10) for i in unique_classes]
        axs[3].legend(handles=legend_elements, loc='upper left')
        axs[3].set_title("Legend")
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()
        
        

    def RenameToCOCO(self, folder, coco_path=""):
        
        if coco_path == "":
            coco_path = self.source_path + "/coco.csv"
            
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise FileNotFoundError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_path, dtype={'nsd_id': str, 'coco_id': str})

        for filename in tqdm(os.listdir(folder)):
            basename, extension = os.path.splitext(filename)

            nsd_id_match = re.search(r'nsd-(\d+)', basename)
            if nsd_id_match:
                nsd_id = nsd_id_match.group(1)
            else:
                nsd_id = basename

            if nsd_id in df['nsd_id'].values:
                coco_id = df[df['nsd_id'] == nsd_id]['coco_id'].values[0]

                # Rename the file
                new_filename = f"{coco_id}{extension}"
                os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
                
                
    def RenameToNSD(self, folder, coco_path=""):
        
        if coco_path == "":
            coco_path = self.source_path + "/coco.csv"
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the COCO to NSD mapping
        df = pd.read_csv(coco_path, dtype={'nsd_id': str, 'coco_id': str})
        
        for filename in tqdm(os.listdir(folder)):
            basename, extension = os.path.splitext(filename)
            
            if basename in df['coco_id'].values:
                nsd_id = df[df['coco_id'] == basename]['nsd_id'].values[0]
                
                # Format NSD ID with leading zeros for 10 thousands
                nsd_id_formatted = nsd_id.zfill(5)

                # Rename the file
                new_filename = f"{nsd_id_formatted}{extension}"
                os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
                
                
                
    def RenameAlgonautsToNSD(self,folder_path):
        # Regular expression to match the NSD ID pattern in the filenames
        nsd_id_pattern = re.compile(r'nsd-(\d+)')

        for filename in os.listdir(folder_path):
            # Search for the NSD ID pattern in the filename
            match = nsd_id_pattern.search(filename)
            if match:
                nsd_id = match.group(1)  # Extract the NSD ID
                file_extension = os.path.splitext(filename)[1]  # Get the file extension
                new_filename = f"{nsd_id}{file_extension}"  # Construct the new filename with NSD ID and extension

                # Construct full paths for the old and new filenames
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)

        print("Finished renaming files.")




        
    

class DatasetAlgonauts_NSD(DatasetAlgonauts_NSD_Shared):
    dataset_name = "Algonauts23_Net2Brain"
    DATASET_URLS = {
        dataset_name: "https://www.dropbox.com/scl/fi/lzxtzp1sb7ivh0esy2d2y/Algonauts23_Net2Brain.zip?rlkey=dpqstfzr0gr03cfwcc0ea1t5v&st=d2d8ba4l&dl=1"
    }
        
    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = self.dataset_folder

    def _load(self):
        self.download_and_extract_zip()
        root = self.dataset_folder
        paths = {
            "root":     root,
            "metadata": os.path.join(root, "metadata.csv"),
            "coco_csv": os.path.join(root, "coco.csv"),
        }
        for s in range(1, 9):
            subj = f"subj0{s}"
            subj_dir = os.path.join(root, subj)
            paths[subj]                  = subj_dir
            paths[f"{subj}_images"]      = os.path.join(subj_dir, "training_split", "training_images")
            paths[f"{subj}_test_images"] = os.path.join(subj_dir, "test_split", "test_images")
            paths[f"{subj}_fmri"]        = os.path.join(subj_dir, "training_split", "training_fmri")
            paths[f"{subj}_rois"]        = os.path.join(subj_dir, "training_split", "training_rois")
        return paths


 
class DatasetThings_fMRI(BaseDataset):
    dataset_name = "Things_test"
    DATASET_URLS = {
        dataset_name: "https://hessenbox-a10.rz.uni-frankfurt.de/dl/fiBk8D3GYTMefWCgvAt7mV/Things_test.zip"
    }


    def __init__(self, path=None):
        super().__init__(path)
        self.source_path = "Things_test"
        
    def _load(self):
        self.download_and_extract_zip()
        # Dictionary to store folder names and their paths
        folder_paths = {}
        
        # Iterate over items in the dataset folder
        for item in os.listdir(self.dataset_folder):
            item_path = os.path.join(self.dataset_folder, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                folder_paths[item] = item_path
                
        return folder_paths