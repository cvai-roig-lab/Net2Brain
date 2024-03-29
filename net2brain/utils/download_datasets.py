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

class DatasetError(Exception):
    pass

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
        dataset_name: "https://drive.google.com/uc?id=1gW4otwb7yPqyAbP3YUiFO08Wnw_HnS8t"
    }

    def __init__(self, path=None):
        super().__init__(path)

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path}
    
    
    
class Dataset78images(BaseDataset):
    dataset_name = "78images"
    DATASET_URLS = {
        dataset_name: "https://drive.google.com/uc?id=1b1SWkkISwzqFl0URE8iNDBGui1vbww6u"
    }

    def __init__(self, path=None):
        super().__init__(path)

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path}
    
    
class Dataset92images(BaseDataset):
    dataset_name = "92images"
    DATASET_URLS = {
        dataset_name: "https://drive.google.com/uc?id=1dpbo5NYD6z7yQUfdpQcg3Y59AKoZD_b-"
    }

    def __init__(self, path=None):
        super().__init__(path)

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path}
    

class WorkshopCuttingGardens(BaseDataset):
    dataset_name = "cutting_gardens23"
    DATASET_URLS = {
        dataset_name: "https://drive.google.com/uc?export=download&id=195MA1fqplzxLZsfs282yAXTPKn0qy5d7"
    }

    def __init__(self, path=None):
        super().__init__(path)

    def _load(self):
        self.download_and_extract_zip()
        stimuli_path = os.path.join(self.dataset_folder, "stimuli_data")
        roi_path = os.path.join(self.dataset_folder, "brain_data")
        return {"stimuli_path": stimuli_path, "roi_path": roi_path}




class DatasetNSD(BaseDataset):
    dataset_name = "NSD Dataset"
    DATASET_URLS = {
        dataset_name: "https://drive.google.com/uc?export=download&id=1OCKE7efSojxwDlNTE93yybEZZl_wJ7mX"
    }


    def __init__(self, path=None):
        super().__init__(path)

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
    
    
    def NSDtoCOCO(self, nsd_id, coco_path="NSD Dataset/coco.csv"):
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the dataset
        df = pd.read_csv(coco_path)
        
        # Ensure nsd_id is treated as a string, then strip leading zeros
        nsd_id_str = str(nsd_id).lstrip('0')

        # Convert the nsd_id in the DataFrame to string and strip leading zeros for matching
        df['nsdId'] = df['nsdId'].astype(str).apply(lambda x: x.lstrip('0'))

        # Find the COCO ID corresponding to the given NSD ID
        if nsd_id_str not in df['nsdId'].values:
            raise DatasetError(f"NSD ID {nsd_id} not found in the dataset.")
        
        return int(df[df['nsdId'] == nsd_id_str]['cocoId'].values[0])  # Assuming cocoId is numeric


    def COCOtoNSD(self, coco_id, coco_path="NSD Dataset/coco.csv"):
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the dataset
        df = pd.read_csv(coco_path, dtype={'cocoId': str})  # Ensure cocoId is read as string
        
        # Find the NSD ID corresponding to the given COCO ID
        if coco_id not in df['cocoId'].values:
            raise DatasetError(f"COCO ID {coco_id} not found in the dataset.")
        
        return df[df['cocoId'] == coco_id]['nsdId'].values[0]  # nsdId is returned as it is in the csv, potentially with leading zeros
    


    def Download_COCO_Images(self, nsd_image_folder, target_folder, NSD_path="NSD Dataset"):
        coco_csv_path = os.path.join(NSD_path, "coco.csv")
        annotations_path = os.path.join(NSD_path, "instances_train2017.json")
        
        # Check if NSD_path and required files exist
        if not os.path.exists(NSD_path) or not os.path.exists(coco_csv_path) or not os.path.exists(annotations_path):
            raise DatasetError(f"The NSD dataset folder or required files within it were not found at '{NSD_path}'. Please download the dataset using 'DatasetNSD.load_dataset()' and ensure the correct path is specified with 'NSD_path = path'.")
        
        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_csv_path)
        
        # Ensure target folder exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Initialize COCO API
        coco = COCO(annotations_path)

        # Iterate through NSD image files and collect NSD IDs
        nsd_ids = [int(filename.split('.')[0]) for filename in os.listdir(nsd_image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Convert NSD IDs to COCO IDs
        coco_ids = df[df['nsdId'].isin(nsd_ids)]['cocoId'].tolist()

        # Download COCO images corresponding to COCO IDs
        for coco_id in tqdm(coco_ids):
            img_info = coco.loadImgs(coco_id)[0]  # Assuming each ID returns a single image
            img_url = img_info['coco_url']
            
            # Strip leading zeros from coco_id for the filename
            filename_no_leading_zeros = str(coco_id).lstrip('0') + os.path.splitext(os.path.basename(img_url))[1]
            img_filename = os.path.join(target_folder, filename_no_leading_zeros)

            # Download the image
            urllib.request.urlretrieve(img_url, img_filename)

        print("All specified COCO images have been downloaded to the target folder.")
    
    
    
    def Download_COCO_Segmentation_Masks(self, nsd_image_folder, target_folder, NSD_path="NSD Dataset"):
        coco_csv_path = os.path.join(NSD_path, "coco.csv")
        annotations_path = os.path.join(NSD_path, "instances_train2017.json")
        
        # Check if NSD_path and required files exist
        if not os.path.exists(NSD_path) or not os.path.exists(coco_csv_path) or not os.path.exists(annotations_path):
            raise DatasetError(f"The NSD dataset folder or required files within it were not found at '{NSD_path}'. Please download the dataset using 'DatasetNSD.load_dataset()' and ensure the correct path is specified with 'NSD_path = path'.")
        
        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_csv_path)
        
        # Ensure target folder exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Initialize COCO API
        coco = COCO(annotations_path)

        # Iterate through NSD image files and collect NSD IDs
        nsd_ids = [int(filename.split('.')[0]) for filename in os.listdir(nsd_image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Convert NSD IDs to COCO IDs
        coco_ids = df[df['nsdId'].isin(nsd_ids)]['cocoId'].tolist()

        # Download COCO segmentation masks corresponding to COCO IDs
        for coco_id in tqdm(coco_ids):
            annIds = coco.getAnnIds(imgIds=coco_id, iscrowd=None)
            anns = coco.loadAnns(annIds)

            # Initialize a mask for the whole image
            img_info = coco.loadImgs(coco_id)[0]
            composite_mask = np.zeros((img_info['height'], img_info['width']))

            for ann in anns:
                # Generate segmentation mask for the current annotation
                mask = coco.annToMask(ann)
                
                # Update the composite mask
                # Note: This logic simply overlays masks. If you need to differentiate overlapping objects, you might need a more complex approach.
                composite_mask = np.maximum(composite_mask, mask * ann['category_id'])

            # Convert the composite mask to an image and save it
            mask_img = Image.fromarray(np.uint8(composite_mask))
            mask_filename = os.path.join(target_folder, f"{coco_id}.png")
            mask_img.save(mask_filename)

        print("All specified COCO segmentation masks have been downloaded to the target folder.")
        
        
        

    def Download_COCO_Captions(self, nsd_image_folder, target_folder, NSD_path="NSD Dataset"):
        coco_csv_path = os.path.join(NSD_path, "coco.csv")
        captions_path = os.path.join(NSD_path, "captions_train2017.json")
        
        # Check if NSD_path and required files exist
        if not os.path.exists(NSD_path) or not os.path.exists(coco_csv_path) or not os.path.exists(captions_path):
            raise DatasetError(f"The NSD dataset folder or required files within it were not found at '{NSD_path}'. Please download the dataset using 'DatasetNSD.load_dataset()' and ensure the correct path is specified with 'NSD_path = path'.")
        
        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_csv_path)
        
        # Ensure target folder exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Initialize COCO API for captions
        coco_captions = COCO(captions_path)

        # Iterate through NSD image files and collect NSD IDs
        nsd_ids = [int(filename.split('.')[0]) for filename in os.listdir(nsd_image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Convert NSD IDs to COCO IDs
        coco_ids = df[df['nsdId'].isin(nsd_ids)]['cocoId'].tolist()

        # Fetch and save COCO captions corresponding to COCO IDs
        for coco_id in tqdm(coco_ids):
            annIds = coco_captions.getAnnIds(imgIds=coco_id)
            anns = coco_captions.loadAnns(annIds)
            captions = [ann['caption'] for ann in anns]

            # Create a .txt file per image with all captions
            with open(os.path.join(target_folder, f"{coco_id}.txt"), 'w') as f:
                for caption in captions:
                    f.write(f"{caption}\n")

        print("All specified COCO captions have been downloaded to the target folder.")
        
        
        
        
    def Crop_COCO_to_NSD(self, source_folder, target_folder, coco_path="NSD Dataset/coco.csv"):
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
                if coco_id in df['cocoId'].values:
                    row = df[df['cocoId'] == coco_id].iloc[0]
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
        
        # Load the original image
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        original_image = Image.open(image_path)

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
        
        

    def RenameToCOCO(self, folder, coco_path="NSD Dataset/coco.csv"):
        
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the NSD to COCO mapping
        df = pd.read_csv(coco_path, dtype={'nsdId': str, 'cocoId': str})  # Ensure IDs are read as strings
        
        # Iterate through the files in the folder
        for filename in tqdm(os.listdir(folder)):
            basename, extension = os.path.splitext(filename)
            
            # Check if the basename is in the nsdId column
            if basename in df['nsdId'].values:
                # Find the corresponding COCO ID
                coco_id = df[df['nsdId'] == basename]['cocoId'].values[0]
                
                # Rename the file
                new_filename = f"{coco_id}{extension}"
                os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
                
                
    def RenameToNSD(self, folder, coco_path="NSD Dataset/coco.csv"):
        # Check if the file exists
        if not os.path.exists(coco_path):
            raise DatasetError(f"The file at '{coco_path}' could not be found. Please ensure the file exists or specify the correct location using the 'coco_path' parameter. This file can be downloaded using 'DatasetNSD.load_dataset()'.")

        # Load the COCO to NSD mapping
        df = pd.read_csv(coco_path, dtype={'nsdId': str, 'cocoId': str})  # Ensure IDs are read as strings
        
        # Iterate through the files in the folder
        for filename in tqdm(os.listdir(folder)):
            basename, extension = os.path.splitext(filename)
            
            # Check if the basename is in the cocoId column
            if basename in df['cocoId'].values:
                # Find the corresponding NSD ID
                nsd_id = df[df['cocoId'] == basename]['nsdId'].values[0]
                
                # Format NSD ID with leading zeros for 10 thousands
                nsd_id_formatted = nsd_id.zfill(5)

                # Rename the file
                new_filename = f"{nsd_id_formatted}{extension}"
                os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))


        
    



