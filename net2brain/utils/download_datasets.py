import os
import zipfile
import gdown

DATASET_URLS = {
    "78images": "https://drive.google.com/uc?id=1b1SWkkISwzqFl0URE8iNDBGui1vbww6u",
    "92images": "https://drive.google.com/uc?id=1dpbo5NYD6z7yQUfdpQcg3Y59AKoZD_b-",
    "bonner_pnas2017": "https://drive.google.com/uc?id=1gW4otwb7yPqyAbP3YUiFO08Wnw_HnS8t"
}

def download_and_extract_zip(url, folder):
    zip_file_path = os.path.join(folder, "temp.zip")
    gdown.download(url, zip_file_path, quiet=False)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(folder)
    
    os.remove(zip_file_path)

def load_dataset(dataset_name, path=None):
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_URLS.keys())}")
    
    if path is None:
        path = os.getcwd()
        
    url = DATASET_URLS[dataset_name]
    
    dataset_folder = os.path.join(path, dataset_name)
    
    if not os.path.exists(dataset_folder):
        download_and_extract_zip(url, path)
    
    stimuli_path = os.path.join(dataset_folder, "stimuli_data")
    roi_path = os.path.join(dataset_folder, "brain_data")
    
    return stimuli_path, roi_path