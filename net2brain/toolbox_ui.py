from tkinter import CURRENT
from helper.helper import *
from helper.helper_ui import *
import os
import os.path as op
import json
from datetime import date
from datetime import datetime
from rdm_generation import RDM
from evaluation import Evaluation
from feature_extraction import FeatureExtraction


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


def clear():
    """Function to clean up the user interface"""
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    
def create_json(json_networks, json_dataset, json_scans, json_metrics):
    """Function to save arguments for evaluation in a .json-file

    Args:
        json_networks (list): list of networks
        json_dataset (list): list of datasets
        json_scans (list): list of brain ROIs
        json_metrics (list): list of selected metrics

    Returns:
        [str/Path]: path to where the file will be saved
    """
    
    # Save arguments in folder for evaluation
    day = date.today()
    day = day.strftime("%d.%m.%Y")
    time = datetime.now()
    time = time.strftime("%Hh%Mm%Ss")

    # Prepare file and folder
    json_dir = op.join(PARENT_DIR, 'output_data', day, time)

    if not op.exists(json_dir):
        os.makedirs(json_dir)

    # saves arguments used for creating RDMs
    args_file = op.join(json_dir, 'args.json')
    log_file = op.join(PARENT_DIR, 'output_data', 'log.json')
    args = [{
        "1_Day": day,
        "2_Time": time,
        "3_Networks": json_networks,
        "4_Dataset": json_dataset,
        "5_ROIs": json_scans,
        "6_Metric": json_metrics}]

    # write individual json
    with open(args_file, 'w') as fp:
        json.dump(args, fp, sort_keys=True, indent=4)

    # write log json
    if op.isfile(log_file):  # if exists append
        with open(log_file, 'r') as fp:
            data = json.load(fp)
            data.append(args)
        with open(log_file, 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)
    else:
        with open(log_file, 'w') as fp:
            json.dump(args, fp, sort_keys=True, indent=4)

    return json_dir


class UserInterface:
    """ This class is the entire UserInterface. It allows the user to select between 
    "Generate Features", "Create RDMs" and "Evaluation" """
    
    def __init__(self):
        os.chdir(BASE_DIR)
        
        # Which task do we want to do?
        question = "What do you want to do?"
        task_list = ["Generate Features", "Create RDMs", "Evaluation", "exit"]
        self.task = UI_Table(question, "Option", task_list)
        
        if self.task == "Generate Features":
            self.task_gen_feats()
        elif self.task == "Create RDMs":
            self.task_gen_rdm()
        elif self.task == "Evaluation":
            self.task_evaluation()
        else:
            pass
        
    
    def print_done(self):
        """ Helping function to show the user that a process has been completed """
        
        print("")
        print("####################")
        print("Done")
        print("####################")
        print("")
        
    
    def task_gen_feats(self):
        """Subfunction for "Generate Features" """
        
        # Select Dataset
        question = "Which Data-Set:"
        available_images = findfolders(STIMULI_DIR)
        dataset = UI_Table(question, "Data-Set", available_images)
        if dataset == 0:
            self.__init__()
            return 0
            
        # Select Network-Set
        dict_all_networks = get_available_nets()  # returns dict with containing all sets with their networks
        question = "Enter Network Number from:"
        network_set = UI_Table(question, "Network-Set", list(dict_all_networks.keys()))
        if network_set == 0:
            self.__init__()
            return 0
        
        
        # Select actual network
        question = "Enter " + network_set + "-Network Number from:"
        network = UI_Table(question, "Network", dict_all_networks[network_set])
        
        if network == 0:
            self.__init__()
            return 0
        else:
            network_set = str(network_set.split(" ")[0])  # exclude amount from set name
            extract = FeatureExtraction(network, dataset, network_set)
            extract.start_extraction()
            
            self.print_done()
            self.__init__()
            
    
    def task_gen_rdm(self):
        """Subfunction for "Create RDMs" """
        
        # Choose Data-Set
        question = "Which Data-Set:"
        available_images = findfolders(FEATS_DIR)
        dataset = UI_Table(question, "Data-Set", available_images)
        if dataset == 0:
            self.__init__()
            return 0
            
        # Choose Network
        question = "Enter Network Number:"
        dataset_path = op.join(FEATS_DIR, dataset)
        available_networks = findfolders(dataset_path)  # we need to know which network has features already created
        network = UI_Table(question, "Network", available_networks)
        
        if network == 0:
            self.__init__()
            return 0
        else:
            # Set paths
            save_path = op.join(RDMS_DIR, dataset, network)
            feats_data_path = op.join(FEATS_DIR, dataset, network)


            rdm = RDM(save_path, feats_data_path)
            rdm.create_rdms()
            
            self.print_done()
            self.__init__()
        
        
    def task_evaluation(self):
        """Subfunction for "Evaluation" """
        
        # Choose Metric:
        question = "Which Metric:"
        available_metrics = get_available_metrics()
        metric = UI_Table(question, "Metric", available_metrics)
        if metric == 0:
            self.__init__()
            return 0
        
        
        # Choose Data-Set
        question = "Which Data-Set:"
        available_images = findfolders(RDMS_DIR)
        dataset = UI_Table(question, "Data-Set", available_images)
        if dataset == 0:
            self.__init__()
            return 0
            
        # Choose Network
        question = "Enter Network Number:"
        dataset_path = op.join(RDMS_DIR, dataset) # we need to know which network has rdms already created
        available_networks = findfolders(dataset_path)
        chosen_networks = UI_Table_Multiple(question, "Network", available_networks)
        if chosen_networks == 0:
            self.__init__()
            return 0
            
        # Choose ROIs
        question = "Which Scan(s) for " + dataset + "?"
        rois_path = op.join(BRAIN_DIR, dataset)
        available_scans = findfilesonly(rois_path, type="npz")
        chosen_scans = UI_Table_Multiple(question, "ROI", available_scans)
        
        if chosen_scans == 0:
            self.__init__()
            return 0
        else:
            # create a json for later access
            json_dir = create_json(chosen_networks, dataset, chosen_scans, metric)
        
            clear()
            
            evaluator = Evaluation(json_dir)
            evaluator.show_results()
            
            self.print_done()
            self.__init__()
     

def main():
    """This function is helpful for google colab"""
    
    UserInterface()
    


if __name__ == "__main__":
    UserInterface()
