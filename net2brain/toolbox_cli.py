import click
from rdm_generation import RDM
from evaluation import Evaluation
from feature_extraction import FeatureExtraction
from helper.helper import *
from helper.helper_ui import *
import json
from datetime import date
from datetime import datetime
from pprint import pprint

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



def context_warner(task, dataset, dnn, eval, roi):
    """Check if the inputs are available

    Args:
        task (str): Name of task
        dataset (str): Name of dataset
        dnn (list): List of dnns
        eval (str): Name of evaluation metric
        roi (list): List of ROIs

    Returns:
        number: 0 if input is incorrect
    """
    
    if task == "gen_feats":
        
        # Check for dataset
        if dataset not in findfolders(STIMULI_DIR):
            click.echo("######")
            click.echo('Dataset not available. Available datasets:')
            click.echo(findfolders(STIMULI_DIR))
            click.echo("######")
            return 0
        
        # Check for DNNs
        available_nets_dict = get_available_nets()
        available_netsets = list(available_nets_dict.keys())
        available_netsets_clean = [x.split(" ")[0] for x in available_netsets]
        
        
        for element in dnn:
            netset = element.split("-")[0]
            net = element.split("-")[1]
            
            if netset not in available_netsets_clean:
                click.echo("######")
                click.echo(f'Netset "{netset}" not available. Available netsets:')
                click.echo(available_netsets_clean)
                click.echo("######")
                return 0
            
            index = available_netsets_clean.index(netset)
            if net not in available_nets_dict[available_netsets[index]]:
                click.echo("######")
                click.echo(f'Net "{net}" not available in netset. Available nets in this netset:')
                click.echo(available_nets_dict[available_netsets[index]])
                click.echo("######")
                return 0
                

    if task == "create_rdm":
        
        # Check for dataset
        if dataset not in findfolders(FEATS_DIR):
            click.echo("######")
            click.echo(f'No features generated for dataset {dataset}. Feats have been generated for:')
            click.echo(findfolders(FEATS_DIR))
            click.echo("######")
            return 0
        
        # Check for DNNs
        dataset_path = op.join(FEATS_DIR, dataset)
        available_networks = findfolders(dataset_path)
        
        for element in dnn:
            netset = element.replace("-", "_")
            if netset not in available_networks:
                click.echo("######")
                click.echo(f'No feats generated for netset "{netset}" in dataset "{dataset}". Available netsets:')
                click.echo(available_networks)
                click.echo("######")
                return 0
            
            
    
    if task == "eval":
        
        # Check for dataset        
        if dataset not in findfolders(RDMS_DIR):
            click.echo("######")
            click.echo(f'No RDMs generated for dataset {dataset}. RDMs have been generated for:')
            click.echo(findfolders(RDMS_DIR))
            click.echo("######")
            return 0
        
        # Check for DNNs
        dataset_path = op.join(RDMS_DIR, dataset) # we need to know which network has rdms already created
        available_networks = findfolders(dataset_path)
        
        for element in dnn:
            netset = element.replace("-", "_")
            if netset not in available_networks:
                click.echo("######")
                click.echo(f'No RDMs generated for netset "{netset}" in dataset "{dataset}". Available RDMs:')
                click.echo(available_networks)
                click.echo("######")
                return 0
            
        # Check for ROIs
        rois_path = op.join(BRAIN_DIR, dataset)
        available_scans = findfilesonly(rois_path, type="npz")
        
        for element in roi:
            if element not in available_scans:
                click.echo("######")
                click.echo(f'THe ROI "{element}" in dataset "{dataset}" does not exist. Available ROIs:')
                click.echo(available_scans)
                click.echo("######")
                return 0
            
    
                

def click_warner(task, dataset, dnn, eval, roi):
    """Method to check if valid inputs have been given to the CLI

    Args:
        task (str): Name of task
        dataset (str): Name of dataset
        dnn (list): List of dnns
        eval (str): Name of evaluation metric
        roi (list): List of ROIs    

    Returns:
        number: 0 in case the input was incorrect
    """
    if task == "gen_feats":
        if dataset == None:
            click.echo("######")
            click.echo('Enter a valid dataset, that runs through the models (e.g. --dataset 78images)')
            click.echo("######")
            return 0
        if dnn == ():
            click.echo("######")
            click.echo('Enter a valid dnn, to generate features from (e.g. standard-AlexNet)')
            click.echo("######")
            return 0
        
    if task == "create_rdm":
        if dataset == None:
            click.echo("######")
            click.echo('Enter a valid dataset, to which feats have been created (e.g. --dataset 78images)')
            click.echo("######")
            return 0
        if dnn == ():
            click.echo("######")
            click.echo(
                'Enter a valid dnn, of which feats have been created (e.g --dnn standard-AlexNet)')
            click.echo("######")
            return 0
        
    if task == "eval":
        if dataset == None:
            click.echo("######")
            click.echo('Enter a valid dataset, to which rdms have been created (e.g. --dataset 78images)')
            click.echo("######")
            return 0
        if dnn == ():
            click.echo("######")
            click.echo('Enter a valid dnn, of which rdms have been created (e.g --dnn standard-AlexNet)')
            click.echo("######")
            return 0
        if eval == None:
            click.echo("######")
            click.echo('Enter a valid method for evaluation (e.g --eval rsa)')
            click.echo("######")
            return 0
        if roi == ():
            click.echo("######")
            click.echo('Enter valid rois for evaluation (e.g. --roi fmri_EVC_RDMs.npz)')
            click.echo("######")
            return 0
        
   
        


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




"""CLI Interface"""

@click.command()
@click.option('--layers',  type = click.Choice(["show"]))
@click.option('--task', type = click.Choice(["gen_feats", "create_rdm", "eval"]), help = "Enter which task you would like to compute")
@click.option('--dataset', help = "Enter the name of one of the available datasets")
@click.option('--dnn', multiple=True, help = "Enter one or multiple dnns in the form of 'netset-net'")
@click.option('--eval', type=click.Choice(["rsa", "wrsa", "searchlight"]), help = "Enter one of the available evaluation methods")
@click.option('--roi', multiple=True, help = "Enter one or multiple available rois")
def main(layers, task, dataset, dnn, eval, roi):
    """Main function to start the tasks of the toolbox

    Args:
        task (str): Name of task
        dataset (str): Name of dataset
        dnn (list): List of dnns
        eval (str): Name of evaluation metric
        roi (list): List of ROIs
    """
    
    click.echo("")
    
    if layers != None:
        if dnn != ():
            netset = dnn[0].split("-")[0]
            net = dnn[0].split("-")[1]
            
            extract = FeatureExtraction(net, "78images", netset)
            pprint(extract.get_all_nodes())

        else:
            click.echo("######")
            click.echo('Enter a valid dnn, to find all available layers (e.g --dnn standard-AlexNet)')
            click.echo("######")
    else:    
        if click_warner(task, dataset, dnn, eval, roi) != 0:
            if context_warner(task, dataset, dnn, eval, roi) != 0:
            
            
                if task == "gen_feats":      
                    for element in dnn:
                        netset = element.split("-")[0]
                        net = element.split("-")[1]
                        extract = FeatureExtraction(net, dataset, netset)
                        extract.start_extraction()
                    
                elif task == "create_rdm":
                    for element in dnn:
                        net = element.replace("-", "_")
                        save_path = op.join(RDMS_DIR, dataset, net)
                        feats_data_path = op.join(FEATS_DIR, dataset, net)
                        rdm = RDM(save_path, feats_data_path)
                        rdm.create_rdms()
                    
                elif task == "eval":
                    if eval == "rsa":
                        eval = "RSA"
                    elif eval == "wrsa":
                        eval = "Weighted RSA"
                    elif eval == "searchlight":
                        eval = "Searchlight"
                        
                    chosen_networks = [x.replace("-", "_") for x in dnn]
                    json_dir = create_json(chosen_networks, dataset, roi, eval)
                    evaluator = Evaluation(json_dir)
                    evaluator.show_results()
            

if __name__ == '__main__':
    main()
