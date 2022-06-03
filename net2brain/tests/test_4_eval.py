
from prettytable import PrettyTable
import sys
sys.path.append(r'net2brain')
import os
from os import path as op
from helper.helper import get_paths
from evaluation import Evaluation
from evaluations.rsa import RSA
from toolbox_ui import create_json
import filecmp
import glob


def create_table(dict_of_each_scan, save_path, kind="total"):
    """This function takes the dictionary as input and creates a pretty table and saves
    all the results in a txt

    Args:
        dict_of_each_scan (dict): dictionary with all values from evaluation
        save_path (str/path): path to where we want to save the table file
        kind (str, optional): Tells what kind of table that is (total or max). Defaults to "total".
    """

    lables = ["Dataset", "Brain Region", "Network", "Layer",
            "R²", "Noise Ceiling %", "Significance", "Sem"]
    table = PrettyTable(lables)

    list_for_txt = [["Dataset", "Brain Region", "Network", "Layer",
                    "R²", "Noise Ceiling %", "Significance", "Sem"]]  # for .txt-file

    for data_set, brain_scan_values in dict_of_each_scan.items():
        for brain_scan, net_values in brain_scan_values.items():
            for net, layer_values in net_values.items():
                for layer, results in layer_values.items():
                    new_row = [data_set, brain_scan[4:], net[4:], layer[4:], results[0], results[1],
                            results[2], results[3]]
                    list_for_txt.append(new_row)  # for .txt-file
                    table.add_row(new_row)

    with open(save_path + "/" + kind + '_data.txt', 'w') as f:  # save in .txt
        for item in list_for_txt:
            f.write("%s\n" % item)

    print(table)


def get_best_layer(dict_of_each_scan):
    """This function looks for the best R² Value in each sub-sub-sub dict and returns
    a dict of the same structure, but with only the best performing layer

    Args:
        dict_of_each_scan (dict): dictionary with all values from evaluation

    Returns:
        (dict): dictionary with only the best values from evaluation for each network
    """

    maximum_dict = {}
    for data_set, brain_scan_values in dict_of_each_scan.items():
        data_dict = {}
        for brain_scan, net_values in brain_scan_values.items():
            brain_dict = {}
            for net, layer_values in net_values.items():
                max_value = 0
                if len(net) > 1:
                    for layer, results in layer_values.items():
                        if results[0] > max_value:
                            max_key = layer
                            max_value = results[0]
                            max_results = results

                    # Starting from here we recreate our dict structure
                    layer_results_dict = {max_key: max_results}
                    net_best_layer_dict = {net: layer_results_dict}
                    brain_dict.update(net_best_layer_dict)

                else:
                    # rebuild the original dict structure
                    net_best_layer_dict = {net: layer_values}
                    brain_dict.update(net_best_layer_dict)

            new_brain_dict = {brain_scan: brain_dict}
            data_dict.update(new_brain_dict)
        new_data_dict = {data_set: data_dict}
        maximum_dict.update(new_data_dict)

    return maximum_dict


def test_RSA():
    
    """Write down all relevant paths"""
    PATH_COLLECTION = get_paths()
    CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]
    
    # Set path for saving activations    
    path = op.join(CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_eval")
    path_truth = op.join(CURRENT_DIR, "net2brain/tests/compare_files/correct_data_eval")
    
    # Create folder if it does not exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    json_dir = op.join(CURRENT_DIR, "net2brain/tests/compare_files/to_be_tested_eval")
    

    RSA_Metric = RSA(json_dir)
    RSA_Metric.test = True
    final_dict = RSA_Metric.evaluation()
    
    # Draw table into the console
    create_table(final_dict, json_dir)

    # Filter out best performing layers per network
    maximum_dict = get_best_layer(final_dict)

    # create table for maximum layers
    create_table(maximum_dict, json_dir, kind="max")

    
    # Compare extractions with ground truth
    to_test = glob.glob(op.join(path, "*.txt"))
    to_compare = glob.glob(op.join(path_truth, "*.txt"))

    for i, truth in enumerate(to_compare):
        status = filecmp.cmp(truth, to_test[i])
        print(truth.split(os.sep)[-1], to_test[i].split(os.sep)[-1], status)
        if status == False:
            return status

    return status
