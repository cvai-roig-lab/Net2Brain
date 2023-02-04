import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from collections import defaultdict
from prettytable import PrettyTable
import json
import os.path as op
import pickle
from helper.helper import *

from evaluations.rsa import RSA
from evaluations.weighted_rsa import WRSA
from evaluations.searchlight import Searchlight

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



class Plotting():
    """Class to plot the final graph
    """
    def __init__(self, pickle_file):
        """Initiate plotting

        Args:
            pickle_file (str/path): path to pickle file
        """
        
        self.pickle_file = pickle_file
        self.save_path = os.path.split(pickle_file)[0]
        
        with open(self.pickle_file, 'rb') as data_file:
            self.maximum_dict = pickle.load(data_file)
            

    def get_max_nc(self, data_dict):
        max = 0
        for layer, lnc_unc in data_dict.items():
            if lnc_unc[1] > max:
                max = lnc_unc[1]
        return max
            
        
    def plot(self):
        for dataset_name, brain_data in self.maximum_dict.items():

            plt.rcParams.update({'font.size': 14})
            colours = ['#fbb4ae',
                    '#b3cde3',
                    '#ccebc5',
                    '#decbe4',
                    '#fed9a6',
                    '#ffffcc',
                    '#e5d8bd',
                    '#fddaec',
                    '#f2f2f2',
                    '#b3e2cd',
                    '#fdcdac',
                    '#cbd5e8',
                    '#f4cae4',
                    '#e6f5c9',
                    '#fff2ae',
                    '#f1e2cc',
                    '#cccccc']

            group_dict = {}

            result_dict = defaultdict(lambda: defaultdict(float))
            errobar_dict = defaultdict(lambda: defaultdict(float))
            lnc_unc_dict = defaultdict(list)

            layers = []
            for brain_scan, nets in brain_data.items():  # for each scan filter out the information we need to plot
                networks = []
                
                group_dict[brain_scan] = {}

                for net, layer_and_results in nets.items():
                    networks.append(net)

                    for layer, results in layer_and_results.items():
                        if brain_scan not in lnc_unc_dict.keys():
                            lnc_unc_dict[brain_scan] = results[4]
               
                        #layers.append(layer[4:].split(".")[-2])
                        layers.append(layer[4:])
                        group_dict[brain_scan][net] = results[0]  # ROI: R² values
                        result_dict[net][brain_scan] = results[0]
                        errobar_dict[net][brain_scan] = results[3]

            for brain_scan, nets in brain_data.items():  # for each scan filter out the information we need to plo
                correlation_data_new = []
                error_bars = []
                for net, layer_and_results in nets.items():
                    correlation_data_new.append(result_dict[net])
                    error_bars.append(errobar_dict[net])
                break

            legend = networks

            df = pd.DataFrame(correlation_data_new, index=legend).transpose()
            df_error = pd.DataFrame(error_bars, index=legend).transpose()
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

            df.plot.bar(ax=ax, yerr=df_error, color=colours[:len(error_bars)])
            ax.set_xlabel('(The best performing layers)')
            ax.set_ylabel('$R^{2}$')

            if lnc_unc_dict[max(lnc_unc_dict)][1] != None:  # if we have noise ceiling
                max_value = self.get_max_nc(lnc_unc_dict)
                ax.set_ylim([0, max_value + 0.1])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            width = 0.44
            if lnc_unc_dict[max(lnc_unc_dict)][1] != None:  # if we have noise ceiling
                for k, brain_scan in enumerate(brain_data.keys()):
                    ax.fill_between((k - width / 2, k + width / 2), lnc_unc_dict[brain_scan][0],
                                    lnc_unc_dict[brain_scan][1], color='gray', alpha=0.5)


            cleaned_layers = []
            amount_net = len(networks)
            for counter, net_single in enumerate(networks):
                cleaned_layers= cleaned_layers + layers[counter::len(networks)]
                
                
            for counter, p in enumerate(ax.patches):
                ax.annotate("*", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 7), textcoords='offset points')
                
                ax.annotate(cleaned_layers[counter], (p.get_x() +
                            p.get_width() / 2., p.get_height()/2), rotation = 90, fontsize=10)
                

            ax.set_title("Results of Evaluation")
            ax.legend(loc="upper right", prop={'size': 10})

            plt.savefig(self.save_path + os.sep + "graph_" + dataset_name +
                        ".svg", bbox_inches="tight")  # save plot
            plt.show()

        


class Evaluation():
    """Evaluation module that decides which evaluation practice to choose
    """
    def __init__(self, json_dir):
        """Initialize Evaluation

        Args:
            json_dir (str/path): path to json dir
        """
        
        json_file = op.join(json_dir, "args.json")
        
        with open(json_file) as json_file:
            data = json.load(json_file)[0]

        self.json_dir = json_dir
        self.day = data["1_Day"]
        self.time = data["2_Time"]
        self.networks = data["3_Networks"]
        self.dataset = data["4_Dataset"]
        self.rois = data["5_ROIs"]
        self.metric = data["6_Metric"]
        
        
        if self.metric == "RSA":
            RSA_Metric = RSA(json_dir)
            self.final_dict = RSA_Metric.evaluation()
        elif self.metric == "Searchlight":
            SL_Metric = Searchlight(json_dir)
            self.final_dict = SL_Metric.evaluation()
        elif (self.metric == "Weighted RSA") or (self.metric == "Weighted RSA (predicting layers)"):
            WRSA_Metric = WRSA(json_dir)
            self.final_dict = WRSA_Metric.evaluation()
        # elif self.metric == "Weighted RSA":
        #     WRSA_Metric = WRSA(json_dir)
        #     self.final_dict = WRSA_Metric.evaluation()
        # elif self.metric == "Weighted RSA (predicting layers)":
        #     WRSA_Metric = WRSA_ROI(json_dir)
        #     self.final_dict = WRSA_Metric.evaluation()
            
            
            
    def create_table(self, dict_of_each_scan, save_path, kind="total"):
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
        
        
    def create_table_searchlight(self, dict_of_each_scan, save_path, kind="total"):
        """This function takes the dictionary as input and creates a pretty table and saves
        all the results in a txt

        Args:
            dict_of_each_scan (dict): dictionary with all values from evaluation
            save_path (str/path): path to where we want to save the table file
            kind (str, optional): Tells what kind of table that is (total or max). Defaults to "total".
        """

        lables = ["Dataset", "Brain Region", "Network", "Layer",
                "R²", "Significance", "Sem"]
        table = PrettyTable(lables)

        list_for_txt = [["Dataset", "Brain Region", "Network", "Layer",
                        "R²", "Significance", "Sem"]]  # for .txt-file

        for data_set, brain_scan_values in dict_of_each_scan.items():
            for brain_scan, net_values in brain_scan_values.items():
                for net, layer_values in net_values.items():
                    for layer, results in layer_values.items():
                        new_row = [data_set, brain_scan[4:], net[4:], layer[4:], results[0],
                                results[2], results[3]]
                        list_for_txt.append(new_row)  # for .txt-file
                        table.add_row(new_row)

        with open(save_path + "/" + kind + '_data.txt', 'w') as f:  # save in .txt
            for item in list_for_txt:
                f.write("%s\n" % item)

        print(table)
        
    
    def get_average_of_layers(self, dict_of_each_scan):
        """Function to get the average performance of a layer over all ROIs

        Args:
            dict_of_each_scan (dict): dictionary with all values from evaluation

        Returns:
            average_dict (dict): dictionary that contains only the average values
        """
   
        net_dict = {} 
        
        counter = 0
        for data_set, brain_scan_values in dict_of_each_scan.items():
            for brain_scan, net_values in brain_scan_values.items():
                for net, layer_values in net_values.items():
                    for layer, results in layer_values.items():

                        if net in net_dict:  # if key already exists
                            if layer in net_dict[net]:  # if key already exists
                                current = net_dict[net][layer]
                                new = [current[0] + results[0], current[1], current[2] + results[2], current[3] + results[3], current[4]]  # Add up the relevant values
                                net_dict[net][layer] = new
                                
                            else:
                                net_dict[net][layer] = results  # if key does not exist, create entry
                        else:
                            net_dict[net] = layer_values  # if key does not exist, create entry
                            break
                counter += 1  # counter for divison
                        
                        
        for net, layer_values in net_dict.items():
            for layer, results in layer_values.items():
                
                current = net_dict[net][layer]
                new = [current[0]/counter, current[1], current[2]/counter, current[3]/counter, current[4]]  # divide entries through counter
                net_dict[net][layer] = new 
                
        
        average_dict = {"searchlight": {"All ROIs average": net_dict}}
        return average_dict
                                
                                
                                

        
        


    def get_best_layer(self, dict_of_each_scan):
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


    def save_pickle(self, full_dict, small_dict, json_dir):
        """Saves the dictionary as pickle
        Args:
            full_dict (dict): Dictionary of each scan
            small_dict (dict): Dictionary of each scan, but only the best layers
            json_dir (str/path): Path to json dir
        """

        with open(json_dir + "/total.pickle", 'wb') as handle:
            pickle.dump(full_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(json_dir + "/max.pickle", 'wb') as handle:
            pickle.dump(small_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def show_results(self):
        """Function to show results and call the plotting function
        """
        
        # Draw table into the console
        if self.metric == "Searchlight":
            self.create_table_searchlight(self.final_dict, self.json_dir)
            # Collect average over all ROIs
            shortened_dict = self.get_average_of_layers(self.final_dict)
            # Plot average performance
            self.create_table_searchlight(shortened_dict, self.json_dir, kind="max")
            # Save tables as pickle
            self.save_pickle(self.final_dict, shortened_dict, self.json_dir)
        else:
            self.create_table(self.final_dict, self.json_dir)
            # filter out maximum layers
            shortened_dict = self.get_best_layer(self.final_dict)
            # create table for maximum layers
            self.create_table(shortened_dict, self.json_dir, kind="max")            
            # Save tables as pickle
            self.save_pickle(self.final_dict, shortened_dict, self.json_dir)

        if shortened_dict != {}:  # If dict not empty
            file_name = op.join(self.json_dir, "max.pickle")
            Plotting(file_name).plot()
        
    
            
   
                
