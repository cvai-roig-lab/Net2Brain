from cv2 import ROTATE_90_CLOCKWISE, rotate, rotatedRectangleIntersection
import matplotlib.pyplot as plt
import os
import pickle
from numpy import angle
import pandas as pd
from collections import defaultdict
from prettytable import PrettyTable
import json
import os.path as op
import pickle

from sympy import rotations
from helper.helper import *
from pprint import pprint

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



class Plotting2():
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

            # group_dict = {}

            # result_dict = defaultdict(lambda: defaultdict(float))
            # errobar_dict = defaultdict(lambda: defaultdict(float))
            # lnc_unc_dict = defaultdict(list)
            
            
            for brain_scan, nets in brain_data.items():
                all_networks = []
                all_layers = []
                all_r2 = []
                all_sig = []
                all_sem = []
                all_nc = []
                for network, layer_and_results in nets.items():
                    layer_list = []
                    r2 = []
                    sig = []
                    sem = []
                    lnc_unc = []
                    for layer, results in layer_and_results.items():
                        layer_list.append(layer)
                        r2.append(results[0])
                        sig.append(results[2])
                        sem.append(results[3])
                        lnc_unc.append(results[4])
                    all_layers.append(layer_list)
                    all_r2.append(r2)
                    all_sig.append(sig)
                    all_sem.append(sem)
                    all_nc.append(lnc_unc)
                    all_networks.append(network)
                
                labels = all_layers[1]

                men_means = all_r2[0]
                women_means = all_r2[1]

                x = np.arange(len(labels))  # the label locations
                width = 0.32  # the width of the bars

                fig, ax = plt.subplots()
                rects1 = ax.bar(x - width/2, men_means, width, label=all_networks[0])
                rects2 = ax.bar(x + width/2, women_means, width, label=all_networks[1])

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('R²')
                ax.set_title("Results of evaluation")
                ax.set_xticks(x- width / 2, labels, rotation = 45, fontsize=8)
                
                ax.legend(loc="upper right", prop={'size': 6})
                
                # Noise Ceiling
                for k in x:
                    ax.fill_between((k - width / 2, k + width / 2), all_nc[0][0][0], all_nc[0][0][1], color='gray')
                
                # For asterix
                counter = 0
                sublist = 0
                for k in ax.patches:

                    if counter <= len(labels)-1:
                        compare = all_sig[sublist][counter]
                        counter += 1
                    else:
                        counter = 0
                        sub_list = 1
                        compare = all_sig[sub_list][counter]
                        counter += 1

                    if compare <= 0.05:
                        ax.annotate("*", (k.get_x() + k.get_width() / 2., k.get_height()),
                                    ha='center', va='center', xytext=(0, 7), textcoords='offset points')

                plt.errorbar(x- width / 2, men_means, yerr= all_sem[0], fmt=' ', ecolor = 'black', color='black')
                plt.errorbar(x+ width / 2, women_means, yerr= all_sem[1], fmt=' ', ecolor = 'black', color='black')
                
                
                ax.set_xlabel(brain_scan)

                fig.tight_layout()
                
                print("ANPIT TTT")

                plt.savefig(self.save_path + os.sep + brain_scan + "_graph_" + dataset_name + ".svg", bbox_inches="tight")  # save plot




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

            # group_dict = {}

            # result_dict = defaultdict(lambda: defaultdict(float))
            # errobar_dict = defaultdict(lambda: defaultdict(float))
            # lnc_unc_dict = defaultdict(list)
            
            
            for brain_scan, nets in brain_data.items():
                all_networks = []
                all_layers = []
                all_r2 = []
                all_sig = []
                all_sem = []
                all_nc = []
                for network, layer_and_results in nets.items():
                    layer_list = []
                    r2 = []
                    sig = []
                    sem = []
                    lnc_unc = []
                    for layer, results in layer_and_results.items():
                        layer_list.append(layer)
                        r2.append(results[0])
                        sig.append(results[2])
                        sem.append(results[3])
                        lnc_unc.append(results[4])
                    all_layers.append(layer_list)
                    all_r2.append(r2)
                    all_sig.append(sig)
                    all_sem.append(sem)
                    all_nc.append(lnc_unc)
                    all_networks.append(network)
                
                labels = all_layers[1]

                men_means = all_r2[0]
                women_means = all_r2[1]

                x = np.arange(len(labels))  # the label locations
                width = 0.32  # the width of the bars

                fig, ax = plt.subplots()
                rects1 = ax.bar(x - width/2, men_means, width, label=all_networks[0])
                rects2 = ax.bar(x + width/2, women_means, width, label=all_networks[1])

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('R²')
                ax.set_title("Results of evaluation")
                ax.set_xticks(x- width / 2, labels, rotation = 45, fontsize=8)
                
                ax.legend(loc="upper right", prop={'size': 6})
                
                # Noise Ceiling
                for k in x:
                    ax.fill_between((k - width / 2, k + width / 2), all_nc[0][0][0], all_nc[0][0][1], color='gray')
                

                counter = 0
                sublist = 0
                for k in ax.patches:

                    if counter <= len(labels)-1:
                        compare = all_sig[sublist][counter]
                        counter += 1
                    else:
                        counter = 0
                        sub_list = 1
                        compare = all_sig[sub_list][counter]
                        counter += 1
                          
                    if compare <= 0.05:
                        ax.annotate("*", (k.get_x() + k.get_width() / 2., k.get_height()), ha='center', va='center', xytext=(0, 7), textcoords='offset points')

                plt.errorbar(x- width / 2, men_means, yerr= all_sem[0], fmt=' ', ecolor = 'black', color='black')
                plt.errorbar(x+ width / 2, women_means, yerr= all_sem[1], fmt=' ', ecolor = 'black', color='black')
                
                ax.set_xlabel(brain_scan)

                fig.tight_layout()

                plt.savefig(brain_scan + "_graph_" + dataset_name + ".svg", bbox_inches="tight")  # save plot
                        
                        


file_name = op.join(r'C:\Users\domen\Documents\GitHub\Neural-Toolbox-Dev\output_data\18.05.2022\16h19m41s', 'total.pickle')

Plotting(file_name).plot()
