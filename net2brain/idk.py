import click
from rdm_generation import RDM
from evaluation import Evaluation
from feature_extraction import FeatureExtraction
from helper.helper import *
from helper.helper_ui import *
import json
from datetime import date
from datetime import datetime


# Check for DNNs
available_nets_dict = get_available_nets()
available_netsets = list(available_nets_dict.keys())


for set in available_netsets:
    all_nets = available_nets_dict[set]
    for net in all_nets:
        print(" - " + set.split(" ")[0] + "_" + net)

