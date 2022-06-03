from prettytable import PrettyTable
import os
import os.path as op
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



def findfolders(path):
    """Function that finds folders in a given path

    Args:
        path (str/path): path to folder

    Returns:
        list: list of folders in that dir
    """
    
    try:  # does this path exist?
        files = os.listdir(path)  # list all dirs we have in this path
    except:
        print("")
        print("#########################")
        print("The following path does not exist yet")
        print(path)
        print("Make sure to generate feats or RDMs first!")
        print("#########################")
        print("")
        return []
    folder_sets = []

    for f in files:
        file_or_folder = path + "//" + f
        if os.path.isdir(file_or_folder):  # Add only if its a folder - we dont want files
            if f != ".ipynb_checkpoints":  # Dont show temp colab files
                folder_sets.append(f)
                
    return folder_sets


def findfilesonly(path, type=False):
    """Function that finds files only in a given path
    Args:
        path (str/path): path to folder
        type (bool, optional): Possible a datatype we want to filter for. Defaults to False.

    Returns:
        list: list of files
    """
    files = os.listdir(path)  # list all dirs we have in this path
    folder_sets = []

    for f in files:
        file_or_folder = path + "//" + f
        if os.path.isdir(file_or_folder) is False:
            if type is not False:  # If datatype matters
                if f.split(".")[1] == type:
                    folder_sets.append(f)
            else:  # If datatype does not matter
                folder_sets.append(f)
    return folder_sets



def clear():
    """Function to clear UI Content"""
    os.system('cls' if os.name == 'nt' else 'clear')


def UI_Table(question, option_name, option_list):
    """Simple UI using Pretty Table for the simple requests
        +--------+-------------------+
        | Number | option_name       |
        +--------+-------------------+
        | (1)    | option_list       |
        | (2)    | option_list       |
        | (3)    | option_list       |
        | (4)    | option_list       |
        +--------+-------------------+
        (Press 'n' to go back)

    Args:
        question (str): Question that should stand on top
        option_name (str): Column name
        option_list (list): list of options

    Returns:
        str: return chosen element from option list
    """

    lables = ["Number", option_name]
    print("")
    print("-----------------------------")
    print(question)
    print("-----------------------------")
    print("")

    # prepare the lists/rows for the pretty table
    # number , option
    table = PrettyTable(lables)
    table.align["Number"] = "c"
    table.align[option_name] = "l"

    for i in range(len(option_list)):
        new_row = ["(" + str(i + 1) + ")", option_list[i]]  # row for the table
        table.add_row(new_row)

    print(table)
    print("(Press 'n' to go back)")

    user_input = input()

    try:  # If the input is valid, return the chosen option
        result = option_list[int(user_input) - 1]
        return result
    except:
        if user_input == "n" or user_input == "N":  # if the user wants to go back
            return 0
        else:
            print("")
            print("#########################")
            print("Invalid Input. Try again.")
            print("#########################")
            print("")
            return UI_Table(question, option_name, option_list)
        



def UI_Table_Multiple(question, option_name, available_options):
    """This function asks which Brain scans we want to compare to a certain network

    Args:
        question (str): Question that should be asked
        option_name (str): column name
        available_options (list): list of options to choose from

    Returns:
        list: selected options
    """

    rows = []
    for i in range(len(available_options)):  # prepare a list for the prettytable
        # add row of all the options
        new_row = ["(" + str(i + 1) + ")", available_options[i]]
        rows.append(new_row)

    chosen_options = []
    stop = 0
    while stop == 0:  # Just as the UI but recursive
        #clear()
        print("")
        print("-----------------------------")
        print("Select which ", option_name, "(s)")
        print("-----------------------------")
        print("")
        print(chosen_options)

        lables = ["Number", option_name]
        table = PrettyTable(lables)
        for row in rows:
            table.add_row(row)

        print(table)
        print(question)
        print("(Press 'n' to go back and 'y' to continue)")

        user_input = input()

        try:  # If the input is valid delete the row from the pretty table
            if rows[int(user_input)-1][1] == "----------":
                continue

            chosen_options.append(rows[int(user_input)-1][1])
            rows[int(user_input)-1][0] = "(x)"
            rows[int(user_input)-1][1] = "----------"

        except:
            if user_input == "n" or user_input == "N":  # if the user wants to go back
                return 0
            elif user_input == "y" or user_input == "Y":  # if the user is done
                return chosen_options
            else:
                print("")
                print("#########################")
                print("Invalid Input. Try again.")
                print("#########################")
                print("")


