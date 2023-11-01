from net2brain.feature_extraction import all_networks
import os
import pandas as pd
from pprint import pprint

def show_all_architectures():
    """Returns available models.

    Returns
    -------
    dict
        Available models by netset.
    """

    print("\n")
    for key, values in all_networks.items():
        print(f"NetSet: {key}")
        print(f"Models: {[v for v in values]}")
        print("\n")
    return

def show_all_netsets():
    """Returns available netsets.

    Returns
    -------
    list
       Available netsets.
    """

    return list(all_networks.keys())



def print_netset_models(netset):
    """Returns available models of a given netset.

    Parameters
    ----------
    netset : str
        Name of netset.

    Returns
    -------
    list
        Available models.

    Raises
    ------
    KeyError
        If netset is not available in the toolbox.
    """
    if netset in list(all_networks.keys()):
        return all_networks[netset]
    else:
        raise KeyError(
            f"This netset '{netset}' is not available. Available netsets are", 
            list(all_networks.keys())
        )



def open_taxonomy():

    file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(file_path)

    taxonomy_path = os.path.join(directory_path, "architectures/taxonomy.csv")

    # Read the CSV file
    df = pd.read_csv(taxonomy_path)

    # Replace "x" with 1 and empty cells with 0
    df = df.replace({'x': 1, '': 0, pd.NA: 0})

    # Drop the 'Unnamed: 34' column if it exists
    if 'Unnamed: 34' in df.columns:
        df = df.drop(columns=['Unnamed: 34'])

    return df



def find_model_like_name(model_name, df=None):
    """Find models containing the given string. Way of finding a model within \
        the model zoo.

    Parameters
    ----------
    name : str
        Name models.
    """

    # Get taxnomy df:

    if df is None:
        df = open_taxonomy()

    # Filter the DataFrame based on whether the model_name is a substring of the 'Model' column values
    similar_models_df = df[df['Model'].str.contains(model_name, case=False)]

    # Drop columns that do not contain a "1" in any row, except for "Model" and "Netset"
    columns_to_drop = [col for col in similar_models_df.columns if col not in ['Model', 'Netset'] and not (similar_models_df[col] == 1).any()]
    similar_models_df = similar_models_df.drop(columns=columns_to_drop)

    return similar_models_df




def find_model_by_custom(category, model_name=None):
    df = open_taxonomy()

    try:
        if isinstance(category, list):
            # Filter the DataFrame based on whether all of the category columns have a 1
            filtered_df = df[df[category].eq(1).all(axis=1)]
        else:
            # Filter the DataFrame based on whether the category column has a 1
            filtered_df = df[df[category] == 1]
    except KeyError:
        print(f"Column '{category}' not found in the DataFrame.")
        print("Available columns are:")
        print(df.columns.tolist())
        return None

    if model_name != None:
        # Apply the find_model_like_name functionality to the filtered DataFrame
        result = find_model_like_name(model_name, filtered_df)
    else:
        result = filtered_df

    # Drop columns that do not contain a "1" in any row, except for "Model" and "Netset"
    columns_to_drop = [col for col in result.columns if col not in ['Model', 'Netset'] and not (result[col] == 1).any()]
    result = result.drop(columns=columns_to_drop)

    return result




def show_taxonomy():
    df = open_taxonomy()

    # Filter rows where 'Header' equals 'Header Category'
    filtered_df = df.loc[df['Header'] == 'Header Category']

    # Get unique values from the filtered DataFrame
    unique_values = list(set(filtered_df.values.flatten()))
    unique_values = [str(x) for x in unique_values]

    # Remove columns if they exist in the unique_values list
    columns_to_remove = ['Header Category', 'nan', 'Model', 'Netset']
    for column in columns_to_remove:
        if column in unique_values:
            unique_values.remove(column)

    # Create a dictionary with unique values as keys and corresponding column names as values
    result = {}
    for value in unique_values:
        columns_with_value = filtered_df.columns[filtered_df.eq(value).any()].tolist()
        result[value] = columns_with_value

    pprint(result)


def find_model_by_dataset(dataset_name):
    return find_model_by_custom(dataset_name)


def find_model_by_training_method(training_method):
    return find_model_by_custom(training_method)


def find_model_by_visual_task(visual_task):
    return find_model_by_custom(visual_task)
