import warnings
import json

def get_function_from_module(function_string):
    module_name, function_name = function_string.rsplit('.', 1)
    
    # Handle hierarchical module names
    segments = module_name.split('.')

    module = __import__(segments[0])
    for segment in segments[1:]:
        module = getattr(module, segment)
    
    return getattr(module, function_name)



def load_from_json(config_path, model_name):
    # Load the JSON file
    with open(config_path, 'r') as file:
        data = json.load(file)

    # Check if model_name exists in the data
    if model_name not in data:
        raise ValueError(f"{model_name} not found in the configuration file.")

    # Retrieve the attributes for the given model_name
    model_entry = data[model_name]

    # If layers are empty, set them to None
    if not model_entry.get("nodes"):
        warnings.warn("There are no layers preselected, will chose all layers")
        model_entry["nodes"] = None

    # Convert model string to function
    model_string = model_entry.get("model")
    if model_string:
        try:
            model_function = get_function_from_module(model_string)
            model_entry["model_function"] = model_function
        except AttributeError:
            raise ValueError(f"{model_string} is not a valid function name.")
    else:
        raise ValueError(f"Data for {model_name} is incomplete in the configuration file.")

    return model_entry

