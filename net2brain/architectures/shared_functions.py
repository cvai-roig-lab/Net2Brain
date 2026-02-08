import warnings
import json
import importlib
from pathlib import Path
import shutil
import requests
import urllib.request


def get_function_from_module(function_string):
    module_name, function_name = function_string.rsplit('.', 1)

    # Handle hierarchical module names
    segments = module_name.split('.')

    if module_name.startswith("."):
        # For relative imports, use importlib.import_module
        if ".implemented_models" in module_name:
            package = "net2brain.architectures"
            module = importlib.import_module(".".join(segments), package=package)
        else:
            raise ValueError("Relative import without '.implemented_models' is not supported.")
    else:
        module = __import__(segments[0])
        for segment in segments[1:]:
            module = getattr(module, segment)
    
    return getattr(module, function_name)


def download_to_path(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)  # atomic-ish move


def download_github_folder(owner, repo, repo_path, out_dir):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
    r = requests.get(url)
    r.raise_for_status()

    for item in r.json():
        if item["type"] == "file":
            out = Path(out_dir) / item["name"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(requests.get(item["download_url"]).content)
        elif item["type"] == "dir":
            download_github_folder(
                owner,
                repo,
                item["path"],
                Path(out_dir) / item["name"],
            )


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
    
    
    # Convert model string to function
    tokenizer_string = model_entry.get("tokenizer")
    if tokenizer_string:
        try:
            token_function = get_function_from_module(tokenizer_string)
            model_entry["tokenizer"] = token_function
        except AttributeError:
            raise ValueError(f"{tokenizer_string} is not a valid function name.")
    else:
        pass
    return model_entry

