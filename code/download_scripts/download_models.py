import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/code/') # add upper code/ directory to path
import argparse
import config_global
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, help="Model type (e.g., DBT, DM, diffusion)", default='DBT')
parser.add_argument("--all", help="Download all pretrained model types", action="store_true", default=False)

args = parser.parse_args()

# Define all model types (same as in results)
ALL_MODEL_TYPES = ['DBT', 'DM', 'diffusion']

# Decide which ones to download
if args.all:
    model_types = ALL_MODEL_TYPES
else:
    model_types = [args.type]

# Download each model 
for model_type in tqdm(model_types):
    remote_path = f"data/pretrained_models/{model_type}"
    local_path = config_global.dir_global+ '/' + remote_path

    if os.path.isdir(local_path):
        print(f"{local_path} already exists. Skipping.")
        continue

    print(f"Downloading {remote_path}...")
    snapshot_download(
        repo_id="didsr/tsynth",
        use_auth_token=True,
        repo_type="dataset",
        allow_patterns=remote_path + "/*",
        local_dir=config_global.dir_global,
        local_dir_use_symlinks=False
    )