import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/code/') # add upper code/ directory to path
import config_global
from huggingface_hub import snapshot_download

remote_path = f"data/volumes_subset"
local_path = config_global.dir_global+ '/' + remote_path

if os.path.isdir(local_path):
    print(f"{local_path} already exists. Skipping.")
else:
    print(f"Downloading {remote_path}...")
    snapshot_download(
        repo_id="didsr/tsynth",
        use_auth_token=True,
        repo_type="dataset",
        allow_patterns=remote_path + "/*",
        local_dir=config_global.dir_global,
        local_dir_use_symlinks=False
    )