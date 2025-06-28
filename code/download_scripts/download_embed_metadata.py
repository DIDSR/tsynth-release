from huggingface_hub import hf_hub_download
import shutil

# Download each result
remote_path = f"data/embed_metadata"
local_path = "./" # metadata should be in code/ directory

print(f"Downloading {remote_path}...")
for filename in ['EMBED_split_dbt.csv', 'EMBED_split_dm.csv', 'existing_Embed_fileset.pkl']:
    hf_hub_download(repo_id="didsr/tsynth",
                    filename=f"{remote_path}/{filename}",
                    repo_type="dataset",
                    local_dir=local_path,
                    local_dir_use_symlinks=False)
    shutil.move(f"{remote_path}/{filename}", filename)

shutil.rmtree('data/')  # to download to code and delete data/ and data/embed_metadata


