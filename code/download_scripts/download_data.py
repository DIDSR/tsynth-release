import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/code/') # add upper code/ directory to path
import shutil
import argparse
import itertools
import config_global
from tqdm import tqdm
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=str, help="Breast Density", default='dense')
parser.add_argument("--size", type=str, help="Lesion Size", default='5.0')
parser.add_argument("--lesionDensity", type=str, help="Lesion Density", default='1.1')
parser.add_argument("--all", help="whether to download all data", action="store_true", default=False)

args = parser.parse_args()

if args.all:
    l_BREAST_DENSITY = ['fatty', 'scattered', 'hetero', 'dense']
    l_LESION_SIZE = ['5.0', '7.0', '9.0']
    l_LESION_DENSITY = ['1.0', '1.06', '1.1']

    l_combinations = list(itertools.product(l_BREAST_DENSITY, l_LESION_SIZE, l_LESION_DENSITY))

else:
    l_combinations = [[args.density, args.size, args.lesionDensity]]

for combo_id in tqdm(range(len(l_combinations))):
    [BREAST_DENSITY, LESION_SIZE, LESION_DENSITY] = l_combinations[combo_id]
    if BREAST_DENSITY == 'hetero' and LESION_SIZE == '9.0':
        continue
    if BREAST_DENSITY == 'dense' and LESION_SIZE == '9.0':
        continue

    filename = (
            "device_data_VICTREPhantoms_spic_"
            + LESION_DENSITY
            + "/"
            + BREAST_DENSITY
            + "/2/"
            + LESION_SIZE
            + "/"
            + "SIM.zip"
    )

    filenameZip = config_global.dir_global + "data/cview/output_cview_det_Victre/" + filename

    # download data if it is not available
    data_dir = filenameZip.replace('.zip', '/')
    if os.path.isdir(data_dir):
        print(f"{data_dir} exists;")
    else:
        # Download dataset from huggingface
        print(f"downloading data {data_dir} from huggingface...")
        print("saving to " + str(config_global.dir_global))
        hf_hub_download(
            repo_id="didsr/tsynth",
            use_auth_token=True,
            repo_type="dataset",
            local_dir=config_global.dir_global,  # download directory for this dataset
            local_dir_use_symlinks=False,
            filename='data/cview/output_cview_det_Victre/' + filename,
        )
        # Extract
        print("unzipping...")
        shutil.unpack_archive(filenameZip, os.path.dirname(filenameZip), "zip")

        print("removing zip archive..")
        os.remove(filenameZip)

