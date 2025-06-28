## Setup

- Create and activate Python environment:
   ```
   conda create -n tsynth python=3.9
   conda activate tsynth
   pip install -r requirements.txt
   ```

- Clone repo:
   ```
   git clone https://github.com/DIDSR/tsynth-release.git
   ```

- Setup hugging face token for data download:
   -  Follow instructions on [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens) to make a token. 
   - Run and paste your token:
   ```
   huggingface-cli login
   ```

- Adjust ```DIR_GLOBAL```path in ```config_global.py``` to set global directory.

- Download Pretrained Checkpoints: this script uses [torchvision's implementation of Faster RCNN](https://pytorch.org/vision/stable/models/faster_rcnn.html). By default, the model is initialized with weights pretrained on COCO. If you cannot automatically fetch this checkpoint from online, then you may need to manually download the checkpoints for [ResNet_50_FPN_COCO](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) and [ResNet_50](https://download.pytorch.org/models/resnet50-0676ba61.pth) and move them into your user cache: ```~/.cache/torch/hub/checkpoints/```, for instance:
   ```
   cd ~/.cache/torch/hub/checkpoints
   wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
   wget https://download.pytorch.org/models/resnet50-0676ba61.pth
   ```


## Download Data
- Download T-SYNTH c-view data:
   ``` 
   python -u download_scripts/download_synthetic_data.py --all
   ```
- Download the patient (EMBED) dataset from [here](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) and metadata:
   ``` 
   python -u download_scripts/download_embed_metadata.py
   ```

- We provide pre-generated c-view images for all T-SYNTH examples, and release a few volumes (all DBT volumes will be released upon acceptance). The process of creating c-view from DBT for a single example can be replicated via:
   ``` 
   python -u download_scripts/download_volumes.py
   jupyter notebook notebooks/create_cview.ipynb
   ```
- Download synthetic digital mammography data from the [M-SYNTH](https://github.com/DIDSR/msynth-release) repo, if needed.

## Visualize Data
We provide some notebooks for visualizing T-SYNTH data:
```
cd notebooks;
jupyter notebook notebooks/dsynth_lesion_density.ipynb # Visualize T-SYNTH across lesion densities
jupyter notebook notebooks/dsynth_breast_density.ipynb # Visualize T-SYNTH across breast densities
jupyter notebook notebooks/dsynth_lesion_size.ipynb    # Visualize T-SYNTH across lesion sizes
```

## Download Pre-trained Detection Models
For each experiment we train 5 detection models. With each model we saved best check point and hyperparamters used (```hyperparams.yaml```) in ```https://huggingface.co/datasets/evsizikova/tsynth/tree/main/data/pretrained_models```, see description below:

- ```DBT/add```, ```DBT/real``` and ```DBT/repl```: include models used to evaluate effect of various ratios of patient and synthetic data used during training and associated AUC-FROC scores across different real-to-synthetic training data ratios.

- ```DBT/synth``` and ```DM/synth```: include models used to evaluate the effect of breast density, lesion size and density on synthetic-only training/testing synthetic-only training.

- ```DBT/train_on_real```, ```DM/train_on_real``` and ```DBT/train_on_real_and_synth```, ```DM/train_on_real_and_synth```: include models used for comparing the effect of adding synthetic data to patient data across modalities (DM and DBT).

- ```diffusion```: include models for comparing T-SYNTH and diffusion synthetic data (**Figure 6** in the manuscript).

   These models can be downloaded from Hugging Face via:
   ```
   python -u download_scripts/download_models.py --all
   ```
   
   To get the data for plotting you can either download the results from Hugginngface via:
   ```
   python -u download_scripts/download_results.py --all
   ```
   
   or reproduce them by evaluating each model's performance using ```evaluate_detectors.py```, as described below:
   
   ```
   DIR_GLOBAL=$(python3 -c "import config_global as cg; print(cg.dir_global)")
   EXPERIMENT_FOLDER_NAME=DBT/trained_on_real_and_synth                         # model to be evaluated
   TEST_TYPE_CONFIG='cfg/test/real.yaml'                                        # test data to be used
   export TESTRESULTS="$DIR_GLOBAL/my_results/"$EXPERIMENT_FOLDER_NAME          # location where to save .csv with results
   echo for run_id in 1 2 3 4 5; do
      export MODEL="$DIR_GLOBAL/data/pretrained_models/${EXPERIMENT_FOLDER_NAME}/${run_id}/best_ckpt.pth"
      python evaluate_detectors.py --config $TEST_TYPE_CONFIG --model_path $MODEL --save_name $TESTRESULTS'/'${run_id}/results.csv
   done
   ```
   You could choose ```EXPERIMENT_FOLDER_NAME``` from the second column [**Experiment Configuration Map**](#experiment-configuration-map) below. You can choose where to save results in ```TESTRESULTS```.
   In the third column of **Experiment Configuration Map** you could choose path to ```--config```(```TEST_TYPE_CONFIG```) file.
   
   To evaluate synthetic models ```DBT/synth``` or ```DM/synth``` and use ```--config cfg/test/synth.yaml```.
   
   #### Experiment Configuration Map

   Below is the list of experiment flags used:
   
   | Experiment Name ```EXPERIMENT_NAME```                     | Experiment Folder Name ```EXPERIMENT_FOLDER_NAME``` | Path to Config File ```TEST_TYPE_CONFIG```|  Description |
   |-------------------------------------|--------------------------|----------|-------------|
   | real                                |DBT/trained_on_real|cfg/test/real.yaml| Training on 100% real data only, using all available data |
   | real_and_synth                      |DBT/trained_on_real_and_synth|cfg/test/real.yaml| Training on a combination of real and synthetic data (unspecified ratio), using all available data |
   | synth                               |DBT/synth|cfg/test/synth.yaml| Training on synthetic data only, using all synthetic data |
   | real100                             |DBT/real/real100|cfg/test/real.yaml| Training on 100% of available real data, 800 images |
   | real80                              |DBT/real/real80|cfg/test/real.yaml| Training on 80% subset of real data |
   | real60                              |DBT/real/real60|cfg/test/real.yaml| Training on 60% subset of real data |
   | real40                              |DBT/real/real40|cfg/test/real.yaml| Training on 40% subset of real data |
   | real20                              |DBT/real/real20|cfg/test/real.yaml| Training on 20% subset of real data |
   | real80_and_synth20                  |DBT/repl/real80_and_synth20|cfg/test/real.yaml| Training on 80% real + 20% synthetic data |
   | real60_and_synth40                  |DBT/repl/real60_and_synth40|cfg/test/real.yaml| Training on 60% real + 40% synthetic data |
   | real40_and_synth60                  |DBT/repl/real40_and_synth60|cfg/test/real.yaml| Training on 40% real + 60% synthetic data |
   | real20_and_synth80                  |DBT/repl/real20_and_synth80|cfg/test/real.yaml| Training on 20% real + 80% synthetic data |
   | real100_and_synth20                 |DBT/add/real100_and_synth20|cfg/test/real.yaml| Training on 100% real + 20% synthetic data |
   | real100_and_synth40                 |DBT/add/real100_and_synth40|cfg/test/real.yaml| Training on 100% real + 40% synthetic data |
   | real100_and_synth60                 |DBT/add/real100_and_synth60|cfg/test/real.yaml| Training on 100% real + 60% synthetic data |
   | real100_and_synth80                 |DBT/add/real100_and_synth80|cfg/test/real.yaml| Training on 100% real + 80% synthetic data |
   | real100_and_synth100                |DBT/add/real100_and_synth100|cfg/test/real.yaml| Training on 100% real + 100% synthetic data |
   | diffusion_simple                    |diffusion/diffusion|cfg/test/real.yaml| Training using simply generated diffusion data |
   | diffusion_finetuned                 |diffusion/diffusion_finetuned|cfg/test/real.yaml| Training with finetuned diffusion model samples |
   | diffusion_exp_tsynth                |diffusion/tsynth|cfg/test/real.yaml| Training using T-SYNTH (experimental diffusion-based synthetic data) |
   | diffusion_exp_real_baseline        |diffusion/embed_baseline|cfg/test/real.yaml|  Real baseline experiment (1st variant), for comparison with diffusion models |
   | diffusion_exp_real_baseline_subset |diffusion/subset_embed_baseline|cfg/test/real.yaml | Real baseline (subset, 2nd variant), possibly fewer samples |
   | DM_real                             |DM/trained_on_real|cfg/test/real.yaml| Training on DM (digital mammography) real data only, using all available data |
   | DM_real_and_synth                   |DM/trained_on_real_and_synth|cfg/test/real.yaml| Training on DM data with a mix of real and synthetic samples, using all available data from synthetic and 10000 real images |
   | DM_synth                           |DM/synth|cfg/test/synth.yaml | Training on DM synthetic data only, using all available data |
   

## Evaluate Detection Models
```
cd notebooks;
jupyter notebook notebooks/synthetic_detection.ipynb             # evaluate synthetic only results (Fig. 4)
jupyter notebook notebooks/data_augmentation_experiments.ipynb   # real:synth ratios and evaluate on patient (Fig. 5)
jupyter notebook notebooks/diffusion_experiments.ipynb           # tsynth vs diffusion vs finetuned diffusion vs embed baselines (Fig. 6)
jupyter notebook notebooks/dbt_and_dm_all_images_results.ipynb   # evaluation of full EMBED data for DM and DBT models with real vs real + synth models (Fig. 9)
jupyter notebook notebooks/comparison_of_FROC_AUC.ipynb          # calculate and compare FROC AUC for real and real+synth training data replacement experiments (Fig. 10)
```

## (Optional) Train Detection Models 

Training the object detection model is done with ```train_detector.py```. To replicate the Real on Synth experiments in the paper, five independent models must be trained with either real data only or real and synthetic data. To train a single model with real data, run the following command from the terminal:

```bash
python train_detector.py --experiment EXPERIMENT_NAME --save_name <my_run>
```

Where ```EXPERIMENT_NAME``` is from column Experiment Name in [**Experiment Configuration Map**](#experiment-configuration-map).

```<my_run>``` should be unique for each trial, and is a path where the script automatically saves the model checkpoints, training logs, and hyperparameters.

e.g.:
```
DIR_GLOBAL=$(python3 -c "import config_global as cg; print(cg.dir_global)")
EXPERIMENT_NAME=real_and_synth
YOUR_EXPERIMENT_FOLDER_NAME=DBT/trained_on_real_and_synth
for run_id in "1" "2" "3" "4" "5"; do
export MODEL_SAVEDIR="${DIR_GLOBAL}/my_models/${YOUR_EXPERIMENT_FOLDER_NAME}/${run_id}/"
mkdir -p $MODEL_SAVEDIR
python train_detector.py --experiment $EXPERIMENT_NAME --save_name $MODEL_SAVEDIR
done
```
