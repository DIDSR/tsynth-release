"""
Train Detector

This script trains a Faster R-CNN detector for lesion detection in DBT.
Training parameters (such as epochs, learning rate, etc.) are provided via command-line arguments,
while the dataset-specific parameters (dataset_kwargs_list, batch_size_list, and dataset_names_list)
are loaded from a YAML configuration file.
"""

import sys
import os
import yaml
import warnings
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate_detectors import calculate_iou, infer_dataset
from utils.model_utils import get_fasterrcnn_model
from utils.eval_utils import calc_correct_boxes, calc_froc, dbtex_criteria, calc_avg_value
import custom_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train Detector for lesion detection in DBT")
    # parser.add_argument('--train_data_config', type=str, default='cfg/train/real_and_synth.yaml',
    #                     help='Path to YAML config file containing dataset_kwargs_list, batch_size_list, and dataset_names_list')

    parser.add_argument('--experiment',
    type=str,
    choices=[
        'real', 'real_and_synth','synth',
        'real100','real80','real60','real40','real20',
        'real80_and_synth20','real60_and_synth40', 'real40_and_synth60','real20_and_synth80',
        'real100_and_synth20','real100_and_synth40','real100_and_synth60','real100_and_synth80','real100_and_synth100',
        'genAI_diffusion_simple','genAI_diffusion_finetuned','genAI_tsynth','genAI_real_baseline','genAI_real_baseline_subset',
        'DM_real', 'DM_real_and_synth','DM_synth',
    ], help='Name of the training experiment configuration (maps to YAML files).', required=True)

    parser.add_argument('--val_data_config', type=str, default='cfg/val/real.yaml',
                        help='Path to YAML config file containing dataset_kwargs_list, batch_size_list, and dataset_names_list')
    parser.add_argument('--save_name', type=str, default='runs/default',
                        help='Base directory for saving the model and training results')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint path for initializing model with preloaded weights. Default: None')
    parser.add_argument('--training_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer to use (e.g., AdamW)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--val_every_n_step', type=int, default=100,
                        help='Frequency (in steps) to run validation')
    parser.add_argument('--positive_iou_threshold', type=float, default=None,
                        help='Threshold for defining a positive ROI for the Faster R-CNN RPN')
    return parser.parse_args()

EXPERIMENT_CONFIG_MAP = {
        'real':                              ('cfg/train/real.yaml',                       'cfg/val/real.yaml'),
        'real_and_synth':                    ('cfg/train/real_and_synth.yaml',             'cfg/val/real.yaml'),
        'synth':                             ('cfg/train/synth.yaml',                      'cfg/val/synth.yaml'),
        'real100':                           ('cfg/train/real100.yaml',                    'cfg/val/real.yaml'),
        'real80':                            ('cfg/train/real80.yaml',                     'cfg/val/real.yaml'),
        'real60':                            ('cfg/train/real60.yaml',                     'cfg/val/real.yaml'),
        'real40':                            ('cfg/train/real40.yaml',                     'cfg/val/real.yaml'),
        'real20':                            ('cfg/train/real20.yaml',                     'cfg/val/real.yaml'),
        'real80_and_synth20':                ('cfg/train/real80_and_synth20.yaml',         'cfg/val/real.yaml'),
        'real60_and_synth40':                ('cfg/train/real60_and_synth40.yaml',         'cfg/val/real.yaml'),
        'real40_and_synth60':                ('cfg/train/real40_and_synth60.yaml',         'cfg/val/real.yaml'),
        'real20_and_synth80':                ('cfg/train/real20_and_synth80.yaml',         'cfg/val/real.yaml'),
        'real100_and_synth20':               ('cfg/train/real100_and_synth20.yaml',        'cfg/val/real.yaml'),
        'real100_and_synth40':               ('cfg/train/real100_and_synth40.yaml',        'cfg/val/real.yaml'),
        'real100_and_synth60':               ('cfg/train/real100_and_synth60.yaml',        'cfg/val/real.yaml'),
        'real100_and_synth80':               ('cfg/train/real100_and_synth80.yaml',        'cfg/val/real.yaml'),
        'real100_and_synth100':              ('cfg/train/real100_and_synth100.yaml',       'cfg/val/real.yaml'),
        'diffusion_simple':                  ('cfg/train/genAI/diffusion_simple.yaml',     'cfg/val/real.yaml'),
        'diffusion_finetuned':               ('cfg/train/genAI/diffusion_finetuned.yaml',  'cfg/val/real.yaml'),
        'diffusion_exp_tsynth':              ('cfg/train/genAI/tsynth.yaml',               'cfg/val/real.yaml'),
        'diffusion_exp_real_baseline':       ('cfg/train/genAI/real_baseline_1.yaml',      'cfg/val/real.yaml'),
        'diffusion_exp_real_baseline_subset':('cfg/train/genAI/real_baseline_2.yaml',      'cfg/val/real.yaml'),
        'DM_real':                           ('cfg/DM/train/real.yaml',                    'cfg/DM/val/real.yaml'),
        'DM_real_and_synth':                 ('cfg/DM/train/real_and_synth.yaml',          'cfg/DM/val/real.yaml'),
        'DM_synth':                          ('cfg/DM/train/synth.yaml',                   'cfg/DM/val/synth.yaml'),
    }


def custom_loss(model, images, targets, step=None):
    # Forward pass to get the original loss from the model
    losses = 0
    loss_dict = model(images, targets)
    for key, value in loss_dict.items():
        losses += value
        writer.add_scalar(key, value, global_step=step)
    total_loss = losses
    return total_loss


@torch.no_grad()
def validate_model(model, data_loader, device, iou_threshold=0.5, confidence_threshold=0.5):
    """
    Validation process. This function computes the sensitivity, false positives, and false negatives.
    """
    model.eval()
    correct = 0
    total_gt_boxes = 0
    false_positives = 0
    false_negatives = 0

    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for i in range(len(outputs)):
            pred_boxes = outputs[i]['boxes']
            pred_scores = outputs[i]['scores']
            true_boxes = targets[i]['boxes'].to(device)

            total_gt_boxes += len(true_boxes)
            # Filter boxes by confidence threshold
            confident_indices = torch.where(pred_scores > confidence_threshold)[0]
            pred_boxes = pred_boxes[confident_indices]

            matched_gt = [False] * len(true_boxes)

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, true_box in enumerate(true_boxes):
                    iou = calculate_iou(true_box.cpu().numpy(), pred_box.cpu().numpy())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    if not matched_gt[best_gt_idx]:
                        correct += 1
                        matched_gt[best_gt_idx] = True
                    else:
                        false_positives += 1
                else:
                    false_positives += 1

            false_negatives += matched_gt.count(False)

    sensitivity = correct / total_gt_boxes if total_gt_boxes > 0 else 0
    return sensitivity, false_positives, false_negatives


@torch.no_grad()
def validate_model2(model, data_loader, device, fp_min, fp_max, criteria=dbtex_criteria):
    """
    Validates model by returning the average sensitivity of the FROC curve between fp_min and fp_max.
    """
    df = infer_dataset(model, data_loader, eval_volume=False, nonmax_suppression_threshold=0.25, device=device)
    df['correct_boxes'] = df.apply(lambda row: calc_correct_boxes(row['true_boxes'],
                                                                  row['pred_boxes'],
                                                                  criteria=criteria,
                                                                  threshold=0.5),
                                                                  axis=1)
    tpr, avg_fp = calc_froc(df['correct_boxes'], df['scores'])
    avg_sensitivity = calc_avg_value(avg_fp, tpr, fp_min, fp_max)
    if np.isnan(avg_sensitivity):
        closest_index = np.argmin(np.abs(avg_fp - fp_min))
        avg_sensitivity = tpr[closest_index]
    return avg_sensitivity


def cycle_loader(dataloader_iter, dataloader):
    """
    Iterates over an iterable dataloader continuously. Reinitializes the iterator once exhausted.
    """
    try:
        outputs = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        outputs = next(dataloader_iter)
    return outputs, dataloader_iter


def train_model(model, optimizer, dataloader_list, device, training_steps, val_hook=None):
    model.train()
    dataloader_iter_list = [iter(d) for d in dataloader_list]
    for step in range(1, training_steps+1):
        optimizer.zero_grad()
        image_batch, target_batch = [], []
        for l in range(len(dataloader_list)):
            (images, targets), dataloader_iter_list[l] = cycle_loader(dataloader_iter_list[l], dataloader_list[l])
            image_batch.extend([image.to(device) for image in images])
            target_batch.extend([{k: v.to(device) for k, v in t.items()} for t in targets])
        loss = custom_loss(model, image_batch, target_batch, step=step)
        loss.backward()
        total_loss = loss.item()
        optimizer.step()
        print(f'Step {step} / {training_steps}.  Loss: {total_loss}')
        sys.stdout.flush()

        if val_hook is not None:
            val_hook(model, step)
            model.train()
    return None

def main():
    args = parse_args()
    train_config_path, val_config_path = EXPERIMENT_CONFIG_MAP[args.experiment]

    # Load train dataset configuration from YAML file.
    with open(train_config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_kwargs_list = cfg.get('dataset_kwargs_list', [])
    batch_size_list = cfg.get('batch_size_list', [])
    dataset_name_list = cfg.get('dataset_names_list', [])

    # Check that all dataset lists have the same length.
    if not (len(dataset_kwargs_list) == len(batch_size_list) == len(dataset_name_list)):
        raise ValueError("The lengths of dataset_kwargs_list, batch_size_list, and dataset_names_list must be equal.")
    
    # Load val dataset configuration from YAML file.
    with open(val_config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    val_dataset_kwargs = cfg.get('dataset_kwargs', [])
    val_batch_size = cfg.get('batch_size', [])
    val_dataset_name = cfg.get('dataset_name', [])

    # Training Parameters from command line.
    checkpoint_path = args.checkpoint_path
    training_steps = args.training_steps
    optimizer_name = args.optimizer
    optimizer_kwargs = {'lr': args.lr}
    val_every_n_step = args.val_every_n_step
    positive_iou_threshold = args.positive_iou_threshold
    save_name = args.save_name

    os.makedirs(f'{save_name}/', exist_ok=True)
        # Save hyperparameters to file.
    hyperparams = {
        'training_steps': training_steps,
        'optimizer': optimizer_name,
        'optimizer_kwargs': optimizer_kwargs,
        'checkpoint_path': checkpoint_path,
        'dataset_names_list': dataset_name_list,
        'dataset_kwargs_list': dataset_kwargs_list,
        'batch_size_list': batch_size_list,
        'val_dataset_name': val_dataset_name,
        'val_dataset_kwargs': val_dataset_kwargs,
        'val_batch_size': val_batch_size,
        'val_every_n_step': val_every_n_step
    }
    with open(f'{save_name}/hyperparams.yaml', 'w') as f:
        yaml.safe_dump(hyperparams, f)

    # Define training datasets and dataloaders.
    trainset_list = [getattr(custom_datasets, dname)(**dkwargs)
                     for (dname, dkwargs) in zip(dataset_name_list, dataset_kwargs_list)]

    total_images = 0
    for dname, ds in zip(dataset_name_list, trainset_list):
        num_samples = len(ds)
        print(f"Loaded training dataset '{dname}' with {num_samples} images")
        total_images += num_samples
        
    print(f"Total number of training images: {total_images}")

    trainloader_list = [DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
                        for (ds, bs) in zip(trainset_list, batch_size_list) if bs > 0]

    val_dataset = getattr(custom_datasets, val_dataset_name)(**val_dataset_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))

    # Load detection model.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_fasterrcnn_model(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1',
                                 checkpoint_path=checkpoint_path,
                                 positive_iou_threshold=positive_iou_threshold)
    model.to(device)

    global writer
    writer = SummaryWriter(log_dir=f'{save_name}/')
    best_val = 0

    def validate_model_hook(model, current_step):
        nonlocal best_val
        if current_step % val_every_n_step == 0:
            print("Validating")
            val_sensitivity, _, _ = validate_model(model, val_loader, device)
            writer.add_scalar("Sensitivity/val", val_sensitivity, global_step=current_step)
            if val_sensitivity >= best_val:
                torch.save(model.state_dict(), f'{save_name}/best_ckpt.pth')
                with open(f'{save_name}/ckpt.txt', 'a') as file:
                    file.write(f'best_ckpt.pth saved at step={current_step}, val_sensitivity={val_sensitivity}\n')
                best_val = val_sensitivity
   
    # Define the optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, optimizer_name)(params, **optimizer_kwargs)

    print("##############   TRAINING   ##############")
    train_model(model, optimizer, trainloader_list, device, training_steps, val_hook=validate_model_hook)
    
    torch.save(model.state_dict(), f'{save_name}/last_ckpt.pth')
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The .* Bits Stored value '10' .*")
    main()
