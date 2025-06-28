"""
Evaluate Detectors

This script evaluates a single saved detector model on a given evaluation set.
Inference results for each sample are saved as a CSV file.
The evaluation dataset configuration is loaded from a YAML file, while the model checkpoint path and output save name
are provided via command-line arguments.
"""

import argparse
import os
import yaml
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.model_utils import get_fasterrcnn_model, infer_3d_input
from utils.eval_utils import calculate_iou
import custom_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Detector Model")
    parser.add_argument('--config', type=str, default='cfg/test/synth.yaml',
                        help="Path to YAML config file containing eval_dataset_name, eval_dataset_kwargs, batch_size, and extra_columns_keys")
    parser.add_argument('--model_path', type=str, default='runs/default/best.ckpt',
                        help="Path to the model checkpoint to be evaluated")
    parser.add_argument('--save_name', type=str, default='runs/default',
                        help="Path where evaluation results will be saved")
    parser.add_argument('--non_max_suppression_threshold', type=float, default=0.25,
                        help="IoU threshold for non-max suppression. Set to None to disable")
    return parser.parse_args()

def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-max suppression on bounding boxes.
    
    Args:
        boxes (np.array): Array of bounding boxes [xmin, ymin, xmax, ymax]
        scores (np.array): Array of corresponding scores
        threshold (float): IoU threshold for merging boxes
        
    Returns:
        (np.array, np.array): Filtered boxes and scores after non-max suppression
    """
    order = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    boxes, scores = boxes[order], scores[order]
    keep = np.ones(len(scores), dtype=bool)
    for i in range(len(boxes)):
        if keep[i]:
            for j in range(i+1, len(boxes)):
                if keep[j]:
                    iou = calculate_iou(boxes[i], boxes[j])
                    if iou > threshold:
                        keep[j] = False
    return boxes[keep], scores[keep]

@torch.no_grad()
def infer_dataset(model, loader, eval_volume=False, nonmax_threshold=None, device=torch.device('cpu')):
    """
    Run inference on the evaluation dataset.
    
    Args:
        model: Detector model
        loader: DataLoader for evaluation dataset
        eval_volume (bool): Whether to evaluate on 3D volumes
        nonmax_threshold (float): Non-max suppression threshold
        device: Torch device
        
    Returns:
        pd.DataFrame: DataFrame containing inference results
    """
    true_boxes = []
    pred_boxes = []
    true_slices = []
    pred_slices = []
    scores = []
    model = model.to(device)
    model.eval()
    
    for images, targets in loader:
        images = [img.to(device) for img in images]
        if eval_volume:
            output = [infer_3d_input(model, volume, device=device) for volume in images]
            pred_slices.extend([out['slices'].cpu().numpy() for out in output])
            true_slices.extend([t['slices'].cpu().numpy() for t in targets])
        else:
            output = model(images)
            pred_slices.extend([[0] * len(out['boxes']) for out in output])
            true_slices.extend([[[0, 0]] * len(t['boxes']) for t in targets])
        true_boxes.extend([t['boxes'].cpu().numpy() for t in targets])
        pred_boxes.extend([out['boxes'].cpu().numpy() for out in output])
        scores.extend([out['scores'].cpu().numpy() for out in output])
    
    # Apply non-max suppression if enabled
    if nonmax_threshold is not None:
        for i in range(len(pred_boxes)):
            pred_boxes[i], scores[i] = non_max_suppression(pred_boxes[i], scores[i], nonmax_threshold)
    
    data = {
        'true_boxes': true_boxes,
        'pred_boxes': pred_boxes,
        'true_slices': true_slices,
        'pred_slices': pred_slices,
        'scores': scores
    }
    return pd.DataFrame(data)

def main():
    args = parse_args()
    
    # Load evaluation configuration from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    eval_dataset_name = config.get('dataset_name')
    eval_dataset_kwargs = config.get('dataset_kwargs', {})
    batch_size = config.get('batch_size', 16)
    extra_columns_keys = config.get('extra_columns_keys', [])
    
    # Create evaluation dataset and DataLoader
    eval_dataset = getattr(custom_datasets, eval_dataset_name)(**eval_dataset_kwargs)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        collate_fn=lambda x: tuple(zip(*x)))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_fasterrcnn_model(checkpoint_path=args.model_path)
    eval_volume = False  # Change if you want to evaluate 3D volumes
    results = infer_dataset(model, loader,
                            eval_volume=eval_volume,
                            nonmax_threshold=args.non_max_suppression_threshold,
                            device=device)
    
    # Optionally add extra columns from the evaluation dataset
    if hasattr(eval_dataset, 'label_df') and not eval_dataset.label_df.empty and extra_columns_keys:
        extra_columns = eval_dataset.label_df[extra_columns_keys]
        results = pd.concat([extra_columns, results], axis=1)
    
    # Save results CSV
    os.makedirs(args.save_name, exist_ok=True)
    save_path = os.path.join(args.save_name, f'result.csv')
    print(f"Saving results to {save_path}.")
    results.to_csv(save_path, index=False)
    
    # Save configuration and parameters for record keeping
    hyperparams = {
        'model_path': args.model_path,
        'eval_dataset_name': eval_dataset_name,
        'eval_dataset_kwargs': eval_dataset_kwargs,
        'batch_size': batch_size,
        'extra_columns_keys': extra_columns_keys,
        'non_max_suppression_threshold': args.non_max_suppression_threshold
    }
    with open(os.path.join(args.save_name, 'hyperparams.yaml'), 'w') as f:
        yaml.safe_dump(hyperparams, f)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The .* Bits Stored value '10' .*")
    main()
