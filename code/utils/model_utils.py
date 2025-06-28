import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_model(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1', checkpoint_path=None, positive_iou_threshold=None):
    # Load a pre-trained model for classification and return it as a Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=weights)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (note: +1 because of the background class)
    num_classes = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    if positive_iou_threshold is not None:# Modify the RPN configuration to decrease the IoU threshold for positive regions
        model.rpn.head.rpn_fg_iou_thresh = positive_iou_threshold
    return model


def infer_3d_input(model, volume, postprocess='max_slice', device='cpu'):
    """Runs inference on a 3D DBT volume by evaluating each slice with model
     in volume individually and aggregating the results
     Inputs: model: torch.nn.Module: 2D detection model used for evaluation
             volume: torch.nn.Tensor: single volume of dim [slices, 1, height, width]
     Outputs: outputs: dict with keys [boxes]: list of bboxes [x1, y1, x2, y2]
                                               scores: list of scores from [0, 1]
                                               slice: list of slice indices for each prediction"""
    images = [image for image in volume]
    outputs = model(images)

    boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
    scores = torch.zeros(0, dtype=torch.int64, device=device)
    slices = torch.zeros(0, dtype=torch.int64, device=device)

    for slice_idx, output in enumerate(outputs):
        boxes = torch.cat([boxes, output['boxes']], dim=0)
        scores = torch.cat([scores, output['scores']], dim=0)
        slices = torch.cat([slices, slice_idx*torch.ones(output['scores'].shape, device=device)], dim=0)

    #  Apply postprocessing method, if specified
    if postprocess == 'max_slice':
        max_slice = slices[torch.argmax(scores)]
        idx = torch.where(slices == max_slice)[0]
        boxes, scores, slices = boxes[idx], scores[idx], slices[idx]

    elif postprocess is not None:
        raise Exception("Error: Post-processing method for infer_3d_input is not recognized")

    return {'boxes': boxes, 'scores': scores, 'slices': slices}
