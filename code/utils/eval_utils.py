import numpy as np
import ast
import re

def calculate_iou(box1, box2):
    """Calculates the intersection over union (IoU) between 2 bounding boxes each
    in the form np.ndarray [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union


def calc_avg_value(x, y, xmin=None, xmax=None):
    """
    Calculate the average value of y over the interval [xmin, xmax] 
    using trapezoidal integration.

    Parameters:
    ----------
    x : np.ndarray
        1D array of x values (independent variable).
    y : np.ndarray
        1D array of y values (dependent variable), same length as x.
    xmin : float, optional
        Lower bound of the interval. If None, defaults to min(x).
    xmax : float, optional
        Upper bound of the interval. If None, defaults to max(x).

    Returns:
    -------
    float
        The average value of y over the specified interval.
    """
    # Sort x and y together if needed
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Set default bounds if not provided
    xmin = x[0] if xmin is None else xmin
    xmax = x[-1] if xmax is None else xmax
 
    # Filter values within xmin and xmax
    mask = (x_sorted >= xmin) & (x_sorted <= xmax)
    x_filtered = x_sorted[mask]
    y_filtered = y_sorted[mask]

    # Compute weighted average using trapezoidal integration
    if len(x_filtered) > 1:
        y_avg = np.trapz(y_filtered, x_filtered) / (x_filtered[-1] - x_filtered[0])
    elif len(y_filtered) == 1:
        y_avg = y_filtered[0]  # Single point case
    else:
        y_avg = np.nan  # Return NaN if no points in range
    return y_avg



def dbtex_criteria(gt_box, predicted_box):
    """Returns whether or not predicted_box successfully detects the target gt_box as defined by criteria in the DBTex challenge.
    Criteria can be found in eAppendix 2 section B.4.1 of A Competition, Benchmark, Code, and Data for Using 
    Artificial Intelligence to Detect Lesions in Digital Breast Tomosynthesis.
    gt_box and predicted_box: np.ndarrays of format [x1, y1, x2, y2]
    returns: bool"""
    box_diff = (gt_box - predicted_box) / 2
    center_distance_squared = (box_diff[2] + box_diff[0])**2 + (box_diff[3] + box_diff[1])**2
    diagonal_squared = (predicted_box[2] - predicted_box[0])**2 + (predicted_box[3] - predicted_box[1])**2
    is_correct = (center_distance_squared < diagonal_squared/4) or (center_distance_squared < 10000)
    return is_correct


def calc_froc(correct_boxes, scores, p=50, use_precision=False):
    """
    Calculate the True Positive Rate (TPR) and average False Positives (FP)
    over some decision threshold tau that ranges from 0-1.
    which are typically plotted together on a free response receiver operator
    characteristic (FROC) curve.

    Returns:
    -------
    tpr : numpy.ndarray
        An array of True Positive Rates at different score thresholds. The 
        length of the array corresponds to the parameter `p`.

    avg_fp : numpy.ndarray
        An array representing the average number of False Positives across all 
        samples at each score threshold. The length of the array corresponds 
        to the parameter `p`.

    if use_precision is True, then precision is returned in place of avg_fp.   
    """
    total_positives = [len(c) for c in correct_boxes]
    tau = np.linspace(0, 1, p, endpoint=True)
    true_positives = np.zeros(p)
    false_positives = np.zeros(p)
    for correct_box, score, total_positive in zip(correct_boxes, scores, total_positives):
        already_counted = np.zeros(total_positive, dtype=bool)
        for idx in np.argsort(score)[::-1]:
            s = score[idx]
            if total_positive > 0:
                detected = correct_box[:, idx]
                true_positives[np.where(s > tau)] += np.sum(detected & ~already_counted)
                already_counted |= detected

                if not np.any(detected):
                    false_positives[np.where(s > tau)] += 1
            else:
                false_positives[np.where(s > tau)] += 1
    
    tpr = true_positives / sum(total_positives)
    avg_fp = false_positives / len(correct_boxes)
    if use_precision:
        total_predictions = true_positives + false_positives
        mask = total_predictions != 0
        precision = true_positives[mask] / total_predictions[mask]
        return tpr[mask], precision
    else:
        return tpr, avg_fp


def calc_correct_boxes(gt_boxes, pred_boxes, criteria=calculate_iou, threshold=None):
    """
    Given a list of ground truth bboxes and predicted bboxes (each entry [x1, y1, x2, y2])
    outputs an 2D array where entry i,j is 1 if pred_box[j] predicts gt_box[i] according to
    the criteria (function).
    """
    output = np.zeros([len(gt_boxes), len(pred_boxes)], dtype='bool')
    for i in range(len(gt_boxes)):
        for j in range(len(pred_boxes)):
            output[i,j] = criteria(gt_boxes[i], pred_boxes[j])
    
    if threshold is not None:
        output = output > threshold

    return output


def format_arrays(arrays_string):
    """
    Formats a string representation of numerical arrays into a NumPy array.
    The function processes the input string by:
    - Replacing newlines and formatting decimal numbers correctly.
    - Adding commas to separate elements.
    - Converting the processed string to a NumPy array.

    Example:
    --------
    Input:  "[1.23 4.56\n7.89 0.12]"
    Output: array([[1.23, 4.56], [7.89, 0.12]])
    """
    s = arrays_string.replace('\n', ', ')
    s = s.replace('. ', '., ')
    s = re.sub(r'(\d\.\d+)(?=\s|$)', r'\1,', s)
    s = re.sub(r'(\d[.\deE+-]+)(?=\s|\])', r'\1,', s)
    s = np.array(ast.literal_eval(s))
    return s