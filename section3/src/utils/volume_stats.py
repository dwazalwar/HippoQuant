"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    # <YOUR CODE HERE>

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    common_elements = np.sum(a_bin * b_bin)
    dice_score = 2 * common_elements / (np.sum(a_bin)+ np.sum(b_bin))

    return round(dice_score, 2)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # <YOUR CODE GOES HERE>

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()

    if union == 0:
        return 1.0  # Both are empty, so perfect match

    jaccard_score = intersection / union
    
    return round(jaccard_score, 2)

def sensitivity_specificity(a, b):
    """
    Compute Sensitivity (Recall) and Specificity for two 3-dimensional volumes.
    Volumes are expected to be of the same size, with binary masks (0 = background, 1 = data).

    Arguments:
        a {Numpy array} -- 3D array representing the predicted labels
        b {Numpy array} -- 3D array representing the ground truth labels

    Returns:
        tuple(float, float) -- Sensitivity and Specificity scores
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3D inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    tp = np.sum((a == 1) & (b == 1))  # True Positives
    fn = np.sum((a == 0) & (b == 1))  # False Negatives
    tn = np.sum((a == 0) & (b == 0))  # True Negatives
    fp = np.sum((a == 1) & (b == 0))  # False Positives

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity


def dice_per_class(a, b, num_classes, epsilon=1e-6):
    """
    Compute Dice coefficient per class in a multi-class segmentation task.

    Arguments:
        a {Numpy array} -- 3D array representing the predicted labels
        b {Numpy array} -- 3D array representing the ground truth labels
        num_classes {int} -- Number of segmentation classes
        epsilon (float): Smoothing factor to avoid division by zero

    Returns:
        dict -- Dictionary with Dice scores per class
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3D inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    dice_scores = {}

    for cls in range(num_classes):
        pred_class = (a == cls)  # Binary mask for class
        gt_class = (b == cls)  # Binary mask for class

        intersection = np.sum(pred_class * gt_class)
        union = np.sum(pred_class) + np.sum(gt_class)
        dice_scores[f"Class {cls}"] = round((2. * intersection  + epsilon) / (union + epsilon), 2)

    return dice_scores