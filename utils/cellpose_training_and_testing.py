import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import shutil
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from cellpose import models
from cellpose.train import train_seg
from cellpose.io import load_train_test_data
from cell_detector import CellPoseDetector


def get_main_dir(main_dir=None):
    """
    Get the main directory path containing training, validation, and test datasets.

    Args:
        main_dir (str, optional): Path to the main directory. If provided, user input is skipped.

    Returns:
        str: Path to the main directory.
    """
    if main_dir is None:
        main_dir = input("Enter the main directory path (containing 'training', 'validation', 'testing'): ").strip()
    if main_dir.startswith('"') and main_dir.endswith('"'):
        main_dir = main_dir[1:-1]

    for subdir in ["training", "validation", "testing"]:
        subdir_path = os.path.join(main_dir, subdir)
        if not os.path.isdir(subdir_path):
            os.makedirs(subdir_path)
    return main_dir


def setup_data_split(main_dir, split_ratio=(70, 15, 15)):
    """
    Split data into training, validation, and testing folders based on the given ratio.

    Args:
        main_dir (str): Path to the main directory containing the 'labeled' subfolder.
        split_ratio (tuple): Ratio for training, validation, and testing. Default is (70, 15, 15).

    Raises:
        ValueError: If the split ratio does not sum to 100.
        FileNotFoundError: If the 'labeled' subfolder is missing.
    """
    if sum(split_ratio) != 100:
        raise ValueError("The split ratio must sum to 100.")

    train_dir = os.path.join(main_dir, "training")
    val_dir = os.path.join(main_dir, "validation")
    test_dir = os.path.join(main_dir, "testing")

    for subdir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    labeled_dir = os.path.join(main_dir, "labeled")
    if not os.path.exists(labeled_dir):
        raise FileNotFoundError(f"'labeled' subfolder not found in: {main_dir}")

    tif_files = [f for f in os.listdir(labeled_dir) if f.endswith(".tif")]
    if not tif_files:
        return

    valid_pairs = []
    for tif_file in tif_files:
        basename = os.path.splitext(tif_file)[0]
        npy_file = basename + "_seg.npy"
        npy_path = os.path.join(labeled_dir, npy_file)
        if os.path.exists(npy_path):
            valid_pairs.append((tif_file, npy_file))

    if not valid_pairs:
        raise ValueError("No valid .tif and _seg.npy pairs found in the 'labeled' subfolder.")

    random.shuffle(valid_pairs)
    total_files = len(valid_pairs)
    train_count = int(total_files * split_ratio[0] / 100)
    val_count = int(total_files * split_ratio[1] / 100)
    test_count = total_files - train_count - val_count

    if train_count + val_count + test_count != total_files:
        remaining_files = total_files - (train_count + val_count + test_count)
        train_count += remaining_files

    for i, (tif_file, npy_file) in enumerate(valid_pairs):
        dest_dir = train_dir if i < train_count else val_dir if i < train_count + val_count else test_dir
        shutil.move(os.path.join(labeled_dir, tif_file), os.path.join(dest_dir, tif_file))
        shutil.move(os.path.join(labeled_dir, npy_file), os.path.join(dest_dir, npy_file))

    print(f"Data split completed: {train_count} training, {val_count} validation, {test_count} testing")


def train_model(main_dir, channels=[0, 0], learning_rate=0.05, weight_decay=0.0001, n_epochs=200,
                batch_size=8, min_train_masks=1, save_every=50, normalize=True, rescale=True):
    """
    Train the Cellpose model using the provided main directory.

    Args:
        main_dir (str): Path to the main directory containing 'training', 'validation', and 'testing' subfolders.
        channels (list): List of channels to use for training. Default is [0, 0] (grayscale).
        learning_rate (float): Learning rate for training. Default is 0.05.
        weight_decay (float): Weight decay for training. Default is 0.0001.
        n_epochs (int): Number of epochs to train. Default is 200.
        batch_size (int): Batch size for training. Default is 8.
        min_train_masks (int): Minimum number of masks required for training. Default is 1.
        save_every (int): Save the model every 'save_every' epochs. Default is 50.
        normalize (bool): Whether to normalize the data. Default is True.
        rescale (bool): Whether to rescale the data. Default is True.

    Returns:
        str: Path to the saved model.
    """
    train_dir = os.path.join(main_dir, "training")
    val_dir = os.path.join(main_dir, "validation")
    model_dir = os.path.join(main_dir, "models")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_images, train_labels, train_paths, val_images, val_labels, val_paths = load_train_test_data(
        train_dir=train_dir, test_dir=val_dir, mask_filter="_seg.npy"
    )

    model = models.CellposeModel(gpu=True, model_type="cyto3")
    model_path = train_seg(
        net=model.net,
        train_data=train_images,
        train_files=train_paths,
        train_labels=train_labels,
        test_data=val_images,
        test_files=val_paths,
        test_labels=val_labels,
        channels=channels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
        min_train_masks=min_train_masks,
        save_path=model_dir,
        save_every=save_every,
        normalize=normalize,
        rescale=rescale,
    )
    return str(model_path[0])


def test_detection(main_dir, model_path=None, diameter=10, flow_threshold=0.4,
                   min_canvas_size: Optional[Tuple[int, int]] = None):
    """
    Test the trained model on the testing dataset.

    Args:
        main_dir (str): Path to the main directory containing 'training', 'validation', and 'testing' subfolders.
        model_path (str): Path to the trained model.
        diameter (int): Diameter parameter for detection. Default is 10.
        flow_threshold (float): Flow threshold for detection. Default is 0.4.
        min_canvas_size (int, optional): Minimum canvas size for detection.

    Raises:
        FileNotFoundError: If the model or test directory is missing.
    """
    test_dir = os.path.join(main_dir, "testing")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".tif")]
    for test_file in test_files:
        with tifffile.TiffFile(test_file) as tif:
            image = tif.asarray()
            detector = CellPoseDetector(image, model=model_path, flow_threshold=flow_threshold,
                                        diameter=diameter, min_canvas_size=min_canvas_size)
            masks, *_ = detector.detect()
            basename = os.path.splitext(os.path.basename(test_file))[0]
            output_path = os.path.join(test_dir, f"{basename}_segrun.tif")
            tifffile.imwrite(output_path, masks)


def compare_and_visualize(main_dir, overlap_threshold=0.5):
    """
    Compare detected masks with ground truth masks and generate a visual representation.

    Args:
        main_dir (str): Path to the main directory containing 'training', 'validation', and 'testing' subfolders.
        overlap_threshold (float): Minimum IoU threshold for matching. Default is 0.5.
    """
    test_dir = os.path.join(main_dir, "testing")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found at: {test_dir}")

    tiff_files = [f for f in os.listdir(test_dir) if f.endswith(".tif") and not f.endswith("_segrun.tif")]
    metrics = []

    for tiff_file in tiff_files:
        image_path = os.path.join(test_dir, tiff_file)
        image = tifffile.imread(image_path)
        basename = os.path.splitext(tiff_file)[0]
        detected_mask_path = os.path.join(test_dir, f"{basename}_segrun.tif")
        npy_path = os.path.join(test_dir, f"{basename}_seg.npy")

        if not os.path.exists(detected_mask_path) or not os.path.exists(npy_path):
            continue

        detected_masks = tifffile.imread(detected_mask_path)
        mask_data = np.load(npy_path, allow_pickle=True).item()
        ground_truth_masks = mask_data.get('masks', None)

        if ground_truth_masks is None:
            continue

        matched_pairs, overlap_image, highlighted_image = match_masks(detected_masks, ground_truth_masks, match_threshold=overlap_threshold)
        metrics.extend(matched_pairs)

        output_path = os.path.join(test_dir, f"{basename}_comparison.tif")
        output_path_fn = os.path.join(test_dir, f"{basename}_fn.tif")
        tifffile.imwrite(output_path, overlap_image.astype(np.uint8), photometric='rgb')
        tifffile.imwrite(output_path_fn, highlighted_image.astype(np.uint8), photometric='rgb')

    if metrics:
        plot_overlap_statistics(metrics, main_dir, threshold=overlap_threshold)


def plot_overlap_statistics(matched_pairs, main_dir, threshold=0.0):
    """
    Generate and save statistics plots for the matched pairs.

    Args:
        matched_pairs (list): List of tuples (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou).
        main_dir (str): Main directory to save results and plots.
        threshold (float): Minimum threshold for metrics to be included in the plots.
    """
    plots_dir = os.path.join(main_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    recall_overlap_values = [pair[2] for pair in matched_pairs if pair[2] is not None and pair[2] >= threshold]
    f1_values = [pair[3] for pair in matched_pairs if pair[3] is not None and pair[3] >= threshold]
    boundary_iou_values = [pair[4] for pair in matched_pairs if pair[4] is not None and pair[4] >= threshold]
    fpr_values = [pair[5] for pair in matched_pairs if pair[5] is not None]
    fnr_values = [pair[6] for pair in matched_pairs if pair[6] is not None]
    asymmetry_values = [pair[7] for pair in matched_pairs if pair[7] is not None]
    iou_values = [pair[8] for pair in matched_pairs if pair[8] is not None and pair[8] >= threshold]

    def safe_stats(values):
        return np.mean(values) if values else np.nan, np.median(values) if values else np.nan, np.std(values) if values else np.nan

    matches = sum(1 for pair in matched_pairs if pair[0] is not None and pair[1] is not None)
    false_positives = sum(1 for pair in matched_pairs if pair[0] is None and pair[1] is not None)
    false_negatives = sum(1 for pair in matched_pairs if pair[0] is not None and pair[1] is None)

    metrics = {
        "IoU": iou_values,
        "Recall Overlap": recall_overlap_values,
        "F1 Score": f1_values,
        "Boundary IoU": boundary_iou_values,
        "False Positive Rate": fpr_values,
        "False Negative Rate": fnr_values,
        "Asymmetry Score": asymmetry_values,
    }

    metrics_file_path = os.path.join(main_dir, 'metrics_overview.txt')
    with open(metrics_file_path, 'w', encoding='utf-8') as metrics_file:
        for name, values in metrics.items():
            mean, median, std = safe_stats(values)
            log_line = f"{name}: Mean = {mean:.2f}, Median = {median:.2f}, Std Dev = {std:.2f}\n"
            print(f"📊 {name}: Mean = {mean:.2f}, Median = {median:.2f}, Std Dev = {std:.2f}")
            metrics_file.write(log_line)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Mask Matching Statistics', fontsize=18)

    metric_list = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'brown', 'pink']

    for i, (metric, values) in enumerate(zip(metric_list, metric_values)):
        if values:
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)

            ax = axes[i // 3, i % 3]
            ax.hist(values, bins=20, color=colors[i], edgecolor='black', alpha=0.7, label=metric)
            ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
            # ax.text(mean_val, ax.get_ylim()[1] * 0.7, f'σ={std_val:.2f}', color='blue', fontsize=10)
            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.legend(title=f"Std Dev: {std_val:.2f}")
            ax.text(0.95, 0.95, f'n = {len(values)}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    horizontalalignment='right', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    categories = ['Matches', 'False Positives', 'False Negatives']
    counts = [matches, false_positives, false_negatives]
    axes[2, 2].bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', alpha=0.7)
    for i, count in enumerate(counts):
        axes[2, 2].text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')
    axes[2, 2].set_title('Match/False Positive/False Negative Counts')
    axes[2, 2].set_ylabel('Count')
    axes[2, 2].text(0.95, 0.95, f'n = {len(matched_pairs)}', transform=axes[2, 2].transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right', color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    used_axes = len(metrics)
    for i in range(used_axes, 8):
        fig.delaxes(axes[i // 3, i % 3])

    full_fig_path = os.path.join(plots_dir, 'overlap_statistics_full.png')
    plt.savefig(full_fig_path, bbox_inches='tight')
    plt.close(fig)

    for metric, values in metrics.items():
        if values:
            fig, ax = plt.subplots(figsize=(8, 6))
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            ax.hist(values, bins=20, color=colors[metric_list.index(metric)], edgecolor='black', alpha=0.7, label=metric)
            ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
            # ax.text(mean_val, ax.get_ylim()[1] * 0.7, f'σ={std_val:.2f}', color='blue', fontsize=10)
            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.legend(title=f"Std Dev: {std_val:.2f}")
            ax.text(0.95, 0.95, f'n = {len(values)}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    horizontalalignment='right', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
            plot_path = os.path.join(plots_dir, f'{metric.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', alpha=0.7)
    for i, count in enumerate(counts):
        ax.text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')
    ax.set_title('Match/False Positive/False Negative Counts')
    ax.set_ylabel('Count')
    ax.text(0.95, 0.95, f'n = {len(matched_pairs)}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
    counts_path = os.path.join(plots_dir, 'match_false_positive_false_negative_counts.png')
    plt.savefig(counts_path, bbox_inches='tight')
    plt.close(fig)

    csv_path = os.path.join(main_dir, 'matched_pairs.csv')
    df = pd.DataFrame(matched_pairs, columns=['gt_id', 'detected_id', 'overlap', 'f1', 'boundary_iou', 'fpr', 'fnr', 'asymmetry_score', 'iou'])
    df.to_csv(csv_path, index=False)


def calculate_f1(gt_mask, detected_mask):
    """
    Calculate F1 Score (Dice Coefficient) for a pair of masks.

    Args:
        gt_mask (np.array): Ground truth mask.
        detected_mask (np.array): Detected mask.

    Returns:
        float: F1 Score.
    """
    intersection = np.logical_and(gt_mask, detected_mask).sum()
    return (2 * intersection) / (gt_mask.sum() + detected_mask.sum()) if (gt_mask.sum() + detected_mask.sum()) > 0 else 0


def calculate_boundary_iou(gt_mask, detected_mask):
    """
    Calculate Boundary IoU for a pair of masks.

    Args:
        gt_mask (np.array): Ground truth mask.
        detected_mask (np.array): Detected mask.

    Returns:
        float: Boundary IoU.
    """
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours, _ = cv2.findContours(detected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gt_boundary = np.zeros_like(gt_mask, dtype=np.uint8)
    detected_boundary = np.zeros_like(detected_mask, dtype=np.uint8)

    cv2.drawContours(gt_boundary, gt_contours, -1, 1, thickness=1)
    cv2.drawContours(detected_boundary, detected_contours, -1, 1, thickness=1)

    boundary_intersection = np.logical_and(gt_boundary, detected_boundary).sum()
    boundary_union = np.logical_or(gt_boundary, detected_boundary).sum()
    return boundary_intersection / boundary_union if boundary_union > 0 else 0


def calculate_fpr(gt_mask, detected_mask):
    """
    Calculate the False Positive Rate (FPR) for a pair of masks.

    Args:
        gt_mask (np.array): Ground truth mask.
        detected_mask (np.array): Detected mask.

    Returns:
        float: False Positive Rate.
    """
    false_positives = np.sum((detected_mask == 1) & (gt_mask == 0))
    true_negatives = np.sum((detected_mask == 0) & (gt_mask == 0))
    return false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0


def calculate_fnr(gt_mask, detected_mask):
    """
    Calculate the False Negative Rate (FNR) for a pair of masks.

    Args:
        gt_mask (np.array): Ground truth mask.
        detected_mask (np.array): Detected mask.

    Returns:
        float: False Negative Rate.
    """
    false_negatives = np.sum((detected_mask == 0) & (gt_mask == 1))
    true_positives = np.sum((detected_mask == 1) & (gt_mask == 1))
    return false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0


def calculate_asymmetry(overlap, f1):
    """
    Calculate the Asymmetry Score for a pair of masks.

    Args:
        overlap (float): Overlap score (IoU).
        f1 (float): F1 score.

    Returns:
        float: Asymmetry Score.
    """
    return abs(overlap - f1)


def generate_highlighted_visualization(unique_matched_pairs, ground_truth_masks, detected_masks):
    """
    Generate an RGB visualization highlighting false negatives.

    Args:
        unique_matched_pairs (list of tuples): List of match tuples (gt_id, detected_id, ...).
        ground_truth_masks (numpy array): Array of ground truth mask IDs.
        detected_masks (numpy array): Array of detected mask IDs.

    Returns:
        numpy array: RGB visualization highlighting false negatives.
    """
    highlight_vis = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)
    for pair in unique_matched_pairs:
        gt_id, detected_id = pair[:2]
        if gt_id is not None and detected_id is None:
            highlight_vis[ground_truth_masks == gt_id] = [255, 0, 0]
    for gt_id, detected_id, *_ in unique_matched_pairs:
        if gt_id is not None and detected_id is not None:
            highlight_vis[(ground_truth_masks == gt_id) | (detected_masks == detected_id)] = [50, 50, 50]
    return highlight_vis


def match_masks(detected_masks, ground_truth_masks, match_threshold=0.2):
    """
    Match detected masks to ground truth masks using one-sided overlap for evaluation.

    Args:
        detected_masks (np.array): Detected masks array.
        ground_truth_masks (np.array): Ground truth masks array.
        match_threshold (float): Minimum IoU threshold for matching.

    Returns:
        list: List of tuples (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou).
        np.array: Color-coded visualization of matches.
    """
    gt_masks = {gt_id: (ground_truth_masks == gt_id).astype(bool) for gt_id in np.unique(ground_truth_masks) if gt_id != 0}
    detected_masks_dict = {detected_id: (detected_masks == detected_id).astype(bool) for detected_id in np.unique(detected_masks) if detected_id != 0}
    matched_pairs = []

    for gt_id, gt_mask in gt_masks.items():
        overlapping_detected_ids = np.unique(detected_masks[gt_mask])
        overlapping_detected_ids = overlapping_detected_ids[overlapping_detected_ids != 0]

        for detected_id in overlapping_detected_ids:
            detected_mask = detected_masks_dict[detected_id]
            iou = np.logical_and(gt_mask, detected_mask).sum() / np.logical_or(gt_mask, detected_mask).sum()
            if iou >= match_threshold:
                overlap = np.logical_and(gt_mask, detected_mask).sum() / detected_mask.sum()
                f1 = calculate_f1(gt_mask, detected_mask)
                boundary_iou = calculate_boundary_iou(gt_mask, detected_mask)
                fpr = calculate_fpr(gt_mask, detected_mask)
                fnr = calculate_fnr(gt_mask, detected_mask)
                asymmetry_score = calculate_asymmetry(overlap, f1)
                matched_pairs.append((gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou))

    for detected_id, detected_mask in detected_masks_dict.items():
        overlapping_gt_ids = np.unique(ground_truth_masks[detected_mask])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids != 0]

        for gt_id in overlapping_gt_ids:
            gt_mask = gt_masks[gt_id]
            iou = np.logical_and(gt_mask, detected_mask).sum() / np.logical_or(gt_mask, detected_mask).sum()
            if iou >= match_threshold:
                overlap = np.logical_and(gt_mask, detected_mask).sum() / gt_mask.sum()
                f1 = calculate_f1(gt_mask, detected_mask)
                boundary_iou = calculate_boundary_iou(gt_mask, detected_mask)
                fpr = calculate_fpr(gt_mask, detected_mask)
                fnr = calculate_fnr(gt_mask, detected_mask)
                asymmetry_score = calculate_asymmetry(overlap, f1)
                matched_pairs.append((gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou))

    unique_matched_pairs = []
    seen_pairs = {}
    for gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou in matched_pairs:
        pair_key = (gt_id, detected_id)
        if pair_key not in seen_pairs or overlap > seen_pairs[pair_key][2]:
            seen_pairs[pair_key] = (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou)
    unique_matched_pairs = list(seen_pairs.values())

    seen_gt_ids = {pair[0] for pair in unique_matched_pairs if pair[0] is not None}
    for gt_id in gt_masks:
        if gt_id not in seen_gt_ids:
            unique_matched_pairs.append((gt_id, None, 0, 0, 0, 0, 0, 0, 0))

    seen_detected_ids = {pair[1] for pair in unique_matched_pairs if pair[1] is not None}
    for detected_id in detected_masks_dict:
        if detected_id not in seen_detected_ids:
            unique_matched_pairs.append((None, detected_id, 0, 0, 0, 0, 0, 0, 0))

    color_coded = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)
    for gt_id in gt_masks:
        color_coded[ground_truth_masks == gt_id, 1] = 255
    for detected_id in detected_masks_dict:
        color_coded[detected_masks == detected_id, 2] = 255

    highlighted_visualization = generate_highlighted_visualization(unique_matched_pairs, ground_truth_masks, detected_masks)
    return unique_matched_pairs, color_coded, highlighted_visualization


def create_visual_representation(image, ground_truth_masks, detected_masks, matched_pairs):
    """
    Create a visual representation of ground truth and detected masks using OpenCV.

    Args:
        image (np.array): Original image.
        ground_truth_masks (np.array): Ground truth masks.
        detected_masks (np.array): Detected masks.
        matched_pairs (list): List of matched pairs (gt_id, detected_id, overlap).

    Returns:
        np.array: Visual representation as an RGB image.
    """
    if image.ndim == 2:
        visual_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        visual_image = image.copy()

    def draw_mask_contours(mask, color):
        for mask_id in np.unique(mask):
            if mask_id == 0:
                continue
            binary_mask = (mask == mask_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visual_image, contours, -1, color, 2)

    draw_mask_contours(ground_truth_masks, (0, 255, 0))
    draw_mask_contours(detected_masks, (255, 0, 0))

    def get_center(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return (x + w // 2, y + h // 2)
        return None

    for gt_id, detected_id, overlap, _ , _ in matched_pairs:
        gt_mask = (ground_truth_masks == gt_id).astype(np.uint8)
        detected_mask = (detected_masks == detected_id).astype(np.uint8)
        gt_center = get_center(gt_mask)
        detected_center = get_center(detected_mask)

        if gt_center and detected_center:
            text_x = (gt_center[0] + detected_center[0]) // 2
            text_y = (gt_center[1] + detected_center[1]) // 2
        elif gt_center:
            text_x, text_y = gt_center
        elif detected_center:
            text_x, text_y = detected_center
        else:
            continue

        overlap_text = f"{overlap:.2f}"
        cv2.putText(visual_image, overlap_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return visual_image


if __name__ == "__main__":
    main_dir = get_main_dir()
    #setup_data_split(main_dir, split_ratio=(60, 20, 20))
    #model_path = train_model(main_dir)
    #test_detection(main_dir, model_path=model_path, min_canvas_size=(2000, 2000), diameter=750)
    compare_and_visualize(main_dir, overlap_threshold=0.02)
    print("Cellpose training and testing completed!")