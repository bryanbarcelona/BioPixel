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
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from math import atan2, degrees, radians, sin, cos
from numba import jit, prange
from cellpose import models
from cellpose.train import train_seg
from cellpose.io import load_train_test_data
from detectors import CellPoseDetector, PodosomeDetector


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


def test_detection(main_dir, model_path=None, diameter=10, flow_threshold=0.4, cellprob_threshold=0.0,
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
                                        cellprob_threshold=cellprob_threshold,
                                        diameter=diameter, min_canvas_size=min_canvas_size)
            masks = detector.detect()
            basename = os.path.splitext(os.path.basename(test_file))[0]
            output_path = os.path.join(test_dir, f"{basename}_segrun.tif")
            tifffile.imwrite(output_path, masks)

def test_detection_podosomes_2(main_dir, model_path=None, diameter=10, flow_threshold=0.4, cellprob_threshold=0.0,
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
    def _equalize_image_clahe(image, clip_limit=20.0, tile_grid_size=(4, 4), normalize_output=False):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image, expected to be in grayscale or BGR format.
        clip_limit : float, optional
            Threshold for contrast limiting (default is 20.0).
        tile_grid_size : tuple of int, optional
            Size of the grid for applying CLAHE (default is (8, 8)).
        normalize_output : bool, optional
            Whether to normalize the output image after equalization (default is False).

        Returns:
        --------
        numpy.ndarray
            The equalized image, with the same shape as the input.
        """

        # Convert image to grayscale if it's in BGR format (3 channels)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        # Ensure image is of type uint8 (required for CLAHE)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        equalized_stack = np.zeros_like(image)

        # Create a CLAHE object with the provided clip limit and grid size
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE (enhancing local contrast)
        depth = image.shape[0]
        for z in range(depth):
            slice = image[z, :, :]
            slice = clahe.apply(slice)
        # If requested, normalize the output (rescale pixel values to 0-255)
            if normalize_output:
                slice = np.where(slice == 0, 0, slice)  # Preserve zero-pixel areas
                slice = cv2.normalize(slice, None, -21, 255, cv2.NORM_MINMAX)
            equalized_stack[z, :, :] = slice
        
        equalized_stack = np.stack(equalized_stack, axis=0)
        equalized_stack = np.squeeze(equalized_stack)

        return equalized_stack

    def erode_labeled_masks(
        mask_array, 
        kernel_size=3, 
        min_area_to_erode=0  # Default: no size filtering (erode all)
    ):
        """
        Erodes labeled masks (uint16) while preserving small masks.
        
        Args:
            mask_array: 2D uint16 array (0=background, non-zero=mask IDs)
            kernel_size: Erosion kernel size (default=3)
            min_area_to_erode: Masks smaller than this area skip erosion (default=0=no skip)
        
        Returns:
            Eroded mask array (uint16)
        """
        eroded_masks = np.zeros_like(mask_array)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for mask_id in np.unique(mask_array):
            if mask_id == 0:
                continue
                
            binary_mask = np.uint8(mask_array == mask_id) * 255
            area = np.sum(binary_mask) / 255  # Calculate mask area in pixels
            
            # Skip erosion for small masks
            if area < min_area_to_erode:
                eroded_masks[binary_mask == 255] = mask_id
                continue
                
            # Proceed with erosion for larger masks
            eroded_binary = cv2.erode(binary_mask, kernel)
            eroded_masks[eroded_binary == 255] = mask_id
        
        return eroded_masks
    
    test_dir = os.path.join(main_dir, "testing")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".tif")]
    for test_file in test_files:
        with tifffile.TiffFile(test_file) as tif:
            image = tif.asarray()
            image = _equalize_image_clahe(image, clip_limit=20.0, tile_grid_size=(4, 4), normalize_output=True)

            detector = CellPoseDetector(image, model=model_path, flow_threshold=flow_threshold, 
                                        cellprob_threshold=cellprob_threshold,
                                        diameter=diameter, min_canvas_size=min_canvas_size)
            masks = detector.detect()
            masks = erode_labeled_masks(masks, kernel_size=3, min_area_to_erode=0)
            basename = os.path.splitext(os.path.basename(test_file))[0]
            output_path = os.path.join(test_dir, f"{basename}_segrun.tif")
            tifffile.imwrite(output_path, masks)

def test_detection_podosomes(main_dir, model_path=None, diameter=10, flow_threshold=0.4, cellprob_threshold=0.0,
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
            detector = PodosomeDetector(image, diameter=diameter)
            masks = detector.detect()
            print(f"Detected mask shape: {masks.shape}")
            masks = np.max(masks, axis=0)
            basename = os.path.splitext(os.path.basename(test_file))[0]
            output_path = os.path.join(test_dir, f"{basename}_segrun.tif")
            tifffile.imwrite(output_path, masks)

def generate_segmentation_overlay(image, gt_masks, det_masks, alpha=0.5):
    """
    Generates an overlaid image showing TP, FP, FN with transparency.
    
    Args:
        image (np.ndarray):      Original 2D grayscale/RGB image.
        gt_masks (np.ndarray):    2D array of ground truth mask IDs.
        det_masks (np.ndarray):   2D array of detected mask IDs.
        alpha (float):            Transparency level (0-1).
    
    Returns:
        overlaid (np.ndarray):    RGB image with transparent error overlay.
    """
    # Ensure input image is 3-channel (RGB)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Initialize error overlay (BGR format)
    overlay = np.zeros_like(image)
    
    # Compute error masks
    TP = (gt_masks > 0) & (det_masks > 0)  # True Positives (correct)
    FP = (det_masks > 0) & (gt_masks == 0)  # False Positives (extra detections)
    FN = (gt_masks > 0) & (det_masks == 0)  # False Negatives (missed)
    
    # Assign colors (BGR: OpenCV default)
    overlay[TP] = [0, 255, 0]    # Green for True Positives (correct detections)
    overlay[FP] = [255, 0, 0]    # Red for False Positives (false alarms)
    overlay[FN] = [0, 255, 255]    # Blue for False Negatives (missed detections)
    
    # Blend with original image
    overlaid = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return overlaid

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
        print(image.shape)
        if image.ndim == 3:
            image = np.max(image, axis=0)
        print(image.shape)

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

        # matched_pairs, overlap_image, highlighted_image = match_masks(detected_masks, ground_truth_masks, match_threshold=overlap_threshold)
        # metrics.extend(matched_pairs)

        # output_path = os.path.join(test_dir, f"{basename}_comparison.tif")
        # output_path_fn = os.path.join(test_dir, f"{basename}_fn.tif")
        # tifffile.imwrite(output_path, overlap_image.astype(np.uint8), photometric='rgb')
        # tifffile.imwrite(output_path_fn, highlighted_image.astype(np.uint8), photometric='rgb')
        matched_pairs, overlap_image, false_negative_vis, false_positive_vis = match_masks(
            detected_masks, 
            ground_truth_masks, 
            match_threshold=overlap_threshold
        )
        metrics.extend(matched_pairs)

        overlay = generate_segmentation_overlay(
            image, 
            ground_truth_masks, 
            detected_masks, 
            alpha=0.25
        )

        # Save all three visualization images
        output_path = os.path.join(test_dir, f"{basename}_comparison.tif")
        output_path_fn = os.path.join(test_dir, f"{basename}_fn.tif")
        output_path_fp = os.path.join(test_dir, f"{basename}_fp.tif")
        outout_path_overlay = os.path.join(test_dir, f"{basename}_overlay.tif")

        tifffile.imwrite(output_path, overlap_image.astype(np.uint8), photometric='rgb')
        tifffile.imwrite(output_path_fn, false_negative_vis.astype(np.uint8), photometric='rgb')
        tifffile.imwrite(output_path_fp, false_positive_vis.astype(np.uint8), photometric='rgb')
        tifffile.imwrite(outout_path_overlay, overlay.astype(np.uint8), photometric='rgb')

    if metrics:
        plot_overlap_statistics(metrics, main_dir, threshold=overlap_threshold)


def plot_overlap_statistics(matched_pairs=None, main_dir=None, threshold=0.0, force_recompute=False):
    """
    Generate and save statistics plots for the matched pairs.

    Args:
        matched_pairs (list): List of tuples (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou).
        main_dir (str): Main directory to save results and plots.
        threshold (float): Minimum threshold for metrics to be included in the plots.
    """
    csv_path = os.path.join(main_dir, 'matched_pairs.csv')
    plots_dir = os.path.join(main_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    recall_overlap_values = [pair[2] for pair in matched_pairs if pair[2] is not None and pair[2] >= threshold]
    f1_values = [pair[3] for pair in matched_pairs if pair[3] is not None and pair[3] >= threshold]
    boundary_iou_values = [pair[4] for pair in matched_pairs if pair[4] is not None and pair[4] >= threshold]
    fpr_values = [pair[5] for pair in matched_pairs if pair[5] is not None]
    fnr_values = [pair[6] for pair in matched_pairs if pair[6] is not None]
    radial_similarity = [pair[7] for pair in matched_pairs if pair[7] is not None]
    iou_values = [pair[8] for pair in matched_pairs if pair[8] is not None and pair[8] >= threshold]

    def safe_stats(values):
        return np.mean(values) if values else np.nan, np.median(values) if values else np.nan, np.std(values) if values else np.nan

    matches = sum(1 for pair in matched_pairs if pair[0] is not None and pair[1] is not None)
    false_positives = sum(1 for pair in matched_pairs if pair[0] is None and pair[1] is not None)
    false_negatives = sum(1 for pair in matched_pairs if pair[0] is not None and pair[1] is None)

    print(f"""
        🔍 Detection Results:
        True Positives (Matches): {matches}
        False Positives: {false_positives}
        False Negatives: {false_negatives}
            """)
    
    metrics = {
        "IoU": iou_values,
        "Recall Overlap": recall_overlap_values,
        "F1 Score": f1_values,
        "Radial Similarity": radial_similarity,
        "Boundary IoU": boundary_iou_values,
        "False Positive Rate": fpr_values,
        "False Negative Rate": fnr_values,
    }

    metrics_file_path = os.path.join(main_dir, 'metrics_overview.txt')
    with open(metrics_file_path, 'w', encoding='utf-8') as metrics_file:
        for name, values in metrics.items():
            mean, median, std = safe_stats(values)
            log_line = f"{name}: Mean = {mean:.2f}, Median = {median:.2f}, Std Dev = {std:.2f}, n = {len(values)}\n"
            print(f"📊 {name}: Mean = {mean:.2f}, Median = {median:.2f}, Std Dev = {std:.2f}, n = {len(values)}")
            metrics_file.write(log_line)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Mask Matching Statistics', fontsize=18)

    metric_list = list(metrics.keys())
    metric_values = list(metrics.values())
    #colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'brown', 'pink']
    #sns.set_theme()
    sns.set_style("white")
    num_colors = max(len(metrics), 7)  # Ensure at least 7 colors
    cmap = sns.color_palette("mako", as_cmap=True)
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    bar_colors = [cmap(i / 3) for i in range(3)]
    for i, (metric, values) in enumerate(zip(metric_list, metric_values)):
        if values:
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)

            ax = axes[i // 3, i % 3]
            #ax.hist(values, bins=20, cmap='mako', edgecolor='black', alpha=0.7)
            #ax.hist(values, bins=20, color=colors[i], edgecolor='black', alpha=0.7, label=metric)
            # Compute histogram manually
            counts, bins_edges = np.histogram(values, bins=20)
            bin_width = bins_edges[1] - bins_edges[0]

            # Normalize counts for colormap
            norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
            cmap = plt.get_cmap('mako')

            # Plot each bar individually with dynamic color
            for count, left in zip(counts, bins_edges[:-1]):
                color = cmap(norm(count))
                ax.bar(left + bin_width/2, count, width=bin_width * 0.95,
                    color=color, edgecolor='black', alpha=0.7)

            # Overlay mean/median lines
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
    #axes[2, 2].bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', alpha=0.7)
    axes[2, 2].bar(categories, counts, color=bar_colors, edgecolor='black', alpha=0.7)
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

    # full_fig_path = os.path.join(plots_dir, 'overlap_statistics_full.png')
    # plt.savefig(full_fig_path, dpi=600, bbox_inches='tight')
    # plt.close(fig)

    full_fig_path = os.path.join(plots_dir, 'overlap_statistics_full.png')
    plt.figure(fig.number)
    plt.savefig(full_fig_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    for metric, values in metrics.items():
        if values:
            fig, ax = plt.subplots(figsize=(8, 2))
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            #ax.hist(values, bins=20, color=colors[metric_list.index(metric)], edgecolor='black', alpha=0.7, label=metric)

            # Compute histogram manually
            counts, bins_edges = np.histogram(values, bins=20)
            bin_width = bins_edges[1] - bins_edges[0]

            # Normalize counts for colormap
            norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
            cmap = plt.get_cmap('mako')

            # Plot each bar individually with dynamic color
            for count, left in zip(counts, bins_edges[:-1]):
                color = cmap(norm(count))
                ax.bar(left + bin_width/2, count, width=bin_width * 0.95,
                    color=color, edgecolor='black', alpha=0.7)

            ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.plot([], [], ' ', label=f'Std Dev: {std_val:.2f}')
            ax.plot([], [], ' ', label=f'n = {len(values)}')

            # ax.text(mean_val, ax.get_ylim()[1] * 0.7, f'σ={std_val:.2f}', color='blue', fontsize=10)
            ax.set_title(f'{metric}', fontsize=16)
            #ax.set_xlabel(metric)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            # ax.legend(title=f"Std Dev: {std_val:.2f}")
            # ax.legend(fontsize=12)
            # legend.get_frame().set_edgecolor('dimgray')
            legend = ax.legend(fontsize=12)
            legend.get_frame().set_edgecolor('dimgray')

            # ax.text(0.5, 0.92, f'n = {len(values)}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
            #         horizontalalignment='right', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
            plot_path = os.path.join(plots_dir, f'{metric.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path, dpi=600, bbox_inches='tight')
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', alpha=0.7)
    # for i, count in enumerate(counts):
    #     ax.text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')
    # cmap = sns.color_palette("mako", 3)
    # colors = [cmap(i / num_colors) for i in range(num_colors)]
    # bar_colors = [cmap(i / 3) for i in range(3)]
    # cmap = sns.color_palette("mako", 3)

    # counts = [matches, false_positives, false_negatives]
    # bar_colors = sns.color_palette("mako", len(categories))

    # print(f"categories = {categories}")
    # print(f"counts = {counts}")
    # print(f"bar_colors = {bar_colors}")
    # axes[2, 2].clear()
    # axes[2, 2].bar(categories, counts, color=bar_colors, edgecolor='black', alpha=0.7)
    # for i, count in enumerate(counts):
    #     axes[2, 2].text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')

    # axes[2, 2].set_title('Match/False Positive/False Negative Counts')
    # axes[2, 2].set_ylabel('Count')
    # axes[2, 2].text(
    #     0.95, 0.95, f'n = {len(matched_pairs)}',
    #     transform=axes[2, 2].transAxes,
    #     fontsize=12,
    #     verticalalignment='top',
    #     horizontalalignment='right',
    #     color='black',
    #     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
    # )
    # counts_path = os.path.join(plots_dir, 'match_false_positive_false_negative_counts.png')
    # plt.savefig(counts_path, dpi=600, bbox_inches='tight')
    # plt.close(fig)

    # === Standalone Match/FP/FN Bar Plot ===
    # categories = ['Matches', 'False Positives', 'False Negatives']
    # counts = [matches, false_positives, false_negatives]
    # bar_colors = sns.color_palette("mako", len(categories))

    # print(f"categories = {categories}")
    # print(f"counts = {counts}")
    # print(f"bar_colors = {bar_colors}")

    # # Create new figure and use ax directly
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.bar(categories, counts, color=bar_colors, edgecolor='black', alpha=0.7)

    # # Add text labels
    # for i, count in enumerate(counts):
    #     ax.text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')

    # # Set title and label
    # ax.set_title('Match/False Positive/False Negative Counts')
    # ax.set_ylabel('Count')
    # ax.text(
    #     0.95, 0.95, f'n = {len(matched_pairs)}',
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment='top',
    #     horizontalalignment='right',
    #     color='black',
    #     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
    # )

    # # Save and close
    # counts_path = os.path.join(plots_dir, 'match_false_positive_false_negative_counts.png')
    # plt.tight_layout()
    # plt.savefig(counts_path, dpi=600, bbox_inches='tight')
    # plt.close(fig)

    categories = ['Matches', 'False Positives', 'False Negatives']
    counts = [matches, false_positives, false_negatives]
    bar_colors = sns.color_palette("mako", len(categories))
    bar_width2 = 0.05

    print(f"categories = {categories}")
    print(f"counts = {counts}")
    print(f"bar_colors = {bar_colors}")
    print(f"bar_width2 = {bar_width2}")
    # Create new figure with slim horizontal layout
    fig, ax = plt.subplots(figsize=(10, 2))  # wider but shorter height

    # Plot horizontal bars
    y_positions = range(len(categories))
    y_positions = [i * 0.075 for i in range(len(categories))] 
    #ax.barh(y_positions, counts, color=bar_colors, width=bar_width2, edgecolor='black', alpha=0.7)
    ax.barh(y=y_positions, width=counts, height=bar_width2, color=bar_colors, edgecolor='black', alpha=0.7)
    # Annotate bars with count values
    for idx, (count, pos) in enumerate(zip(counts, y_positions)):
        ax.text(count + max(counts) * 0.01, pos, f'{count}', va='center', fontsize=12, color='black')

    # Style the axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Count')
    ax.set_title('Per-Instance Detection Breakdown')
    ax.grid(False)  # Turn off grid to match other plots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('gray')

    # Optional: Add total count box
    ax.text(0.95, 0.25, f'n = {len(matched_pairs)}',
            transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    # Tight layout for clean spacing
    plt.tight_layout()
    counts_path = os.path.join(plots_dir, 'match_false_positive_false_negative_counts.png')
    plt.savefig(counts_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    csv_path = os.path.join(main_dir, 'matched_pairs.csv')
    df = pd.DataFrame(matched_pairs, columns=['gt_id', 'detected_id', 'overlap', 'f1', 'boundary_iou', 'fpr', 'fnr', 'radial_similarity', 'iou'])
    df.to_csv(csv_path, index=False)

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
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

def calculate_radial_similarity(gt_mask, detected_mask, num_directions=360):
    """
    Calculate radial shape similarity between two masks.
    
    Args:
        gt_mask (np.array): Ground truth mask
        detected_mask (np.array): Detected mask
        num_directions (int): Number of radial directions to sample (default 360)
        
    Returns:
        float: Radial similarity score (0-1)
    """
    # Find centroids
    gt_moments = cv2.moments(gt_mask.astype(np.uint8))
    detected_moments = cv2.moments(detected_mask.astype(np.uint8))
    
    gt_centroid = (gt_moments['m10']/gt_moments['m00'], gt_moments['m01']/gt_moments['m00'])
    detected_centroid = (detected_moments['m10']/detected_moments['m00'], 
                         detected_moments['m01']/detected_moments['m00'])
    
    # Calculate radial distances for both masks
    gt_distances = _calculate_radial_distances(gt_mask, gt_centroid, num_directions)
    detected_distances = _calculate_radial_distances(detected_mask, detected_centroid, num_directions)
    
    # Calculate similarity scores for each direction
    similarity_scores = np.zeros(num_directions)
    for i in range(num_directions):
        gt_dist = gt_distances[i]
        detected_dist = detected_distances[i]
        
        if gt_dist == 0 and detected_dist == 0:
            similarity_scores[i] = 1.0
        elif gt_dist == 0 or detected_dist == 0:
            similarity_scores[i] = 0.0
        else:
            ratio = min(gt_dist, detected_dist) / max(gt_dist, detected_dist)
            similarity_scores[i] = ratio
    
    return np.mean(similarity_scores)

@jit(nopython=True)
def _calculate_radial_distances(mask, centroid, num_directions):
    """
    Calculate radial distances from centroid to boundary in each direction.
    """
    height, width = mask.shape
    distances = np.zeros(num_directions)
    
    for angle_idx in prange(num_directions):
        theta = radians(angle_idx * (360 / num_directions))
        dx, dy = cos(theta), sin(theta)
        
        max_dist = min(width, height)  # Maximum possible distance
        
        for dist in np.linspace(0, max_dist, num=int(max_dist*2)):
            x = int(centroid[0] + dist * dx)
            y = int(centroid[1] + dist * dy)
            
            if x < 0 or x >= width or y < 0 or y >= height:
                break
                
            if not mask[y, x]:
                distances[angle_idx] = dist
                break
                
    return distances

def generate_highlighted_visualizations(unique_matched_pairs, ground_truth_masks, detected_masks):
    """
    Generate two RGB visualizations:
    1. Highlighting false negatives (red)
    2. Highlighting false positives (blue)
    
    Args:
        unique_matched_pairs (list of tuples): List of match tuples (gt_id, detected_id, ...).
        ground_truth_masks (numpy array): Array of ground truth mask IDs.
        detected_masks (numpy array): Array of detected mask IDs.
        
    Returns:
        tuple: (false_negative_vis, false_positive_vis)
    """
    # Initialize both visualizations
    false_negative_vis = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)
    false_positive_vis = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)
    
    # First pass: mark all matched pairs in gray and false negatives in red
    for pair in unique_matched_pairs:
        gt_id, detected_id = pair[:2]
        if gt_id is not None and detected_id is None:  # False negative
            false_negative_vis[ground_truth_masks == gt_id] = [255, 0, 0]  # Red
    
    for gt_id, detected_id, *_ in unique_matched_pairs:
        if gt_id is not None and detected_id is not None:  # Matched pair
            false_negative_vis[(ground_truth_masks == gt_id) | (detected_masks == detected_id)] = [50, 50, 50]
    
    # Second pass: mark all matched pairs in gray and false positives in blue
    for pair in unique_matched_pairs:
        gt_id, detected_id = pair[:2]
        if detected_id is not None and gt_id is None:  # False positive
            false_positive_vis[detected_masks == detected_id] = [0, 0, 255]  # Blue
    
    for gt_id, detected_id, *_ in unique_matched_pairs:
        if gt_id is not None and detected_id is not None:  # Matched pair
            false_positive_vis[(ground_truth_masks == gt_id) | (detected_masks == detected_id)] = [50, 50, 50]
    
    return false_negative_vis, false_positive_vis


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
                #asymmetry_score = calculate_asymmetry(overlap, f1)
                radial_similarity = calculate_radial_similarity(gt_mask, detected_mask)
                matched_pairs.append((gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, radial_similarity, iou))

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
                #asymmetry_score = calculate_asymmetry(overlap, f1)
                radial_similarity = calculate_radial_similarity(gt_mask, detected_mask)
                matched_pairs.append((gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, radial_similarity, iou))

    unique_matched_pairs = []
    seen_pairs = {}
    for gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, radial_similarity, iou in matched_pairs:
        pair_key = (gt_id, detected_id)
        if pair_key not in seen_pairs or overlap > seen_pairs[pair_key][2]:
            seen_pairs[pair_key] = (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, radial_similarity, iou)
    unique_matched_pairs = list(seen_pairs.values())

    seen_gt_ids = {pair[0] for pair in unique_matched_pairs if pair[0] is not None}
    for gt_id in gt_masks:
        if gt_id not in seen_gt_ids:
            unique_matched_pairs.append((gt_id, None, 0, 0, 0, None, None, None, 0))

    seen_detected_ids = {pair[1] for pair in unique_matched_pairs if pair[1] is not None}
    for detected_id in detected_masks_dict:
        if detected_id not in seen_detected_ids:
            unique_matched_pairs.append((None, detected_id, 0, 0, 0, None, None, None, 0))

    # color_coded = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)
    # for gt_id in gt_masks:
    #     color_coded[ground_truth_masks == gt_id, 1] = 255
    # for detected_id in detected_masks_dict:
    #     color_coded[detected_masks == detected_id, 2] = 255

    # highlighted_visualization = generate_highlighted_visualization(unique_matched_pairs, ground_truth_masks, detected_masks)
    # return unique_matched_pairs, color_coded, highlighted_visualization
    color_coded = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)
    for gt_id in gt_masks:
        color_coded[ground_truth_masks == gt_id, 1] = 255  # Green for ground truth
    for detected_id in detected_masks_dict:
        color_coded[detected_masks == detected_id, 2] = 255  # Blue for detected
    
    # Generate both visualizations
    false_negative_vis, false_positive_vis = generate_highlighted_visualizations(
        unique_matched_pairs, ground_truth_masks, detected_masks
    )
    
    return unique_matched_pairs, color_coded, false_negative_vis, false_positive_vis


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
    model_path = r"D:\Coding\BioPixel\models\podotect"
    
    # THIS IS FOR MACROPHAGE DETECTION
    # test_detection(main_dir, model_path=model_path, min_canvas_size=(2000, 2000), diameter=750)

    # THIS IS FOR Podosome DETECTION
    #test_detection_podosomes(main_dir, model_path=model_path)

    compare_and_visualize(main_dir, overlap_threshold=0.02)
    #plot_overlap_statistics(main_dir=main_dir)
    print("Cellpose training and testing completed!")

