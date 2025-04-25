import os
import sys
import random
import shutil
import copy

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from cellpose import models
from cellpose.train import train_seg
from cellpose.io import load_train_test_data
from cell_detector import CellPoseDetector
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_main_dir(main_dir=None):
    """
    Get the main directory path containing training, validation, and test datasets.
    
    Args:
        main_dir (str, optional): Path to the main directory. If provided, user input will be skipped.
                                  If None, the user will be prompted for input.
    
    Returns:
        str: Path to the main directory.
    """
    if main_dir is None:
        main_dir = input("📂 Enter the main directory path (containing 'training', 'validation', 'testing' subfolders): ").strip()
    # Remove any leading/trailing quotes if the user accidentally added them
    if main_dir.startswith('"') and main_dir.endswith('"'):
        main_dir = main_dir[1:-1]

    # Ensure all required subdirectories exist
    required_dirs = ["training", "validation", "testing"]
    for subdir in required_dirs:
        subdir_path = os.path.join(main_dir, subdir)
        print(f"📂 Checking for required folder: {subdir_path}")
        if not os.path.isdir(subdir_path):
            os.makedirs(subdir_path)
            print(f"📁 Created missing folder: {subdir_path}")
    
    return main_dir

def setup_data_split(main_dir, split_ratio=(70, 15, 15)):
    """
    Set up the data split into training, validation, and testing folders based on the given ratio.
    Ensures that every .tif file has a corresponding _seg.npy file.
    The .tif and _seg.npy pairs are expected to be in a subfolder called 'labeled'.

    Args:
        main_dir (str): Path to the main directory containing the 'labeled' subfolder.
        split_ratio (tuple): A tuple representing the ratio for training, validation, and testing.
                             Default is (70, 15, 15).
    """
    # Ensure the split ratio sums to 100
    if sum(split_ratio) != 100:
        raise ValueError("The split ratio must sum to 100.")

    # Create subfolders if they don't exist
    train_dir = os.path.join(main_dir, "training")
    val_dir = os.path.join(main_dir, "validation")
    test_dir = os.path.join(main_dir, "testing")

    for subdir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
            print(f"📁 Created directory: {subdir}")

    # Path to the 'labeled' subfolder
    labeled_dir = os.path.join(main_dir, "labeled")
    if not os.path.exists(labeled_dir):
        raise FileNotFoundError(f"❌ 'labeled' subfolder not found in: {main_dir}")

    # Get all .tif files in the 'labeled' subfolder
    tif_files = [f for f in os.listdir(labeled_dir) if f.endswith(".tif")]

    if not tif_files:
        return
    
    # Ensure that each .tif file has a corresponding _seg.npy file
    valid_pairs = []
    for tif_file in tif_files:
        basename = os.path.splitext(tif_file)[0]
        npy_file = basename + "_seg.npy"
        npy_path = os.path.join(labeled_dir, npy_file)
        
        if not os.path.exists(npy_path):
            print(f"⚠️ No matching _seg.npy file found for: {tif_file}. Skipping this file.")
        else:
            valid_pairs.append((tif_file, npy_file))

    if not valid_pairs:
        raise ValueError("❌ No valid .tif and _seg.npy pairs found in the 'labeled' subfolder.")

    # Shuffle the list of valid pairs to ensure random distribution
    random.shuffle(valid_pairs)

    # Calculate the number of files for each split
    total_files = len(valid_pairs)
    train_count = int(total_files * split_ratio[0] / 100)
    val_count = int(total_files * split_ratio[1] / 100)
    test_count = total_files - train_count - val_count

    # Check if the sum of train_count, val_count, and test_count equals total_files
    # If not, redistribute the remaining files to the training set (or any other set)
    if train_count + val_count + test_count != total_files:
        remaining_files = total_files - (train_count + val_count + test_count)
        train_count += remaining_files  # Add remaining files to training set
        print(f"⚠️ Rounding issue detected. Redistributing {remaining_files} files to training set.")

    # Distribute files into training, validation, and testing folders
    for i, (tif_file, npy_file) in enumerate(valid_pairs):
        if i < train_count:
            dest_dir = train_dir
        elif i < train_count + val_count:
            dest_dir = val_dir
        else:
            dest_dir = test_dir

        # Move .tif file
        shutil.move(os.path.join(labeled_dir, tif_file), os.path.join(dest_dir, tif_file))
        # Move corresponding .npy file
        shutil.move(os.path.join(labeled_dir, npy_file), os.path.join(dest_dir, npy_file))

    print(f"📊 Data split completed: {train_count} training, {val_count} validation, {test_count} testing")

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
    """

    print("🚀 Starting macrophage model training...")

    # 🗂 Automatically set paths
    train_dir = os.path.join(main_dir, "training")
    val_dir = os.path.join(main_dir, "validation")
    test_dir = os.path.join(main_dir, "testing")
    model_dir = os.path.join(main_dir, "models")

    # Ensure the models directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created models directory at: {model_dir}")

    # 🏗 Load Training & Validation Data Automatically
    train_images, train_labels, train_paths, val_images, val_labels, val_paths = load_train_test_data(
        train_dir=train_dir,
        test_dir=val_dir,
        mask_filter="_seg.npy"
    )

    print(f"📊 Training on {len(train_images)} images, Validating on {len(val_images)} images")

    # 🔥 Initialize Model (start from pre-trained cyto3)
    model = models.CellposeModel(gpu=True, model_type="cyto3")

    # 🎯 Training with Validation
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
    print(f"✅ Training complete! Model saved at: {str(model_path[0])}")
    return str(model_path[0])

def test_detection(main_dir, model_path=None, diameter=10, flow_threshold=0.4, min_canvas_size=None):
    """
    Test the trained model on the testing dataset.
    
    Args:
        main_dir (str): Path to the main directory containing 'training', 'validation', and 'testing' subfolders.
    """

    print("🚀 Running test detection...")

    # 🗂 Automatically set paths
    test_dir = os.path.join(main_dir, "testing")
    # model_dir = os.path.join(main_dir, "models", "cellpose_1741136855.6751761")

    # # Ensure the models directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ Test directory not found at: {test_dir}")

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".tif")]
    print(f"📊 Testing on {len(test_files)} images")

    for test_file in test_files:
        print(f"🔍 Processing: {test_file}")

        with tifffile.TiffFile(test_file) as tif:
            image = tif.asarray()

            detector = CellPoseDetector(image, model=model_path, flow_threshold=flow_threshold, 
                                        diameter=diameter, min_canvas_size=min_canvas_size)

            masks, *_ = detector.detect()

            # Get the basename of the test file (without extension)
            basename = os.path.splitext(os.path.basename(test_file))[0]

            # Define the output path for the mask
            output_path = os.path.join(test_dir, f"{basename}_segrun.tif")

            # Save the mask as a TIFF file
            tifffile.imwrite(output_path, masks)

            print(f"✅ Saved mask to: {output_path}")

def compare_and_visualize(main_dir, overlap_threshold=0.5):
    """
    Compare detected masks with ground truth masks and generate a visual representation using OpenCV.

    Args:
        main_dir (str): Path to the main directory containing 'training', 'validation', and 'testing' subfolders.
    """

    print("🚀 Comparing detection to ground truth masks (IoU)...")

    # 🗂 Automatically set paths
    test_dir = os.path.join(main_dir, "testing")

    # Ensure the testing directory exists
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ Testing directory not found at: {test_dir}")

    # Get all TIFF files in the testing directory
    tiff_files = [f for f in os.listdir(test_dir) if f.endswith(".tif") and not f.endswith("_segrun.tif")]

    metrics = []

    for tiff_file in tiff_files:
        print(f"🔍 Processing: {tiff_file}")

        # Load the original image
        image_path = os.path.join(test_dir, tiff_file)
        image = tifffile.imread(image_path)

        # Load the detected masks
        basename = os.path.splitext(tiff_file)[0]
        detected_mask_path = os.path.join(test_dir, f"{basename}_segrun.tif")
        if not os.path.exists(detected_mask_path):
            print(f"⚠️ Detected masks not found for: {tiff_file}")
            continue
        detected_masks = tifffile.imread(detected_mask_path)

        # Load the ground truth masks
        npy_path = os.path.join(test_dir, f"{basename}_seg.npy")
        if not os.path.exists(npy_path):
            print(f"⚠️ Ground truth masks not found for: {tiff_file}")
            continue
        mask_data = np.load(npy_path, allow_pickle=True).item()
        ground_truth_masks = mask_data.get('masks', None)

        if ground_truth_masks is None:
            print(f"⚠️ No masks found in the ground truth file: {npy_path}")
            continue
        
        
        # Match predicted masks to ground truth masks based on overlap
        matched_pairs, overlap_image, highlighted_image = match_masks(detected_masks, ground_truth_masks, match_threshold=overlap_threshold)

        # Collect overlap values
        for pairs in matched_pairs:
            metrics.append(pairs)

        # Generate a visual representation
        #visual_image = create_visual_representation(image, ground_truth_masks, detected_masks, matched_pairs)

        # Save the visual representation
        output_path = os.path.join(test_dir, f"{basename}_comparison.tif")
        output_path_fn = os.path.join(test_dir, f"{basename}_fn.tif")
        tifffile.imwrite(output_path, overlap_image.astype(np.uint8), photometric='rgb')
        tifffile.imwrite(output_path_fn, highlighted_image.astype(np.uint8), photometric='rgb')
        print(f"✅ Saved comparison image to: {output_path}")

    if metrics:
        plot_overlap_statistics(matched_pairs, main_dir, threshold=overlap_threshold)

def plot_overlap_statistics(matched_pairs, main_dir, threshold=0.0):
    """
    Generate and save statistics plots for the matched pairs, including IoU, F1, Boundary IoU, Recall Overlap, FPR, FNR, and Asymmetry Score.

    Args:
        matched_pairs (list): List of tuples (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou).
        main_dir (str): Main directory to save results and plots.
        threshold (float): Minimum threshold for metrics to be included in the plots.
    """
    # Create a plots subfolder
    plots_dir = os.path.join(main_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Extract metrics from matched pairs, handling None values
    recall_overlap_values = [pair[2] for pair in matched_pairs if pair[2] is not None and pair[2] >= threshold]
    f1_values = [pair[3] for pair in matched_pairs if pair[3] is not None and pair[3] >= threshold]
    boundary_iou_values = [pair[4] for pair in matched_pairs if pair[4] is not None and pair[4] >= threshold]
    fpr_values = [pair[5] for pair in matched_pairs if pair[5] is not None]
    fnr_values = [pair[6] for pair in matched_pairs if pair[6] is not None]
    asymmetry_values = [pair[7] for pair in matched_pairs if pair[7] is not None]
    iou_values = [pair[8] for pair in matched_pairs if pair[8] is not None and pair[8] >= threshold]

    # Convert empty lists to NaN to avoid calculation errors
    def safe_stats(values):
        return np.mean(values) if values else np.nan, np.median(values) if values else np.nan, np.std(values) if values else np.nan

    # Count matches, false positives, and false negatives
    matches = sum(1 for pair in matched_pairs if pair[0] is not None and pair[1] is not None)
    false_positives = sum(1 for pair in matched_pairs if pair[0] is None and pair[1] is not None)
    false_negatives = sum(1 for pair in matched_pairs if pair[0] is not None and pair[1] is None)

    # Print statistics
    print(f"\n📊 Matches: {matches}, False Positives: {false_positives}, False Negatives: {false_negatives}")
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

    # Create a figure with subplots for the full overview
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 3 rows, 3 columns
    fig.suptitle('Mask Matching Statistics', fontsize=18)

    # Define visualization mappings
    metric_list = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'brown', 'pink']

    # Plot histograms for the full overview
    for i, (metric, values) in enumerate(zip(metric_list, metric_values)):
        if values:
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)  # Standard deviation

            ax = axes[i // 3, i % 3]
            ax.hist(values, bins=20, color=colors[i], edgecolor='black', alpha=0.7, label=metric)
            ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')

            # Show standard deviation in plot
            ax.text(mean_val, ax.get_ylim()[1] * 0.7, f'σ={std_val:.2f}', color='blue', fontsize=10)

            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.text(
                0.95, 0.95, f'n = {len(values)}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
            )

    # Plot match/false positive/false negative counts in the last subplot (bottom-right corner)
    categories = ['Matches', 'False Positives', 'False Negatives']
    counts = [matches, false_positives, false_negatives]
    axes[2, 2].bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', alpha=0.7)

    for i, count in enumerate(counts):
        axes[2, 2].text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')

    axes[2, 2].set_title('Match/False Positive/False Negative Counts')
    axes[2, 2].set_ylabel('Count')
    axes[2, 2].text(
        0.95, 0.95, f'n = {len(matched_pairs)}', transform=axes[2, 2].transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
    )

    # Remove any unused subplots if necessary
    used_axes = len(metrics)  # Number of metrics being plotted
    for i in range(used_axes, 8):  # Remove all subplots beyond the used ones
        fig.delaxes(axes[i // 3, i % 3])

    # Save the full overview figure
    full_fig_path = os.path.join(plots_dir, 'overlap_statistics_full.png')
    plt.savefig(full_fig_path, bbox_inches='tight')
    print(f"📁 Saved full figure to {full_fig_path}")

    # Close the full overview figure to free memory
    plt.close(fig)

    # Recreate and save each plot individually
    for metric, values in metrics.items():
        if values:
            # Create a new figure for the individual plot
            fig, ax = plt.subplots(figsize=(8, 6))
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)

            # Plot the histogram
            ax.hist(values, bins=20, color=colors[metric_list.index(metric)], edgecolor='black', alpha=0.7, label=metric)
            ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')

            # Show standard deviation in plot
            ax.text(mean_val, ax.get_ylim()[1] * 0.7, f'σ={std_val:.2f}', color='blue', fontsize=10)

            ax.set_title(f'{metric} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.text(
                0.95, 0.95, f'n = {len(values)}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
            )

            # Save the individual plot
            plot_path = os.path.join(plots_dir, f'{metric.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"📁 Saved {metric} plot to {plot_path}")

    # Save match/false positive/false negative counts as a separate plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', alpha=0.7)
    for i, count in enumerate(counts):
        ax.text(i, count + 1, f'n={count}', ha='center', fontsize=10, color='black')
    ax.set_title('Match/False Positive/False Negative Counts')
    ax.set_ylabel('Count')
    ax.text(
        0.95, 0.95, f'n = {len(matched_pairs)}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
    )
    counts_path = os.path.join(plots_dir, 'match_false_positive_false_negative_counts.png')
    plt.savefig(counts_path, bbox_inches='tight')
    plt.close(fig)
    print(f"📁 Saved match/false positive/false negative counts plot to {counts_path}")

    # Save matched_pairs data as a CSV file
    csv_path = os.path.join(main_dir, 'matched_pairs.csv')
    df = pd.DataFrame(matched_pairs, columns=['gt_id', 'detected_id', 'overlap', 'f1', 'boundary_iou', 'fpr', 'fnr', 'asymmetry_score', 'iou'])
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved matched pairs data to {csv_path}")

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

    gt_boundary = np.zeros_like(gt_mask, dtype=np.uint8)  # Convert to uint8
    detected_boundary = np.zeros_like(detected_mask, dtype=np.uint8)  # Convert to uint8

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
    false_positives = np.sum((detected_mask == 1) & (gt_mask == 0))  # Detected as positive but ground truth is negative
    true_negatives = np.sum((detected_mask == 0) & (gt_mask == 0))   # Detected as negative and ground truth is negative
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
    false_negatives = np.sum((detected_mask == 0) & (gt_mask == 1))  # Detected as negative but ground truth is positive
    true_positives = np.sum((detected_mask == 1) & (gt_mask == 1))   # Detected as positive and ground truth is positive
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
    Generate an RGB visualization where:
    - False negatives (gt_id exists but detected_id is None) are bright red.
    - All other matched masks are faint grey.
    
    Args:
        unique_matched_pairs (list of tuples): List of match tuples (gt_id, detected_id, ...).
        ground_truth_masks (numpy array): Array of ground truth mask IDs.
        detected_masks (numpy array): Array of detected mask IDs.
        
    Returns:
        numpy array: RGB visualization highlighting false negatives.
    """
    highlight_vis = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)  # RGB image

    # Paint false negatives in bright red
    for pair in unique_matched_pairs:
        gt_id, detected_id = pair[:2]
        if gt_id is not None and detected_id is None:
            highlight_vis[ground_truth_masks == gt_id] = [255, 0, 0]  # Bright red

    # Paint all other matched masks in faint grey
    for gt_id, detected_id, *_ in unique_matched_pairs:
        if gt_id is not None and detected_id is not None:
            highlight_vis[(ground_truth_masks == gt_id) | (detected_masks == detected_id)] = [50, 50, 50]  # Faint grey

    return highlight_vis

def match_masks(detected_masks, ground_truth_masks, match_threshold=0.2):
    """
    Match detected masks to ground truth masks using one-sided overlap for evaluation.

    Args:
        detected_masks (np.array): Detected masks array.
        ground_truth_masks (np.array): Ground truth masks array.
        match_threshold (float): Minimum IoU threshold for matching.

    Returns:
        list: List of tuples (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou) representing matched pairs.
        np.array: Color-coded visualization of matches.
    """
    # Precompute all ground truth masks
    gt_masks = {gt_id: (ground_truth_masks == gt_id).astype(bool) for gt_id in np.unique(ground_truth_masks) if gt_id != 0}
    
    # Precompute all detected masks
    detected_masks_dict = {detected_id: (detected_masks == detected_id).astype(bool) for detected_id in np.unique(detected_masks) if detected_id != 0}
    
    matched_pairs = []
    
    # Match ground truth masks to detected masks
    for gt_id, gt_mask in gt_masks.items():
        overlapping_detected_ids = np.unique(detected_masks[gt_mask])
        overlapping_detected_ids = overlapping_detected_ids[overlapping_detected_ids != 0]  # Skip background

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
    
    # Match detected masks to ground truth masks
    for detected_id, detected_mask in detected_masks_dict.items():
        overlapping_gt_ids = np.unique(ground_truth_masks[detected_mask])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids != 0]  # Skip background

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
    
    # Step 3: Remove duplicate matches by keeping the one with the higher overlap
    unique_matched_pairs = []
    seen_pairs = {}
    
    for gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou in matched_pairs:
        pair_key = (gt_id, detected_id)
        if pair_key not in seen_pairs or overlap > seen_pairs[pair_key][2]:
            seen_pairs[pair_key] = (gt_id, detected_id, overlap, f1, boundary_iou, fpr, fnr, asymmetry_score, iou)
    
    unique_matched_pairs = list(seen_pairs.values())
    
    # Add unmatched ground truth masks (false negatives)
    seen_gt_ids = {pair[0] for pair in unique_matched_pairs if pair[0] is not None}
    for gt_id in gt_masks:
        if gt_id not in seen_gt_ids:
            unique_matched_pairs.append((gt_id, None, 0, 0, 0, 0, 0, 0, 0))

    # Add unmatched detected masks (false positives)
    seen_detected_ids = {pair[1] for pair in unique_matched_pairs if pair[1] is not None}
    for detected_id in detected_masks_dict:
        if detected_id not in seen_detected_ids:
            unique_matched_pairs.append((None, detected_id, 0, 0, 0, 0, 0, 0, 0))

    # Visualization
    color_coded = np.zeros((*detected_masks.shape, 3), dtype=np.uint8)  # RGB image
    
    # Paint ground truth masks in green
    for gt_id in gt_masks:
        color_coded[ground_truth_masks == gt_id, 1] = 255  # Green channel

    # Paint detected masks in blue
    for detected_id in detected_masks_dict:
        color_coded[detected_masks == detected_id, 2] = 255  # Blue channel

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
    # Convert the original image to RGB if grayscale
    if image.ndim == 2:
        visual_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        visual_image = image.copy()

    # Function to draw contours
    def draw_mask_contours(mask, color):
        for mask_id in np.unique(mask):
            if mask_id == 0:
                continue  # Skip background

            binary_mask = (mask == mask_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours in the specified color
            cv2.drawContours(visual_image, contours, -1, color, 2)

    # Draw ground truth contours in green
    draw_mask_contours(ground_truth_masks, (0, 255, 0))

    # Draw detected contours in red
    draw_mask_contours(detected_masks, (255, 0, 0))

    # Function to get the center of a bounding box
    def get_center(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])  # Get bounding box of first contour
            return (x + w // 2, y + h // 2)  # Return center point
        return None

    # Label overlap values in matched regions
    for gt_id, detected_id, overlap, _ , _ in matched_pairs:
        gt_mask = (ground_truth_masks == gt_id).astype(np.uint8)
        detected_mask = (detected_masks == detected_id).astype(np.uint8)

        # Find the center of the bounding boxes
        gt_center = get_center(gt_mask)
        detected_center = get_center(detected_mask)

        # Use the average of both centers if both exist, otherwise use the one available
        if gt_center and detected_center:
            text_x = (gt_center[0] + detected_center[0]) // 2
            text_y = (gt_center[1] + detected_center[1]) // 2
        elif gt_center:
            text_x, text_y = gt_center
        elif detected_center:
            text_x, text_y = detected_center
        else:
            continue  # No valid location found

        # Write the overlap value at the calculated position
        overlap_text = f"{overlap:.2f}"
        cv2.putText(visual_image, overlap_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return visual_image


if __name__ == "__main__":

    # 🔹 Get user input for the main directory
    main_dir = get_main_dir()
    
    setup_data_split(main_dir, split_ratio
                     =(60, 20, 20))
    
    #🎯 Train the model with default or user-adjusted parameters
    model_path = train_model(
        main_dir,
        channels=[0, 0],  # Adjust if needed
        learning_rate=0.05,  # Adjust if needed
        weight_decay=0.0001,  # Adjust if needed
        n_epochs=200,  # Adjust if needed
        batch_size=8,  # Adjust if needed
        min_train_masks=1,  # Adjust if needed
        save_every=50,  # Adjust if needed
        normalize=True,  # Adjust if needed
        rescale=True  # Adjust if needed
    )

    #model_path = r"D:\Microscopy Testing\20250302 Macrophage detection training set\Podosomes 002 - biased with good test images\models\models\cellpose_1741390882.7478871"
    test_detection(main_dir, model_path=model_path)

    compare_and_visualize(main_dir, overlap_threshold=0.02)

    print("🎉 Cellpose training and testing completed!")
