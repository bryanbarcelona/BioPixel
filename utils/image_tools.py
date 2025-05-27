import os

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import tifffile

def save_tif(
    image: np.ndarray,
    path: str,
    metadata: Dict,
    colormap: Optional[np.ndarray] = None,
    photometric: str = "minisblack",
    imagej: bool = True,
) -> None:
    """Save a 5D image array as a TIFF file.

    Args:
        image: 5D image array with shape (Time, Z, Channel, Y, X).
        path: Path to save the TIFF file.
        metadata: Dictionary containing metadata for ImageJ compatibility.
        colormap: Optional colormap for the image. Defaults to None.
        photometric: Photometric interpretation mode. Defaults to "minisblack".
        imagej: If True, include ImageJ-compatible metadata. Defaults to True.

    Raises:
        OSError: If the directory cannot be created or file cannot be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(
        path,
        image,
        photometric=photometric,
        imagej=imagej,
        resolution=(metadata["x_resolution"], metadata["y_resolution"]),
        metadata=metadata,
    )

def normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
    """Normalize image array to [0, 255] range and convert to uint8.

    Args:
        image_array: Input image array.

    Returns:
        Normalized image array of dtype uint8.

    Raises:
        ValueError: If the input dtype is not uint8, uint16, or float64.
    """
    supported_dtypes = {np.uint8, np.uint16, np.float64}
    if image_array.dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype {image_array.dtype}. Supported: {supported_dtypes}")

    min_val, max_val = image_array.min(), image_array.max()
    if max_val == min_val:
        return np.zeros_like(image_array, dtype=np.uint8)
    normalized = (image_array - min_val) * (255 / (max_val - min_val))
    return normalized.astype(np.uint8)

def save_as_png(
    image_array: np.ndarray,
    directory: str,
    hist_equalization: bool = False,
) -> None:
    """Save a 2D NumPy array as a PNG image.

    Args:
        image_array: 2D NumPy array representing the image.
        directory: Path and filename for saving the PNG image.
        hist_equalization: If True, apply histogram equalization. Defaults to False.

    Raises:
        ValueError: If the input array is not 2D.

    Notes:
        Input values outside [0, 255] are normalized to uint8 range.
    """
    image_array = np.squeeze(image_array)
    if image_array.ndim != 2:
        raise ValueError("Input array must be a 2D image.")

    if not np.all((image_array >= 0) & (image_array <= 255)):
        print("Images are not 8-bit, normalizing (image_tools.save_as_png)")

    image_array = normalize_to_uint8(image_array)
    # Histogram equalization is currently disabled
    # if hist_equalization:
    #     image_array = normalize_to_uint8(exposure.equalize_hist(image_array))

    pil_image = Image.fromarray(image_array)
    pil_image.save(directory)
    print(f"Blended image saved as PNG: {directory}")

def load_5d_tif(tiff_path: str) -> Tuple[np.ndarray, Dict, Tuple[float, float], Tuple]:
    """Load a multi-dimensional TIFF file into a 5D array.

    Args:
        tiff_path: Path to the input TIFF file.

    Returns:
        Tuple containing:
        - 5D NumPy array (TZCYX).
        - ImageJ metadata dictionary.
        - Tuple of (x_resolution, y_resolution).
        - Tuple of (z_spacing, channel_count, frame_count, slice_count).
    """
    all_axes = "TZCYX"
    with tifffile.TiffFile(tiff_path) as tif:
        data = tifffile.imread(tiff_path)
        axes = tif.series[0].axes
        metadata = tif.imagej_metadata

        x_res = tif.pages[0].tags["XResolution"].value
        x_resolution = float(x_res[0]) / float(x_res[1])
        y_res = tif.pages[0].tags["YResolution"].value
        y_resolution = float(y_res[0]) / float(y_res[1])
        xy_cal = (x_resolution, y_resolution)

        missing_axes = [i for i, dim in enumerate(all_axes) if dim not in axes]
        for missing_index in missing_axes:
            data = np.expand_dims(data, axis=missing_index)

        config = (metadata.get("spacing", 0.0), data.shape[2], data.shape[0], data.shape[1])
        return data, metadata, xy_cal, config

def get_ranges(image_array: np.ndarray) -> Dict[str, Tuple]:
    """Calculate min/max ranges for each channel in a 5D image array.

    Args:
        image_array: 5D image array with shape (Time, Z, Channel, Y, X).

    Returns:
        Dictionary containing:
        - 'Ranges': Tuple of (min, max) for each channel.
        - 'min': Maximum of all channel minimums.
        - 'max': Minimum of all channel maximums.

    Raises:
        ValueError: If input array is not 5D.
    """
    if len(image_array.shape) != 5:
        raise ValueError("Input must be a 5D array (Time, Z, Y, X).")

    min_max_tuples = []
    all_min_values = []
    all_max_values = []

    for channel in range(image_array.shape[2]):
        channel_data = image_array[:, :, channel, :, :]
        min_val = float(np.min(channel_data))
        max_val = float(np.max(channel_data))
        all_min_values.append(min_val)
        all_max_values.append(max_val)
        min_max_tuples.extend([min_val, max_val])

    range_tuple = tuple(round(val, 1) for val in min_max_tuples)
    return {
        "Ranges": range_tuple,
        "min": int(np.max(all_min_values)),
        "max": int(np.min(all_max_values)),
    }
