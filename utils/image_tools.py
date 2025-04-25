import os
import numpy as np
from PIL import Image
import tifffile
#import matplotlib.pyplot as plt
#from skimage import exposure

def test_func():
    print('test')
'''
def save_tif(image: np.ndarray, path: str, metadata):
        """
        Save a 5D image array as a TIFF file.

        Args:
            image (np.ndarray): The input 5D image array with shape (Time, Z, Channel, Y, X).
            path (str): The path to save the TIFF file.
            info (str): Additional information or metadata about the image which ImageJ can interpret for the 'Show Info' section.
            xy_cal (tuple): Tuple containing pixel spacing values for the X and Y dimensions. 
            config (tuple): Tuple containing configuration parameters such as pixel spacing in z, channel count, frames, and slices.
            ranges (tuple): Tuple containing range information for each channel.
            min_range (int): The maximum value across all channels' minimum values.
            max_range (int): The minimum value across all channels' maximum values.

        Returns:
            None

        Raises:
            None

        Output:
            Tif file in denoted path. 
        """

        # Check if the directory exists, if not, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        metadata = metadata.metadata

        tifffile.imwrite(path, 
                        image,
                        photometric='minisblack', 
                        imagej=True,  # Add ImageJ metadata
                        resolution=(metadata.x_resolution, metadata.y_resolution),
                        metadata={
                                "axes": "TZCYX",  # Specify the order of the axes
                                "spacing": metadata.slice_distances,  # Specify the pixel spacing
                                "unit": "micron",  # Specify the units of the pixel spacing
                                "hyperstack": "true",  # Specify that the data is a hyperstack
                                "mode": "color",  # Specify the display mode
                                "channels": metadata.channel_count,  # Specify the channel colors
                                "frames": metadata.time_count,  # Specify the number of frames
                                "Info": metadata.info_string,
                                "slices": metadata.z_size,  # Specify the number of slices
                                "Ranges": metadata.range_tuple,
                                'min': metadata.min_range, 
                                'max': metadata.max_range,
                                "metadata": "ImageJ=1.53c\n",  # Add a blank metadata field for ImageJ
                            })
'''
def save_tif(image: np.ndarray, path: str, metadata, colormap=None, photometric='minisblack', imagej=True):
        """
        Save a 5D image array as a TIFF file.

        Args:
            image (np.ndarray): The input 5D image array with shape (Time, Z, Channel, Y, X).
            path (str): The path to save the TIFF file.
            info (str): Additional information or metadata about the image which ImageJ can interpret for the 'Show Info' section.
            xy_cal (tuple): Tuple containing pixel spacing values for the X and Y dimensions. 
            config (tuple): Tuple containing configuration parameters such as pixel spacing in z, channel count, frames, and slices.
            ranges (tuple): Tuple containing range information for each channel.
            min_range (int): The maximum value across all channels' minimum values.
            max_range (int): The minimum value across all channels' maximum values.

        Returns:
            None

        Raises:
            None

        Output:
            Tif file in denoted path. 
        """

        # Check if the directory exists, if not, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))


        tifffile.imwrite(path, 
                        image,
                        photometric=photometric, 
                        imagej=imagej,  # Add ImageJ metadata
                        #colormap=colormap,
                        resolution=(metadata["x_resolution"], metadata["y_resolution"]),
                        metadata=metadata)        

'''
def save_tif(image: np.ndarray, path: str, info: str, xy_cal: tuple, config: tuple, ranges: tuple, min_range: int, max_range: int):
    """
    Save a 5D image array as a TIFF file.

    Args:
        image (np.ndarray): The input 5D image array with shape (Time, Z, Channel, Y, X).
        path (str): The path to save the TIFF file.
        info (str): Additional information or metadata about the image which ImageJ can interpret for the 'Show Info' section.
        xy_cal (tuple): Tuple containing pixel spacing values for the X and Y dimensions. 
        config (tuple): Tuple containing configuration parameters such as pixel spacing in z, channel count, frames, and slices.
        ranges (tuple): Tuple containing range information for each channel.
        min_range (int): The maximum value across all channels' minimum values.
        max_range (int): The minimum value across all channels' maximum values.

    Returns:
        None

    Raises:
        None

    Output:
        Tif file in denoted path. 
    """
    tifffile.imwrite(path, 
                    image,
                    photometric='minisblack', 
                    imagej=True,  # Add ImageJ metadata
                    resolution=xy_cal,
                    metadata={
                            "axes": "TZCYX",  # Specify the order of the axes
                            "spacing": config[0],  # Specify the pixel spacing
                            "unit": "micron",  # Specify the units of the pixel spacing
                            "hyperstack": "true",  # Specify that the data is a hyperstack
                            "mode": "color",  # Specify the display mode
                            "channels": config[1],  # Specify the channel colors
                            "frames": config[2],  # Specify the number of frames
                            "Info": info,
                            "slices": config[3],  # Specify the number of slices
                            "Ranges": ranges,
                            'min': min_range, 
                            'max': max_range,
                            "metadata": "ImageJ=1.53c\n",  # Add a blank metadata field for ImageJ
                        })

'''

def normalize_to_uint8(image_array):
    """
    Normalize the input image array to the [0, 255] range and convert to uint8.

    Args:
        image_array (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized and converted image array of dtype uint8.
    """
    if image_array.dtype == np.uint8:
        # Normalize the uint8 array to [0, 255] range
        normalized_image_array = (image_array - image_array.min()) * (255 / (image_array.max() - image_array.min()))
        return normalized_image_array.astype(np.uint8)
    elif image_array.dtype == np.uint16:
        # Normalize the uint16 array to [0, 255] range
        normalized_image_array = ((image_array - image_array.min()) /
                                  (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        return normalized_image_array
    elif image_array.dtype == np.float64:
        # Normalize the float64 array to [0, 255] range
        normalized_image_array = ((image_array - image_array.min()) /
                                  (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        return normalized_image_array
    else:
        raise ValueError(f"{image_array.dtype} is an unsupported input dtype. Supported dtypes: uint8, uint16, float64")

def save_as_png(image_array, directory, hist_equalization=False):

    """
    Save a NumPy array as a PNG image.

    This function takes a 2D NumPy array representing an image and saves it as a PNG image file.
    Optionally, it can perform histogram equalization on the input image array before saving.

    Parameters:
    image_array (np.ndarray): A 2D NumPy array representing the image.
    directory (str): The path and filename for saving the PNG image.
    hist_equalization (bool, optional): Whether to apply histogram equalization to the image.
        Defaults to False.

    Raises:
    ValueError: If the input array is not 2D or if some elements are outside the valid range (0-255).

    Notes:
    - The input array should be a 2D NumPy array containing pixel values within the valid range (0-255).
    - If hist_equalization is set to True, histogram equalization is applied before saving.

    Example:
    ```python
    image = np.array([[0, 255, 128], [64, 192, 32]], dtype=np.uint8)
    save_as_png(image, 'output.png', hist_equalization=True)
    ```

    This will save the image as 'output.png' with optional histogram equalization.
    """

    print(image_array.shape)
    # Force the array into a 2D array if any singleton dimensions are passed
    image_array = np.squeeze(image_array)
    print(image_array.shape)
    # Check if the resulting array elements are within valid range
    is_within_range = np.all((image_array >= 0) & (image_array <= 255))

    if not is_within_range:
        #raise ValueError("Some elements are outside the valid range (0-255).")
        print(f"Images are not 8-Bit (image_tools.save_as_png)")

    # Check if the resulting array is not 2D
    if image_array.ndim != 2:
        raise ValueError("The array passed to save_as_png is not a 2D image.")

    # Convert the normalized array to uint8
    image_array = normalize_to_uint8(image_array)

    # if hist_equalization == True:
    #     # Apply histogram equalization
    #     image_array = normalize_to_uint8(exposure.equalize_hist(image_array))

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image_array)

    # Save the Pillow object as a PNG file
    pil_image.save(directory)

    print(f"Blended image saved as PNG: {directory}")

def load_5D_tif(tiff: str) -> np.ndarray:
    """
    Loads a multi-dimensional TIFF File, parses the dimensions and adds singleton dimensions for missing dimensions to
    comply with downstrem requirments for 5D arrays.

    Parameters:
    tiff (str): The path to the input TIFF file.
        
    Returns:
    5D numpy array
    """
    all_axes = 'TZCYX'
    with tifffile.TiffFile(tiff) as tif:
        tiff_data = tifffile.imread(tiff)
        axes = tif.series[0].axes
        metadata = tif.imagej_metadata
        #ifd_tags = dict(tif.pages[0].tags)
        xres = str(tif.pages[0].tags['XResolution'])
        xres = xres.split("(")[1].split(")")[0].split(", ")
        xres = int(xres[0]) / int(xres[1])

        yres = str(tif.pages[0].tags['YResolution'])
        yres = yres.split("(")[1].split(")")[0].split(", ")
        yres = int(yres[0]) / int(yres[1])

        xy_cal = (xres, yres)

        missing_axes = []
        for i, dimension in enumerate(all_axes):
            if dimension not in axes:
                missing_axes.append(i)

        # Add singleton dimensions at missing indexes
        for missing_index in missing_axes:
            tiff_data = np.expand_dims(tiff_data, axis=missing_index)
        
        config = (metadata["spacing"], tiff_data.shape[2], tiff_data.shape[0], tiff_data.shape[1])

        return tiff_data, tif.imagej_metadata, xy_cal, config

def get_ranges(image_array: np.ndarray) -> tuple[tuple, int, int]:
    """
    Calculates the ranges of minimum and maximum values for each channel across its other dimensions.

    Args:
        image_array (np.ndarray): The input 5D image array with shape (Time, Z, Channel, Y, X).
                                  Note: A 5D array is necessary for the code to work correctly.

    Returns:
        tuple, int, int:
            - A tuple with the range of minimum and maximum values for each channel's data.
              The tuple format is (Channel 1 minimum, Channel 1 maximum, Channel 2 minimum, Channel 2 maximum, ...).
            - The maximum value across all channels' minimum values.
            - The minimum value across all channels' maximum values.

    Explanation:
        This function calculates the minimum and maximum values for each channel's data across its other dimensions
        (Time, Z, Y, X), and returns these values as a tuple. Additionally, it calculates the maximum value of all
        the minimum values from each channel, as well as the minimum value of all the maximum values from each channel.

    Raises:
        ValueError: If the input array is not a 5D array.
        
    Example:
        For a 5D image array with shape (10, 20, 3, 512, 512), the return value might be:
        ((chan1_min, chan1_max, chan2_min, chan2_max, chan3_min, chan3_max), global_max_of_mins, global_min_of_maxes)
    """

    # Get the shape of the data
    shape = image_array.shape

    # Checks if the array conforms to the 5D requirement
    if len(image_array.shape) != 5:
        raise ValueError("Input array must be a 5D image array (Time, Z, Channel, Y, X).")

    # Initialize lists to store the min and max values
    all_min_values = []
    all_max_values = []

    # Initialize an empty list to store the min-max tuples
    min_max_tuples = []

    # Iterate through each channel
    for channel in range(shape[2]):
        # Slice the data to get the current channel's data
        channel_data = image_array[:, :, channel, :, :]

        # Calculate the min and max values across all other dimensions
        min_values = float(np.min(channel_data))
        max_values = float(np.max(channel_data))

        # Append the min and max values to the respective lists
        all_min_values.append(min_values)
        all_max_values.append(max_values)
        min_max_tuples.append(min_values)
        min_max_tuples.append(max_values)

    # Calculate the maximum of all minimum values and the minimum of all maximum values
    global_max_of_mins = int(np.max(all_min_values))
    global_min_of_maxes = int(np.min(all_max_values))
    range_tuple = tuple(float("{:.1f}".format(value)) for value in min_max_tuples)
    return {'Ranges': range_tuple, 'min': global_max_of_mins, 'max': global_min_of_maxes}

'''
def get_ranges(image_array: np.ndarray) -> (tuple, int, int):
    """
    Calculates the ranges of minimum and maximum values for each channel across its other dimensions.

    Args:
        image_array (np.ndarray): The input 5D image array with shape (Time, Z, Channel, Y, X).
                                  Note: A 5D array is necessary for the code to work correctly.

    Returns:
        tuple, int, int:
            - A tuple with the range of minimum and maximum values for each channel's data.
              The tuple format is (Channel 1 minimum, Channel 1 maximum, Channel 2 minimum, Channel 2 maximum, ...).
            - The maximum value across all channels' minimum values.
            - The minimum value across all channels' maximum values.

    Explanation:
        This function calculates the minimum and maximum values for each channel's data across its other dimensions
        (Time, Z, Y, X), and returns these values as a tuple. Additionally, it calculates the maximum value of all
        the minimum values from each channel, as well as the minimum value of all the maximum values from each channel.

    Raises:
        ValueError: If the input array is not a 5D array.
        
    Example:
        For a 5D image array with shape (10, 20, 3, 512, 512), the return value might be:
        ((chan1_min, chan1_max, chan2_min, chan2_max, chan3_min, chan3_max), global_max_of_mins, global_min_of_maxes)
    """

    # Get the shape of the data
    shape = image_array.shape

    # Checks if the array conforms to the 5D requirement
    if len(image_array.shape) != 5:
        raise ValueError("Input array must be a 5D image array (Time, Z, Channel, Y, X).")

    # Initialize lists to store the min and max values
    all_min_values = []
    all_max_values = []

    # Initialize an empty list to store the min-max tuples
    min_max_tuples = []

    # Iterate through each channel
    for channel in range(shape[2]):
        # Slice the data to get the current channel's data
        channel_data = image_array[:, :, channel, :, :]

        # Calculate the min and max values across all other dimensions
        min_values = int(np.min(channel_data))
        max_values = int(np.max(channel_data))

        # Append the min and max values to the respective lists
        all_min_values.append(min_values)
        all_max_values.append(max_values)
        min_max_tuples.append(min_values)
        min_max_tuples.append(max_values)

    # Calculate the maximum of all minimum values and the minimum of all maximum values
    global_max_of_mins = np.max(all_min_values)
    global_min_of_maxes = np.min(all_max_values)
    range_tuple = tuple(min_max_tuples)

    return range_tuple, global_max_of_mins, global_min_of_maxes
'''

