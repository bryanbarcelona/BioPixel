import os
import warnings
import numpy as np
import tifffile

from PIL import Image
#from skimage import exposure
#import cv2
#import matplotlib.pyplot as plt
from utils import io, image_tools
import glob

from image_tensors import ImageReader, TifImageReader

class ProjectAndBlendTif:
    def __init__(self, tif_path: str):
        self._tif_path = tif_path
        self._image_name = os.path.splitext(os.path.basename(self._tif_path))[0]
        self._image_data = TifImageReader(tif_path).image_data
        self._slices = self._image_data.shape[1]

    @property
    def image_data(self):
        return self._image_data

    def _initialize_folders(self):
        base_dir = os.path.dirname(self._image_path)
        self._output_folder = os.path.join(base_dir, 'PNG')

    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        if os.path.isabs(value) and ':' in value:
            self._output_folder = value
        else:
            self._output_folder = os.path.join(os.path.dirname(self._image_path), value)

    def _tensor_modulation(self, image_data,
                        z_range: list[int] | tuple[int] | str = "all",
                        selected_z: list[int] | tuple[int] = None,
                        separate_z_layers: bool = False,
                        channel_range: list[int] | tuple[int] | str = "all",
                        selected_channels: list[int] | tuple[int] = None,
                        separate_channels: bool = True,
                        time_range: list[int] | tuple[int] | str = "all",
                        selected_time: list[int] | tuple[int] = None,
                        separate_time_points: bool = True):

        t_dim, z_dim, c_dim, _ , _ = image_data.shape

        def _validate_indices(indices, dim_size, dim_name):
            validated_indices = []
            for idx in indices:
                if 0 <= idx < dim_size:
                    validated_indices.append(idx)
                else:
                    warnings.warn(
                        f"Index {idx} out of range for {dim_name} dimension. Adjusting to valid range.",
                        UserWarning
                    )
                    adjusted_idx = max(0, min(idx, dim_size - 1))
                    validated_indices.append(adjusted_idx)
            
            # Remove duplicates by converting to a set and then to a tuple
            validated_indices = tuple(set(validated_indices))
            return validated_indices
    
        def _get_indices(dim_size, range_val, selected_val, dim_name):
            # Highest priority: explicit indices provided by the user
            if selected_val is not None:
                return tuple(selected_val)
            
            # If 'all' is specified, use the full range of the dimension
            if isinstance(range_val, str) and range_val.lower() == "all":
                return tuple(range(dim_size))
            
            # If a start and end point are provided, create a range
            if isinstance(range_val, (list, tuple)) and len(range_val) == 2:
                start_idx, end_idx = range_val
                if start_idx > end_idx:
                    warnings.warn(
                        f"Start index {start_idx} is greater than end index {end_idx} for {dim_name} dimension. Swapping values.",
                        UserWarning
                    )
                    start_idx, end_idx = end_idx, start_idx
                return tuple(range(start_idx, end_idx + 1))
            
            # Warn if multiple indices are provided directly in range_val
            if isinstance(range_val, (list, tuple)):
                warnings.warn(
                    f"Providing multiple indices directly in {dim_name}_range is not the intended usage. "
                    f"Use selected_{dim_name} for this purpose.",
                    UserWarning
                )
                return tuple(range_val)
            
            # Default to the full range if no valid input is provided
            return tuple(range(dim_size))

        def _select_subtensor(tensor, time_indices, z_indices, channel_indices):
            """
            Selects a subtensor based on specified indices for time, z, and channel dimensions.
            
            Parameters:
            tensor (np.ndarray): Input 5D tensor with shape (T, Z, C, Y, X).
            time_indices (tuple): Indices for the time dimension.
            z_indices (tuple): Indices for the z dimension.
            channel_indices (tuple): Indices for the channel dimension.
            
            Returns:
            np.ndarray: The selected subtensor.
            """
            # Use np.ix_ to create an open mesh of indices
            t_ix, z_ix, c_ix = np.ix_(time_indices, z_indices, channel_indices)
            
            # Select the subtensor using advanced indexing
            selected_tensor = tensor[t_ix, z_ix, c_ix, :, :]
            
            return selected_tensor
        
        def _modulate_tensor(tensor, split_t=separate_time_points, split_c=separate_channels, split_z=separate_z_layers):
            """
            Modulates a 5D tensor based on user preferences for splitting or stacking.
            
            Parameters:
            tensor (np.ndarray): Input 5D tensor with shape (T, Z, C, Y, X).
            split_t (bool): Whether to split the time dimension.
            split_c (bool): Whether to split the channel dimension.
            split_z (bool): Whether to split the z dimension.
            
            Returns:
            list: Nested list structure containing 3D arrays as per the specified modulation.
            """
            # Get the shape of the input tensor
            T, Z, C, _, _X = tensor.shape
                       
            time_stacks = []
            for t in range(T):
                temp_time = tensor[t, :, :, :, :]
                time_stacks.append(temp_time)
            if split_t is False:
                time_stacks = [np.concatenate(time_stacks, axis=0)]
                Z = time_stacks[0].shape[0]
            print(len(time_stacks), type(time_stacks), time_stacks[0].shape)
            channel_stacks = []
            for time_stack in time_stacks:
                temp_channel_stack = []
                for channel in range(C):
                    temp_channel = time_stack[:, channel, :, :]
                    temp_channel_stack.append(temp_channel)

                if split_c is False:
                    temp_channel_stack = [np.concatenate(temp_channel_stack, axis=0)]
                    Z = temp_channel_stack[0].shape[0]
                channel_stacks.append(temp_channel_stack)
            print(len(channel_stacks), type(channel_stacks), channel_stacks[0][0].shape)  

            z_stacks  = []
            for timepoint in channel_stacks:
                temp_channel_stack = []
                for channel_stack in timepoint:
                    temp_z_stack = []
                    for z in range(Z):
                        temp_z = channel_stack[z, :, :]
                        temp_z = np.expand_dims(temp_z, axis=0)
                        temp_z_stack.append(temp_z)  

                    if split_z is False:
                        temp_z_stack = [np.concatenate(temp_z_stack, axis=0)]
                    temp_channel_stack.append(temp_z_stack)
                z_stacks.append(temp_channel_stack)

            return z_stacks
        
        # Get indices for each dimension
        z_indices = _get_indices(z_dim, z_range, selected_z, "z")
        channel_indices = _get_indices(c_dim, channel_range, selected_channels, "channel")
        time_indices = _get_indices(t_dim, time_range, selected_time, "time")

        # Validate indices for each dimension
        z_indices = _validate_indices(z_indices, z_dim, "z")
        channel_indices = _validate_indices(channel_indices, c_dim, "channel")
        time_indices = _validate_indices(time_indices, t_dim, "time")      

        sliced_tensor = _select_subtensor(image_data, time_indices, z_indices, channel_indices)

        tensor_tree = _modulate_tensor(sliced_tensor)

        return tensor_tree
        
    
    def max_project(self):
        filename = f'{self._image_name}'
        output_folder = self._output_folder
        image_data = self._image_data

        for stage in range(len(images)):
            tif_filepath = f'{output_folder}{os.path.sep}{filename} Series {stage+1}.tif'
            time_image_series = []
            for time in range(len(images[stage])):
                channel_series = []
                for channel in range(len(images[stage][time])):
                    #print(f"Channel: {channel+1}, Time: {time+1}, Stage: {stage+1} - {images[stage][time][channel]}")
                    stk_file = tifffile.imread(images[stage][time][channel])
                    if len(stk_file.shape) < 3:
                        stk_file = np.expand_dims(stk_file, axis=0)
                    channel_series.append(stk_file)
                channel_series = np.stack(channel_series, axis=0)
                time_image_series.append(channel_series)
            stage_image_series.append(time_image_series)
            time_image_series = np.stack(time_image_series, axis=0)
            time_image_series = np.transpose(time_image_series, (0, 2, 1, 3, 4))

            ranges = image_tools.get_ranges(time_image_series)
            self.metadata.update(**ranges)
            
            yield tif_filepath, time_image_series

    def max_projection(self):
        """
        Generate a maximum intensity z-projection for a specific channel within a given z-range.

        Args:
            image_array (np.ndarray): 5D image array (T, Z, C, Y, X) containing image data.
            channel (int): Index of the channel to process.
            z_range (tuple): Range of z-planes to include in the projection (first_plane, last_plane).
                The z_range is inclusive of the first_plane and last_plane values.

        Returns:
            np.ndarray: Adjusted image with a maximum intensity z-projection for the specified channel.
                Shape of the returned array: (T, 1, Y, X) 4D array.
                
        Raises:
            ValueError: If the input image_array is not a 5D array.

        Notes:
            The commented out code below provides an optional way of image enhancement
            using histogram stretching. Feel free to uncomment and modify it if needed.
            
            # Optional: Calculate the percentiles of the pixel values
            p2, p98 = np.percentile(max_projection, (2, 98))

            # Apply histogram stretching
            equalized_image = exposure.rescale_intensity(max_projection, in_range=(p2, p98))

        """
        image_tensor = self.get_image_data()
        print(f" image tensor is  {image_tensor.shape}")
        
        if image_tensor.ndim != 5:
            raise ValueError("Input image_array must be a 5D array (T, Z, C, Y, X).")
        else:
            print(image_tensor.shape)

        timepoints = range(image_tensor.shape[0])
        channels = range(image_tensor.shape[2])

        filename = "TEST"
        output_path = "C:\\Users\\bryan\\Desktop\\20240216 +TIPs PLA\\Publishables"
        for timepoint in timepoints:
            for channel in channels:
                output_filepath = f"{output_path}{os.path.sep}{filename}_Channel_{channel+1}_Timepoint_{timepoint+1}_MaxProjection.png"
                print(output_filepath)
                
                channel_stack = image_tensor[timepoint, :, channel, :, :]
                max_projection = channel_stack.astype(np.uint16)
                max_projection = np.max(max_projection, axis=0)

                image_tools.save_as_png(max_projection, output_filepath, hist_equalization=False)

    def average_projection(self):
        """
        Generate a maximum intensity z-projection for a specific channel within a given z-range.

        Args:
            image_array (np.ndarray): 5D image array (T, Z, C, Y, X) containing image data.
            channel (int): Index of the channel to process.
            z_range (tuple): Range of z-planes to include in the projection (first_plane, last_plane).
                The z_range is inclusive of the first_plane and last_plane values.

        Returns:
            np.ndarray: Adjusted image with a maximum intensity z-projection for the specified channel.
                Shape of the returned array: (T, 1, Y, X) 4D array.
                
        Raises:
            ValueError: If the input image_array is not a 5D array.

        Notes:
            The commented out code below provides an optional way of image enhancement
            using histogram stretching. Feel free to uncomment and modify it if needed.
            
            # Optional: Calculate the percentiles of the pixel values
            p2, p98 = np.percentile(max_projection, (2, 98))

            # Apply histogram stretching
            equalized_image = exposure.rescale_intensity(max_projection, in_range=(p2, p98))

        """
        image_tensor = self.get_image_data()
        
        if image_tensor.ndim != 5:
            raise ValueError("Input image_array must be a 5D array (T, Z, C, Y, X).")
        else:
            print(image_tensor.shape)

        timepoints = range(image_tensor.shape[0])
        channels = range(image_tensor.shape[2])

        filename = "TEST"
        output_path = "C:\\Users\\bryan\\Desktop\\20240216 +TIPs PLA\\Publishables"
        for timepoint in timepoints:
            for channel in channels:
                output_filepath = f"{output_path}{os.path.sep}{filename}_Channel_{channel+1}_Timepoint_{timepoint+1}_AverageProjection.png"
               
                channel_stack = image_tensor[timepoint, :, channel, :, :]
                average_projection = channel_stack.astype(np.uint16)
                average_projection = np.mean(average_projection, axis=0)

                yield average_projection
                #image_tools.save_as_png(average_projection, output_filepath, hist_equalization=False)        

    def multiply_blend(self):
        """
        Generate a multiply blend of channel projections from a 5D image array.

        Args:
            image_array (np.ndarray): 5D array containing channel projections (T, Z, C, Y, X).
                Z must be 1, as the blending is performed across channels.

        Returns:
            np.ndarray: Blended 2D image resulting from the multiply blend operation.
        
        Raises:
            ValueError: If the input image_array is not a 5D array.
            ValueError: If Z dimension is not equal to 1.
            ValueError: If there are fewer than 2 channels for blending.
        """
        
        blended_image = None
        for i, image in enumerate(self.average_projection()):

            if blended_image is None:
                print("Doing this")
                blended_image = np.ones_like(image)

            largest_value = np.amax(image)
            print(f"Largest value before normalization:", largest_value, image.dtype)
            # Normalize images to the range [0, 1] if needed
            if np.max(image) > 1:
                image = image / 255.0
            else:
                image = image
            largest_value = np.amax(image)
            print("Largest value in normalized image:", largest_value, image.dtype)
            print(image.shape)
            blended_image = blended_image * image
            largest_value = np.amax(blended_image)
            p2, p98 = np.percentile(blended_image, (2, 98))

            # Apply histogram stretching
            blended_image = exposure.rescale_intensity(blended_image, in_range=(p2, p98))
            print(f"Largest value in iteration {i}:", largest_value, blended_image.dtype)
            plt.imshow(blended_image, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')  # Turn off axis
            plt.title(f'Multiply Image')
            plt.show()

    def screen_blend(self):
        """
        Generate a multiply blend of channel projections from a 5D image array.

        Args:
            image_array (np.ndarray): 5D array containing channel projections (T, Z, C, Y, X).
                Z must be 1, as the blending is performed across channels.

        Returns:
            np.ndarray: Blended 2D image resulting from the multiply blend operation.
        
        Raises:
            ValueError: If the input image_array is not a 5D array.
            ValueError: If Z dimension is not equal to 1.
            ValueError: If there are fewer than 2 channels for blending.
        """
        
        blended_image = None
        for i, image in enumerate(self.average_projection()):

            if blended_image is None:
                print("Doing this")
                blended_image = image

            largest_value = np.amax(image)
            print(f"Largest value before normalization:", largest_value, image.dtype)
            # Normalize images to the range [0, 1] if needed
            if np.max(image) > 1:
                image = image / 255.0
            else:
                image = image
            largest_value = np.amax(image)
            print("Largest value in normalized image:", largest_value, image.dtype)
            print(image.shape)
            blended_image = 1 - (1 - blended_image) * (1 - image)
            largest_value = np.amax(blended_image)
            p2, p98 = np.percentile(blended_image, (2, 98))

            # Apply histogram stretching
            blended_image = exposure.rescale_intensity(blended_image, in_range=(p2, p98))
            print(f"Largest value in iteration {i}:", largest_value, blended_image.dtype)
            plt.imshow(blended_image, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')  # Turn off axis
            plt.title(f'Screen Image')
            plt.show()

    def separate_z_slices(self, start=0, end=None):
        
        if end is None:
            end = self._slices

        # Validate start and end
        if start < 0:
            start = 0
        if end > self._slices:
            end = self._slices
        if start > end:
            start = end - 1
        print(f"Slicing from {start} to {end}")
        print(list(range(start, end)))
        return slice(start, end)

# image_processors.py

def check_attributes(*attributes):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            for attribute in attributes:
                if not hasattr(self, attribute):
                    raise AttributeError(f"Attribute '{attribute}' is required but not found in the main class")
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

class BaseImageProcessor:
    pass

class DenoisingProcessor(BaseImageProcessor):

    @check_attributes
    def add_something(self):
        self.something = "KnockKnock"

    def denoisefdgdfg(self, array):
        print("Denoising image...")
        print(f"denoised {array.shape}")

class ZProjectionProcessor(BaseImageProcessor):
    #def __init__(self):
    #    self.zlahzlah = "zlahzlah"
    #    #self.array = array

    def z_projektion(self):
        print("Performing z-projection...")
        #print(f"projected {array.shape}")