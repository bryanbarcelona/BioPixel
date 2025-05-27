import os

import warnings

import numpy as np

import tifffile

from typing import Optional, List, Dict

from PIL import Image

from utils import io, image_tools

import glob

from image_tensors import ImageReader, TifImageReader

import os
import warnings
import numpy as np
from typing import Optional, Union, List, Tuple, Dict
from PIL import Image
from image_tensors import TifImageReader

class ProjectAndBlendTif:
    def __init__(self, tif_path: str):
        tiffile = TifImageReader(tif_path)
        #self._image_data = next(tiffile._process_image_and_metadata())[1]
        self._image_data = tiffile.image_data
        self._metadata = tiffile.metadata.metadata
        self._imagej_metadata = tiffile.metadata.imagej_compatible_metadata
        self._tif_path = tif_path
        self._image_name = os.path.splitext(os.path.basename(self._tif_path))[0]
        self._slices = self._image_data.shape[1]
        self._initialize_folders()
        self._resolution_x = self._metadata.x_resolution
        self._resolution_y = self._metadata.y_resolution

    @property
    def image_data(self):
        return self._image_data
    
    @property
    def image_area(self):
        tensor_shape = self._image_data.shape
        return tensor_shape[3] * tensor_shape[4]
        
    @property
    def metadata(self):
        return self._metadata

    @property
    def resolution_x(self):
        return self._resolution_x
    
    @property
    def resolution_y(self):
        return self._resolution_y

    def _initialize_folders(self):
        base_dir = os.path.dirname(self._tif_path)
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

    def tensor_modulation(self, image_data,
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

    def projection(self, provide_filename=False, channel=0, projection_type="max", filename=None):
        filename = f'{self._image_name}'

        three_d_tensor = self.tensor_modulation(self._image_data, selected_channels=(channel,))

        time_length = len(three_d_tensor)
        channel_length = len(three_d_tensor[0])
        z_length = len(three_d_tensor[0][0])
        max_projected_images: Dict = {}
        for time_index, time_slice in enumerate(three_d_tensor):
            for channel_index, channel_slice in enumerate(time_slice):
                for z_index, z_slice in enumerate(channel_slice):

                    image_name = filename
                    
                    # Add index to the filename only if there is more than one element
                    if time_length > 1:
                        image_name += f'_t{time_index}'
                    if channel_length > 1:
                        image_name += f'_c{channel_index}'
                    if z_length > 1:
                        image_name += f'_z{z_index}'
                    
                    if projection_type == "max":
                        max_project_array = np.max(z_slice, axis=0)
                        max_projected_images[image_name] = max_project_array
                    if projection_type == "average":
                        max_project_array = np.mean(z_slice, axis=0).astype(np.uint8)
                        max_projected_images[image_name] = max_project_array
        
        if provide_filename:
            return max_projected_images
        else:
            return max_project_array
        
    def max_project(self, provide_filename=False, channel=0):
        filename = f'{self._image_name}'

        three_d_tensor = self.tensor_modulation(self._image_data, selected_channels=(channel,))

        time_length = len(three_d_tensor)
        channel_length = len(three_d_tensor[0])
        z_length = len(three_d_tensor[0][0])
        max_projected_images: Dict = {}
        for time_index, time_slice in enumerate(three_d_tensor):
            for channel_index, channel_slice in enumerate(time_slice):
                for z_index, z_slice in enumerate(channel_slice):

                    image_name = filename
                    
                    # Add index to the filename only if there is more than one element
                    if time_length > 1:
                        image_name += f'_t{time_index}'
                    if channel_length > 1:
                        image_name += f'_c{channel_index}'
                    if z_length > 1:
                        image_name += f'_z{z_index}'
                    
                    max_project_array = np.max(z_slice, axis=0)
                    max_projected_images[image_name] = max_project_array
        
        if provide_filename:
            return max_projected_images
        else:
            return max_project_array

    def avg_project(self, provide_filename=False):
        filename = f'{self._image_name}'

        three_d_tensor = self.tensor_modulation(self._image_data, selected_channels=(0,))

        time_length = len(three_d_tensor)
        channel_length = len(three_d_tensor[0])
        z_length = len(three_d_tensor[0][0])
        max_projected_images: Dict = {}
        for time_index, time_slice in enumerate(three_d_tensor):
            for channel_index, channel_slice in enumerate(time_slice):
                for z_index, z_slice in enumerate(channel_slice):

                    image_name = filename
                    
                    # Add index to the filename only if there is more than one element
                    if time_length > 1:
                        image_name += f'_t{time_index}'
                    if channel_length > 1:
                        image_name += f'_c{channel_index}'
                    if z_length > 1:
                        image_name += f'_z{z_index}'
                    
                    max_project_array = np.mean(z_slice, axis=0).astype(np.uint8)
                    max_projected_images[image_name] = max_project_array
        
        if provide_filename:
            return max_projected_images
        else:
            return max_project_array

class Projection:
    def __init__(self, image_array: np.ndarray):
        self._image_data = image_array
        self._slices = self._image_data.shape[1]

    @property
    def image_data(self):
        return self._image_data
    
    @property
    def image_area(self):
        tensor_shape = self._image_data.shape
        return tensor_shape[3] * tensor_shape[4]
        
    def tensor_modulation(self, image_data,
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

    def projection(self, provide_filename=False, channel=0, projection_type="maximum", filename=None):

        tensor_shape = self._image_data.shape

        if len(tensor_shape) != 3:
            raise ValueError("Image data needs to be a 3D tensor.")
                    
        if projection_type == "maximum":
            projection_array = np.max(self._image_data, axis=0).astype(np.uint8)

        if projection_type == "average":
            projection_array = np.mean(self._image_data, axis=0).astype(np.uint8)

        return projection_array
    
class ImageProjector:
    """Multidimensional image processor with dimension control and intelligent output handling.
    
    Handles 2D-5D image data from TIFF files or numpy arrays, providing:
    - Flexible dimension selection (time, Z-slices, channels)
    - Multiple projection types (max, mean, sum)
    - Configurable output with automatic file naming
    - Metadata preservation for TIFF outputs
    
    Attributes:
        image_data (np.ndarray): Standardized 5D tensor in (T, Z, C, Y, X) format
        output_folder (str): Configured output directory path
        metadata (dict): Image metadata including resolution information
    """

    def __init__(self, input_data: Union[str, np.ndarray]) -> None:
        """Initialize image processor with file path or array.
        
        Args:
            input_data: TIFF file path or numpy array (2D-5D dimensions)
            
        Raises:
            ValueError: For invalid input types or unsupported array dimensions
        """
        self._projections = None
        self.metadata = {}
        self._original_shape = None
        self._tif_path = None
        self._base_name = "projection"

        if isinstance(input_data, str):
            self._init_from_tiff(input_data)
        elif isinstance(input_data, np.ndarray):
            self._init_from_array(input_data)
        else:
            raise ValueError("Input must be file path or numpy array")

        self._standardize_shape()
        self._initialize_folders()

    @property
    def image_data(self) -> np.ndarray:
        """Standardized 5D image tensor in (T, Z, C, Y, X) format."""
        return self._image_data

    @property
    def output_folder(self) -> str:
        """Output directory path for saved projections."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, path: str) -> None:
        """Set output directory, creating if necessary.
        
        Args:
            path: Absolute path or relative path from input file directory/CWD
        """
        if os.path.isabs(path):
            self._output_folder = path
        else:
            base_dir = os.path.dirname(self._tif_path) if self._tif_path else os.getcwd()
            self._output_folder = os.path.join(base_dir, path)
        os.makedirs(self._output_folder, exist_ok=True)

    def project(self,
              projection_type: str = "max",
              z_range: Union[List[int], Tuple[int], str] = "all",
              selected_z: Optional[List[int]] = None,
              channel_range: Union[List[int], Tuple[int], str] = "all",
              selected_channels: Optional[List[int]] = None,
              time_range: Union[List[int], Tuple[int], str] = "all",
              selected_time: Optional[List[int]] = None,
              separate_time: bool = True,
              separate_channels: bool = True,
              separate_z: bool = False) -> None:
        """Generate projections with full dimension control.
        
        Args:
            projection_type: Projection method - 'max', 'mean', or 'sum'
            z_range: Z indices as 'all', (start, end), or list
            selected_z: Explicit Z indices (overrides z_range)
            channel_range: Channel indices as 'all', (start, end), or list
            selected_channels: Explicit channel indices (overrides channel_range)
            time_range: Time indices as 'all', (start, end), or list 
            selected_time: Explicit time indices (overrides time_range)
            separate_time: Maintain time points as separate projections
            separate_channels: Maintain channels as separate projections
            separate_z: Maintain Z-slices as separate projections
        """
        processed = self._process_tensor(
            time_range=time_range,
            selected_time=selected_time,
            z_range=z_range,
            selected_z=selected_z,
            channel_range=channel_range,
            selected_channels=selected_channels,
            separate_time=separate_time,
            separate_channels=separate_channels,
            separate_z=separate_z
        )
        
        self._projections = {}
        proj_func = self._get_projection_function(projection_type)

        for t_idx, time_slice in enumerate(processed):
            for c_idx, channel_slice in enumerate(time_slice):
                for z_idx, z_slice in enumerate(channel_slice):
                    proj = proj_func(z_slice, axis=0)
                    dim_tags = self._build_dimension_tags(
                        t_idx, c_idx, z_idx,
                        num_time=len(processed),
                        num_channels=len(time_slice[0]),
                        num_z=len(channel_slice)
                    )
                    self._projections[f"{self._base_name}{dim_tags}"] = proj

    def save_projections(self, format: str = "tif", overwrite: bool = False, include_metadata: bool = True) -> List[str]:
        """Save generated projections to disk.
        
        Args:
            format: Output format - 'tif' or 'png'
            overwrite: Overwrite existing files with same name
            include_metadata: Whether to include metadata when saving TIFF files. Default is True.
            
        Returns:
            List of saved file paths
            
        Raises:
            ValueError: For unsupported file formats
        """
        if not self._projections:
            warnings.warn("No projections available - call project() first", UserWarning)
            return []

        saved_files = []
        for name, data in self._projections.items():
            if data.ndim != 2:
                warnings.warn(f"Skipping non-2D data for {name}", UserWarning)
                continue

            full_path = self._generate_filename(name, format, overwrite)
            self._save_image_data(data, full_path, format, include_metadata=include_metadata)
            saved_files.append(full_path)
        
        return saved_files

    # region #################### PRIVATE METHODS #############################
    
    def _init_from_tiff(self, tif_path: str) -> None:
        """Initialize from TIFF file using TifImageReader."""
        reader = TifImageReader(tif_path)
        self._image_data = reader.image_data
        self.metadata = {
            'x_res': reader.metadata.imagej_compatible_metadata["x_resolution"],
            'y_res': reader.metadata.imagej_compatible_metadata["y_resolution"],
            'imagej_meta': reader.metadata.imagej_compatible_metadata
        }
        self._tif_path = tif_path
        self._base_name = os.path.splitext(os.path.basename(tif_path))[0]

    def _init_from_array(self, array: np.ndarray) -> None:
        """Initialize from numpy array with automatic shape detection."""
        self._image_data = array
        self._original_shape = array.shape
        self.metadata = {'x_res': 1.0, 'y_res': 1.0}
        self._base_name = "array_data"

    def _standardize_shape(self) -> None:
        """Convert input array to 5D (T, Z, C, Y, X) format."""
        shape = self._image_data.shape
        dims = len(shape)
        
        if dims == 2:  # (Y, X)
            self._image_data = self._image_data[np.newaxis, np.newaxis, np.newaxis, :, :]
        elif dims == 3:  # (Z, Y, X)
            self._image_data = self._image_data[np.newaxis, :, np.newaxis, :, :]
        elif dims == 4:  # (C, Z, Y, X)
            self._image_data = self._image_data[np.newaxis, :, :, :, :]
        elif dims != 5:
            raise ValueError(f"Unsupported array shape: {shape}")

    def _initialize_folders(self) -> None:
        """Set default output directory based on input source."""
        if self._tif_path:
            self.output_folder = os.path.join(os.path.dirname(self._tif_path), 'projections')
        else:
            self.output_folder = os.path.join(os.getcwd(), 'projections')

    def _process_tensor(self, **dim_params) -> list:
        """Core tensor processing pipeline with dimension selection."""
        t_dim, z_dim, c_dim, _, _ = self._image_data.shape

        # Resolve indices for each dimension
        t_idx = self._get_dim_indices(t_dim, dim_params.get('time_range', "all"), 
                                     dim_params.get('selected_time'), "time")
        z_idx = self._get_dim_indices(z_dim, dim_params.get('z_range', "all"), 
                                     dim_params.get('selected_z'), "z")
        c_idx = self._get_dim_indices(c_dim, dim_params.get('channel_range', "all"), 
                                     dim_params.get('selected_channels'), "channel")

        # Advanced indexing and tensor slicing
        t_ix, z_ix, c_ix = np.ix_(t_idx, z_idx, c_idx)
        sliced_tensor = self._image_data[t_ix, z_ix, c_ix, :, :]

        return self._split_tensor_layers(
            sliced_tensor,
            dim_params.get('separate_time', True),
            dim_params.get('separate_channels', True),
            dim_params.get('separate_z', False)
        )

    def _get_dim_indices(self,
                        dim_size: int,
                        range_spec: Union[str, tuple],
                        selected: Optional[tuple],
                        dim_name: str) -> Tuple[int]:
        """Resolve dimension indices from user parameters."""
        if selected is not None:
            return self._validate_indices(selected, dim_size, dim_name)
        
        if isinstance(range_spec, str) and range_spec.lower() == "all":
            return tuple(range(dim_size))
        
        if isinstance(range_spec, (list, tuple)):
            if len(range_spec) == 2:
                start, end = sorted(range_spec)
                return tuple(range(start, end + 1))
            return self._validate_indices(range_spec, dim_size, dim_name)
        
        return tuple(range(dim_size))

    def _validate_indices(self,
                         indices: List[int],
                         dim_size: int,
                         dim_name: str) -> Tuple[int]:
        """Validate and clamp indices to valid range."""
        validated = []
        for idx in indices:
            adj_idx = max(0, min(idx, dim_size-1))
            if idx != adj_idx:
                warnings.warn(f"Clamping {dim_name} index {idx} to {adj_idx}", UserWarning)
            validated.append(adj_idx)
        return tuple(sorted(set(validated)))

    def _split_tensor_layers(self,
                            tensor: np.ndarray,
                            split_t: bool,
                            split_c: bool,
                            split_z: bool) -> list:
        """Split tensor according to dimension separation flags."""
        T, Z, C, H, W = tensor.shape
        
        # Process time dimension
        time_slices = [tensor[t] for t in range(T)]
        if not split_t:
            time_slices = [np.concatenate(time_slices, axis=0)]
            Z = time_slices[0].shape[0]

        # Process channels
        channel_slices = []
        for t_slice in time_slices:
            c_slices = [t_slice[:, c] for c in range(C)]
            if not split_c:
                c_slices = [np.concatenate(c_slices, axis=0)]
                Z = c_slices[0].shape[0]
            channel_slices.append(c_slices)

        # Process Z-slices
        final_slices = []
        for c_slice in channel_slices:
            z_slices = []
            for ch_slice in c_slice:
                z_stack = [np.expand_dims(ch_slice[z], 0) for z in range(Z)]
                if not split_z:
                    z_stack = [np.concatenate(z_stack, axis=0)]
                z_slices.append(z_stack)
            final_slices.append(z_slices)
        
        return final_slices

    def _get_projection_function(self, projection_type: str):
        """Resolve projection function from type string."""
        proj_map = {
            "max": np.max,
            "mean": np.mean,
            "sum": np.sum
        }
        pt = projection_type.lower()
        if pt not in proj_map:
            warnings.warn(f"Unknown projection type {pt}, using max", UserWarning)
        return proj_map.get(pt, np.max)

    def _build_dimension_tags(self,
                             t_idx: int,
                             c_idx: int,
                             z_idx: int,
                             num_time: int,
                             num_channels: int,
                             num_z: int) -> str:
        """Generate filename dimension tags when needed."""
        tags = []
        if num_time > 1:
            tags.append(f"_t{t_idx}")
        if num_channels > 1:
            tags.append(f"_c{c_idx}")
        if num_z > 1:
            tags.append(f"_z{z_idx}")
        return "".join(tags)

    def _generate_filename(self,
                          base_name: str,
                          format: str,
                          overwrite: bool) -> str:
        """Generate unique filename with conflict resolution."""
        base = os.path.join(self.output_folder, f"{base_name}.{format.lower()}")
        if overwrite or not os.path.exists(base):
            return base

        counter = 1
        while True:
            new_name = f"{base_name}_{counter:02d}.{format.lower()}"
            full_path = os.path.join(self.output_folder, new_name)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    def _save_image_data(self,
                        data: np.ndarray,
                        path: str,
                        format: str,
                        include_metadata: bool = True) -> None:
        """
        Save image data using appropriate writer.

        Args:
            data (np.ndarray): Image data to save.
            path (str): Path to save the image.
            format (str): File format ("tif" or "png").
            include_metadata (bool): Whether to include metadata when saving TIFF files. Default is True.
        """
        if format.lower() == "tif":
            if include_metadata:
                # Save with metadata
                tifffile.imwrite(path, data, metadata=self.metadata.get('imagej_meta', {}))
            else:
                # Save without metadata
                tifffile.imwrite(path, data)
        elif format.lower() == "png":
            Image.fromarray(data).save(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

if __name__ == "__main__":
    import os
    import glob
    import sys
    
    # Get input folder through terminal prompt
    input_folder = input("Enter path to folder containing TIFF files: ").strip()
    
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist!")
        sys.exit(1)

    output_base = os.path.join(input_folder, "channel_results")
    tiff_files = glob.glob(os.path.join(input_folder, "*.tif"))

    if not tiff_files:
        print("No TIFF files found in the specified folder!")
        sys.exit(1)

    for tiff_path in tiff_files:
        print(f"\n{'='*40}\nProcessing: {os.path.basename(tiff_path)}")
        processor = ImageProjector(tiff_path)
        
        # Get channel count from standardized tensor shape (T, Z, C, Y, X)
        num_channels = processor.image_data.shape[2]

        # 1. First Channel Processing
        processor.project(
            selected_channels=[0],  # First channel
            projection_type="max",
            z_range="all",
            time_range="all",
            separate_time=False,
            separate_z=False,
            separate_channels=False
        )
        processor.output_folder = os.path.join(output_base, "first_channel")
        saved = processor.save_projections(include_metadata=False)
        print(f"Saved first channel results to: {saved}")

        # 2. Last Channel Processing
        last_channel = num_channels - 1
        processor.project(
            selected_channels=[last_channel],  # Last channel
            projection_type="max",
            z_range="all",
            time_range="all",
            separate_time=False,
            separate_z=False,
            separate_channels=False
        )
        processor.output_folder = os.path.join(output_base, "last_channel")
        saved = processor.save_projections(include_metadata=False)
        print(f"Saved last channel results to: {saved}")

    print("\nProcessing complete! Results saved in:", output_base)