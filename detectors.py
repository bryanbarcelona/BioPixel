import copy
import math
import random

from typing import Dict, Generator, List, Optional, Tuple, Any

import cv2
import numpy as np
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from cellpose import models
from data_structures import Podosome, PodosomeCellResult, Signal
from params import DetectionParams, ModelPaths
from utils.io import get_resource_path
from image_processor import Projection

mods = ModelPaths()
model_path = get_resource_path(mods.MODEL_MAP["macrophage"])

class CellPoseDetector:
    def __init__(self, image_array: np.array, model: str = "macrophage", detection_channel: int = 1, 
                 auxiliary_channel: int = 0, diameter: int = 10, do_3d: bool = False, 
                 flow_threshold: float = 0.9, cellprob_threshold: float = 0.0, 
                 filename: Optional[str] = None, output_folder: Optional[str] = None, 
                 gpu: bool = False, min_canvas_size: Optional[Tuple[int, int]] = None,
                 min_size=15) -> None:
        """
        Initialize the detector with the image and parameters.

        Args:
            image_array (np.array): The input image array.
            model (str): The model type to use for detection. Default is "macrophage".
            detection_channel (int): The primary channel for detection. Default is 1.
            auxiliary_channel (int): The auxiliary channel for detection. Default is 0.
            diameter (int): The diameter of the objects to detect. Default is 10.
            do_3d (bool): Whether to perform 3D detection. Default is False.
            flow_threshold (float): The flow threshold for segmentation. Default is 0.9.
            cellprob_threshold (float): The cell probability threshold for segmentation. Default is 0.0.
            filename (Optional[str]): The filename associated with the image. Default is None.
            output_folder (Optional[str]): The folder to save the output. Default is None.
            gpu (bool): Whether to use GPU for computation. Default is False.
            min_canvas_size (Optional[Tuple[int, int]]): The minimum canvas size (height, width) for detection.
                                                        If the image is smaller, it will be padded. Default is None.
        """
        # Store parameters in a dataclass using the current instance's attributes
        detection_params = locals()
        detection_params.pop('self')
        detection_params.pop('image_array')
        detection_params.pop('min_canvas_size')

        # Extract the specific channel tensor
        self._single_channel_tensor = self._get_tensor(image_array, detection_channel - 1)

        self._original_shape = self._single_channel_tensor.shape  # Store the original shape of the image
        self._min_canvas_size = min_canvas_size

        # Pad the image if necessary
        if self._min_canvas_size is not None:
            self._single_channel_tensor = self._pad_image_to_min_size(self._single_channel_tensor, self._min_canvas_size)

        self.params = DetectionParams(**detection_params)

    def _get_tensor(self, array, channel):
        """
        Extracts a specific channel tensor from the image data.

        Parameters:
        -----------
        array (np.array): The input image array.
        channel (int): The channel index to extract from the image data.

        Returns:
        --------
        numpy.ndarray
            The extracted tensor for the specified channel.

        Notes:
        ------
        - For 4D arrays (czyx), extracts the specified channel.
        - For 5D arrays (tczyx), extracts the specified channel from the first time point.
        - For all other dimensionalities, the array is returned as-is.
        """
        if array.ndim == 4:
            # 4D array: (c, z, y, x) - extract the specified channel
            return array[channel, :, :, :]
        elif array.ndim == 5:
            # 5D array: (t, c, z, y, x) - extract the specified channel from the first time point
            return array[0, channel, :, :, :]
        else:
            # For all other dimensionalities, return the array as-is
            return array

    @property
    def image(self):
        """
        Provides access to the processed image tensor (after channel extraction and optional padding).

        Returns:
        --------
        numpy.ndarray
            The processed image tensor.
        """
        return self._single_channel_tensor

    def _pad_image_to_min_size(self, image: np.array, min_canvas_size: Tuple[int, int]) -> np.array:
        """
        Pad the image to the minimum canvas size if it is smaller than the specified size.

        Args:
            image (np.array): The input image.
            min_canvas_size (Tuple[int, int]): The minimum canvas size (height, width).

        Returns:
            np.array: The padded image.
        """
        # Get the last two dimensions (height and width)
        height, width = image.shape[-2], image.shape[-1]
        min_height, min_width = min_canvas_size

        if height < min_height or width < min_width:
            # Calculate the amount of padding needed
            pad_height = (min_height - height) // 2
            pad_width = (min_width - width) // 2

            # Create padding configuration for the last two dimensions
            padding = [(0, 0)] * (image.ndim - 2)  # No padding for non-spatial dimensions
            padding += [(pad_height, pad_height), (pad_width, pad_width)]  # Padding for y and x dimensions

            # Pad the image symmetrically
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)
            return padded_image
        else:
            return image

    def _trim_to_original_size(self, image: np.array) -> np.array:
        """
        Trim the image back to its original size after detection.
        Only the last two dimensions (y and x) are trimmed.

        Args:
            image (np.array): The image to trim.

        Returns:
            np.array: The trimmed image.
        """
        if self._min_canvas_size is not None:
            original_height, original_width = self._original_shape[-2], self._original_shape[-1]
            height, width = image.shape[-2], image.shape[-1]

            # Calculate the padding that was added
            pad_height = (height - original_height) // 2
            pad_width = (width - original_width) // 2

            # Create a slice object to trim the last two dimensions
            trim_slice = [slice(None)] * (image.ndim - 2)  # Preserve all non-spatial dimensions
            trim_slice += [
                slice(pad_height, pad_height + original_height),  # Trim y dimension
                slice(pad_width, pad_width + original_width)      # Trim x dimension
            ]

            # Apply the slice to trim the image
            return image[tuple(trim_slice)]
        else:
            return image

    def detect(self, mask_dict=False, **detection_params):
        """
        Perform detection using the specified parameters.

        Args:
            **param_overrides (dict): Keyword arguments to override the default parameters.
                Possible keys include:
                - model (str): The model type to use for detection.
                - detection_channel (int): The primary channel for detection.
                - auxiliary_channel (int): The auxiliary channel for detection.
                - diameter (int): The diameter of the objects to detect.
                - do_3d (bool): Whether to perform 3D detection.
                - flow_threshold (float): The flow threshold for segmentation.
                - cellprob_threshold (float): The cell probability threshold for segmentation.
                - filename (Optional[str]): The filename associated with the image.
                - output_folder (Optional[str]): The folder to save the output.
                - gpu (bool): Whether to use GPU for computation.

        Returns:
            np.array: The detected masks.
        """
        params = copy.deepcopy(self.params)
        params.update_params(**detection_params)

        image = self._single_channel_tensor

        model = models.CellposeModel(gpu=params.gpu, pretrained_model=params.model_path)

        masks, flows, styles = model.eval(
            image,
            channels=[params.detection_channel, params.auxiliary_channel],
            diameter=params.diameter if params.diameter != 0 else model.diam_labels,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
            do_3D=params.do_3d,
            min_size=params.min_size,
        )

        masks = masks.astype(np.uint16)

        masks = self._trim_to_original_size(masks)

        return masks

class MacrophageDetector:
    def __init__(self, image_array, diameter=700, channel=1, only_center_mask=False):
        self._image_tensor = image_array
        self._detection_channel = channel
        self.diameter = diameter
        self.masks = self.detect()

        if only_center_mask:
            self.masks = self.reduce_to_centermost_mask(self.masks)

    def detect(self):
            
        image_3d = self._get_tensor(self._image_tensor, self._detection_channel)

        image_2d = self._find_slice_with_highest_intensity(image_3d)
    
        masks_2d = CellPoseDetector(image_2d, do_3d=False, diameter=self.diameter, flow_threshold=0.4,
                                       model="macrophage", detection_channel=1, cellprob_threshold=0.0, 
                                       gpu=True, min_canvas_size=(3000, 3000)).detect()

        return masks_2d

    @property
    def image(self):
        return self._single_channel_tensor

    def reduce_to_centermost_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Reduces the segmentation to just the centermost mask, or the closest to the center if the center is background.
        
        Parameters:
            segmentation (np.ndarray): A 2D or 3D labeled mask array.
        
        Returns:
            np.ndarray: A segmentation array with only the centermost mask, or the closest to the center.
        """
        if np.max(segmentation) == 0:
            return segmentation

        center = tuple(s // 2 for s in segmentation.shape)

        center_label = segmentation[center]
        if center_label != 0:
            return (segmentation == center_label).astype(np.uint16)

        mask_coords = np.argwhere(segmentation > 0)

        distances = np.linalg.norm(mask_coords - np.array(center), axis=1)

        closest_mask_idx = np.argmin(distances)
        closest_label = segmentation[tuple(mask_coords[closest_mask_idx])]

        return (segmentation == closest_label).astype(np.uint16)

    def blackout_non_mask_areas(self):

        unique_mask_ids = np.unique(self.masks)

        for mask_id in unique_mask_ids:
            if mask_id == 0:
                continue
            mask = self.masks == mask_id
            
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)

            masked_image = np.zeros_like(self._image_tensor)

            mask_broadcasted = np.broadcast_to(mask, self._image_tensor.shape[:-2] + mask.shape[-2:])

            masked_image[mask_broadcasted] = self._image_tensor[mask_broadcasted]

            sliced_masked_image = masked_image[
                ..., min_y:max_y+1, min_x:max_x+1
            ]

            yield sliced_masked_image
        
    def get_masks(self):

        image_3d = self._get_tensor(self._image_tensor, self._detection_channel)
        
        # Find the 2D slice with the highest intensity (assumed method)
        image_2d = self._find_slice_with_highest_intensity(image_3d)
        plt.imshow(image_2d)
        plt.title("Image with Random Colored Transparent Masks")
        plt.axis('off')  # Hide axes
        plt.show()        
        # Convert the 2D grayscale image to a 3-channel RGB image
        image_with_masks = np.dstack([image_2d] * 3)  # Convert to RGB (3 channels)
        plt.imshow(image_with_masks)
        plt.title("Image with Random Colored Transparent Ma2321sks")
        plt.axis('off')  # Hide axes
        plt.show()  
        # Create an alpha channel (transparency) initialized to 0 (fully transparent)
        alpha_channel = np.zeros(image_2d.shape, dtype=np.uint8)

        # Get all unique mask IDs from the `self.masks`
        unique_mask_ids = np.unique(self.masks)
        print(unique_mask_ids)
        # Loop through each mask ID
        for mask_id in unique_mask_ids:
            if mask_id == 0:  # Skip background
                continue
            
            # Create a binary mask for the current mask ID
            mask = self.masks == mask_id
            
            # Generate a random color (R, G, B) for the mask
            random_color = [random.randint(0, 255) for _ in range(3)]  # Random color in RGB
            
            # Apply the blending to the mask region (similar to the "fill" mode in your snippet)
            alpha = 0.5  # Transparency level (0.0 = fully transparent, 1.0 = fully opaque)

            # Create an overlay for the mask area
            overlay = image_with_masks.copy()
            overlay[mask == 1] = (alpha * np.array(random_color) + (1 - alpha) * overlay[mask == 1])

            # Update the original image with the overlayed mask
            image_with_masks = overlay

            # Set the alpha channel (transparency) to a non-zero value for the mask area
            alpha_channel[mask] = 128  # Set transparency for mask region (semi-transparent)

        plt.imshow(image_with_masks)
        plt.title("Image with Random Colored Transparent Masks")
        plt.axis('off')  # Hide axes
        plt.show()

        # Stack the RGB image with the alpha channel to form an RGBA image
        image_with_masks_rgba = np.dstack([image_with_masks, alpha_channel])

        # Ensure the image is properly in RGBA format before displaying
        image_with_masks_rgba = image_with_masks_rgba.astype(np.uint8)

    def _get_tensor(self, array, channel):
        """
        Extracts a specific channel tensor from the image data.

        Parameters:
        -----------
        channel : int
            The channel index to extract from the image data.

        Returns:
        --------
        numpy.ndarray
            The extracted tensor for the specified channel.

        Notes:
        ------
        - Handles both 5D and 4D image data shapes.
        - Returns a 3D tensor for the specified channel.
        """
        if len(array.shape) == 5:
            image_tensor = array[0, :, channel, :, :]
        elif len(array.shape) == 4:
            image_tensor = array[:, channel, :, :]
        
        return image_tensor        

    def _find_slice_with_highest_intensity(self, z_stack):
        """
        Finds the 2D array (slice) in the z-stack with the highest overall intensity.

        Parameters:
        z_stack (numpy.ndarray): A 3D numpy array representing the z-stack of 2D arrays.

        Returns:
        numpy.ndarray: The 2D array (slice) with the highest overall intensity.
        """
        # Calculate the sum of intensities for each slice
        intensity_sums = np.sum(z_stack, axis=(1, 2))
        
        # Find the index of the slice with the highest intensity sum
        max_intensity_index = np.argmax(intensity_sums)

        # Return the slice with the highest intensity
        return z_stack[max_intensity_index]
 
class PodosomeManager:
    """
    Internal class to manage Podosome objects and perform all processing during initialization.
    """
    def __init__(self, mask_3d: np.ndarray, size_um: float = 1, 
                 resolution_xy: float = 5, resolution_z: float = 1, 
                 iterations: int = 1, cell_id: Optional[int] = None):
        """
        Args:
            mask_3d: 3D segmentation mask (Z,Y,X)
            cell_id: Optional identifier for future time-series tracking
        """
        self.mask_3d = mask_3d
        self.cell_id = cell_id or 0  # Default ID
        
        # Initialize processing tools
        self.kernel_xy, self.kernel_z = self._generate_kernels(size_um, resolution_xy, resolution_z)
        self.kernel_offsets_xy = np.argwhere(self.kernel_xy == 1) - (self.kernel_xy.shape[0] // 2, self.kernel_xy.shape[1] // 2)
        self.kernel_offsets_z = np.argwhere(self.kernel_z == 1) - (self.kernel_z.shape[0] // 2, self.kernel_z.shape[1] // 2)

        # Process immediately (or can separate into load()/process() steps)
        self.cell_result = self.process()

    def process(self) -> PodosomeCellResult:
        """Main pipeline: returns a fully processed PodosomeCell"""
        # Temporary storage during processing
        podosomes = {}
        
        # Existing processing logic (modified slightly)
        unique_ids = np.unique(self.mask_3d)
        for podosome_id in unique_ids[unique_ids != 0]:
            #if podosome_id % 10 == 0:
            print(f"Processing podosome ID: {podosome_id}")
            binary_mask = (self.mask_3d == podosome_id).astype(np.uint8)
            dilated_mask = self._process_single_podosome(binary_mask)
            podosomes[podosome_id] = self._create_podosome(podosome_id, binary_mask, dilated_mask)

        # Create the cell container
        cell = PodosomeCellResult(
            cell_id=self.cell_id,
            podosomes=podosomes,
            label_map=self.mask_3d,
        )
        
        # Calculate relative volumes across this cell's podosomes
        self._calculate_relative_volumes(cell)
        
        return cell

    # ----------- Core Processing Methods (Modified) -----------
    def _process_single_podosome(self, binary_mask: np.ndarray) -> np.ndarray:
        """Process one podosome mask through dilation pipeline"""
        cropped_mask, coords = self._extract_mask_with_margin(binary_mask, 30)
        dilated_mask = self._create_dilated_mask_3d(cropped_mask)
        return self._reinsert_dilated_mask(binary_mask.shape, dilated_mask, coords)

    def _create_podosome(self, id: int, binary_mask: np.ndarray, dilated_mask: np.ndarray) -> Podosome:
        podosome = Podosome(id=id)
        
        for z in range(binary_mask.shape[0]):
            # Process ORIGINAL mask (undilated)
            original_slice = (binary_mask[z] * 255).astype(np.uint8)
            original_contours, _ = cv2.findContours(original_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            original_area = cv2.contourArea(original_contours[0]) if original_contours else 0
            y_orig, x_orig = np.where(original_slice > 0)
            podosome._add_slice_data(
                z, 
                contour=original_contours[0] if original_contours else None,
                pixels=list(zip(y_orig, x_orig)),
                area=original_area,
                dilated=False
            )

            # Process DILATED mask (with identical null-safety)
            dilated_slice = (dilated_mask[z] * 255).astype(np.uint8)
            dilated_contours, _ = cv2.findContours(dilated_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dilated_area = cv2.contourArea(dilated_contours[0]) if dilated_contours else 0
            y_dil, x_dil = np.where(dilated_slice > 0)
            podosome._add_slice_data(
                z,
                contour=dilated_contours[0] if dilated_contours else None,
                pixels=list(zip(y_dil, x_dil)),
                area=dilated_area,
                dilated=True
            )

        # Calculate properties if any slices exist
        if podosome.slices:
            podosome._calculate_centroid()
            podosome._calculate_bounding_box(dilated=False)
            podosome._calculate_bounding_box(dilated=True)
        
        return podosome

    # ----------- Analysis Methods -----------
    def _calculate_relative_volumes(self, cell: PodosomeCellResult) -> None:
        """Normalize volumes across podosomes in a cell"""
        volumes = [p.volume for p in cell.podosomes.values()]
        if not volumes:
            return
            
        min_vol, max_vol = min(volumes), max(volumes)
        for p in cell.podosomes.values():
            p.relative_volume = (p.volume - min_vol) / (max_vol - min_vol) if max_vol != min_vol else 1.0

    # ----------- Time-Series Ready Methods -----------
    def get_centroids(self) -> Dict[int, Tuple[float, float, float]]:
        """Returns {podosome_id: centroid} for tracking"""
        return {p.id: p.centroid for p in self.cell_result.podosomes.values()}
    
    def get_podosomes(self) -> Dict[int, Podosome]:
        """
        Return the dictionary of Podosome objects.
        """
        return self.cell_result.podosomes

    @property
    def masks(self) -> np.ndarray:
        """
        Get the dilated masks for all podosomes.
        """
        return np.where(self.masks_3d > 0, 255, 0).astype(np.uint8)
    
    @property
    def dilated_masks(self) -> np.ndarray:
        """
        Get the dilated masks for all podosomes.
        """
        return self._create_dilated_mask_3d(self.mask_3d, size_um=1, resolution_xy=5, resolution_z=1, dilation_iterations=1)
    
    @property
    def label_map(self) -> np.ndarray:
        """
        Get the label map for all podosomes.
        """
        return self._create_label_map_from_dilated_masks()
    
    @property
    def topographic_map(self) -> np.ndarray:
        """
        Get the topographic map for all podosomes.
        """
        return self._generate_podosome_topographic()
    
    def _generate_kernels(self, size_um: float, resolution_xy: float, resolution_z: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate elliptical kernels for XY and Z dilation based on physical size and resolution.
        """
        # Calculate kernel sizes in pixels (convert from μm to pixels)
        kernel_radius_xy = int(round(size_um * resolution_xy))
        kernel_radius_z = int(round(size_um * resolution_z))
        
        # Ensure kernel sizes are odd
        kernel_size_xy = 2 * kernel_radius_xy + 1
        kernel_size_z = 2 * kernel_radius_z + 1
        
        # Create circular kernels for XY and Z dilation
        kernel_xy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_xy, kernel_size_xy))
        kernel_z = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_z, kernel_size_z))
        
        return kernel_xy, kernel_z

    def _expand_mask_cv2(self, input_mask, scale_factor=1.2):
        """
        Expands the mask by scaling the contour of the shape in the binary mask.
        
        Parameters:
        - input_mask: 2D array (binary mask with 1 for the mask, 0 for background)
        - scale_factor: Factor by which to expand the shape (default is 1.2)
        
        Returns:
        - expanded_mask: 2D array of the expanded mask
        """
        # Find contours in the binary mask
        input_mask = (input_mask / 255).astype(np.uint8)

        contours, _ = cv2.findContours(input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # plt.imshow(input_mask, cmap='gray')  # Use 'gray' colormap for grayscale images
        # plt.axis('off')  # Hide axes
        # plt.show()

        if len(contours) == 0:
            return np.zeros_like(input_mask)
        
        # Get the largest contour (assuming there is only one object in the mask)
        contour = contours[0]
        
        # Approximate the contour to simplify its shape (optional, depending on the shape complexity)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calculate the center of the contour (center of mass)
        moments = cv2.moments(approx_contour)

        if moments["m00"] == 0:
            # Optionally, return an empty mask or handle this case as needed
            return np.zeros_like(input_mask)
    
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        # Scale the contour points around the center
        expanded_contour = np.array([[(point[0][0] - center_x) * scale_factor + center_x,
                                    (point[0][1] - center_y) * scale_factor + center_y]
                                    for point in approx_contour])
        
        # Create an empty mask for the expanded shape
        expanded_mask = np.zeros_like(input_mask)

        if not expanded_mask.flags['C_CONTIGUOUS']:
            expanded_mask = np.ascontiguousarray(expanded_mask)

        # Draw the expanded contour onto the expanded mask
        cv2.drawContours(expanded_mask, [expanded_contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        #cv2.fillPoly(expanded_mask, [expanded_contour.astype(np.int32)], 255)
        
        return expanded_mask
        
    def _create_dilated_mask_3d(self, segmentation_3d, size_um=1, resolution_xy=5, resolution_z=1, dilation_iterations=1):
        """
        Converts a 3D array of segmented masks into a fully dilated binary mask using OpenCV,
        considering anisotropic resolution and dilation in all three spatial dimensions.
        
        Parameters:
            segmentation_3d (numpy.ndarray): A 3D array where each z-layer contains 
                                            segmented masks with unique values 
                                            for segmentation. Background is assumed to be 0.
            size_um (float): Desired physical kernel size in micrometers.
            resolution_xy (float): Resolution of the image in pixels per micrometer in XY.
            resolution_z (float): Resolution of the image in pixels per micrometer in Z.
            dilation_iterations (int): Number of dilation iterations to apply.
        
        Returns:
            numpy.ndarray: A fully dilated binary 3D mask of the same shape as segmentation_3d.
        """
        binary_mask = np.where(segmentation_3d > 0, 255, 0).astype(np.uint8)
        
        dilated_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        
        for z in range(binary_mask.shape[0]):
            #dilated_mask[z] = cv2.dilate(binary_mask[z], self.kernel_xy, iterations=dilation_iterations)
            dilated_mask[z] = self._expand_mask_cv2(binary_mask[z], scale_factor=2.0)
        dilated_mask_yz = np.transpose(dilated_mask, (1, 0, 2))
        for y in range(dilated_mask_yz.shape[0]):
            dilated_mask_yz[y] = cv2.dilate(dilated_mask_yz[y], self.kernel_z, iterations=dilation_iterations)
        dilated_mask_yz = np.transpose(dilated_mask_yz, (1, 0, 2))
        
        dilated_mask_xz = np.transpose(dilated_mask, (2, 1, 0))
        for x in range(dilated_mask_xz.shape[0]):
            dilated_mask_xz[x] = cv2.dilate(dilated_mask_xz[x], self.kernel_z, iterations=dilation_iterations)

        dilated_mask_xz = np.transpose(dilated_mask_xz, (2, 1, 0))
        
        final_dilated_mask = np.maximum.reduce([dilated_mask, dilated_mask_yz, dilated_mask_xz])
        
        return final_dilated_mask

    def _extract_mask_with_margin(self, mask, kernel_size):
        """
        Extracts a 3D mask with a safe margin for dilation along X and Y.
        The entire Z range is considered.

        Parameters:
            mask (numpy.ndarray): 3D binary mask (z, y, x)
            kernel_size (int): Size of the elliptical structuring element used for dilation
        
        Returns:
            cropped_mask (numpy.ndarray): Extracted mask with margin (same Z size as input)
            start_coords (tuple): (min_y, min_x) coordinates for reinsertion
        """
        non_zero_y = np.any(mask, axis=(0, 2))
        non_zero_x = np.any(mask, axis=(0, 1))

        if not non_zero_y.any() or not non_zero_x.any():
            return None, None

        min_y, max_y = np.where(non_zero_y)[0][[0, -1]]
        min_x, max_x = np.where(non_zero_x)[0][[0, -1]]

        margin = kernel_size // 2  

        _, shape_y, shape_x = mask.shape
        min_y, max_y = max(min_y - margin, 0), min(max_y + margin, shape_y - 1)
        min_x, max_x = max(min_x - margin, 0), min(max_x + margin, shape_x - 1)

        cropped_mask = mask[:, min_y:max_y+1, min_x:max_x+1]

        return cropped_mask, (min_y, min_x)

    def _reinsert_dilated_mask(self, original_shape, dilated_mask, start_coords):
        """
        Reinserts a dilated mask into its original 3D position across the full Z range.

        Parameters:
            original_shape (tuple): Shape of the original 3D volume (z, y, x)
            dilated_mask (numpy.ndarray): The dilated cropped mask (same Z size as original)
            start_coords (tuple): (min_y, min_x) - The original start position of the cropped mask

        Returns:
            numpy.ndarray: A new array of the same shape as original, with the dilated mask inserted
        """
        # Create an empty mask with the same shape as the original
        reinserted_mask = np.zeros(original_shape, dtype=dilated_mask.dtype)

        # Extract starting coordinates and size of dilated mask
        min_y, min_x = start_coords
        _, dy, dx = dilated_mask.shape  # Keep full Z range

        # Reinsert into the corresponding (Y, X) position across all Z slices
        reinserted_mask[:, min_y:min_y+dy, min_x:min_x+dx] = dilated_mask

        return reinserted_mask
    
    def filter_podosomes(self, distance_threshold=5.0, volume_threshold=15.0):
        """
        Filters podosomes based on nearest-neighbor distance and volume.
        Keeps only podosomes where:
            - Distance to nearest neighbor > distance_threshold
            - OR Volume >= volume_threshold
        Updates self.podosomes with the filtered subset.
        
        Args:
            distance_threshold (float): Maximum allowed distance for filtering.
            volume_threshold (float): Minimum volume required to keep a podosome.
        """
        podosome_ids = list(self.podosomes.keys())

        if len(podosome_ids) == 0:
            return
        centroids = np.array([self.podosomes[i].centroid for i in podosome_ids], dtype=np.float64)
        volumes = np.array([self.podosomes[i].volume for i in podosome_ids], dtype=np.float64)

        if len(podosome_ids) > 1:
            diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=-1))

            np.fill_diagonal(distances, np.inf)
            nearest_distances = np.min(distances, axis=1)
        else:
            nearest_distances = np.inf

        keep_mask = (nearest_distances > distance_threshold) | (volumes >= volume_threshold)

        self.podosomes = {pid: self.podosomes[pid] for i, pid in enumerate(podosome_ids) if keep_mask[i]}

    def _determine_relative_podosome_volume(self, podosomes: dict):
        """
        Assigns a color to each podosome based on its volume, mapping min volume to green and max to red.
        
        Parameters:
            podosomes (dict): Dictionary of Podosome objects.

        Returns:
            None (modifies podosomes in-place)
        """
        volumes = np.array([p.volume for p in podosomes.values()])
        
        if len(volumes) == 0:
            return
        
        min_vol, max_vol = volumes.min(), volumes.max()

        if min_vol == max_vol:
            for podosome in podosomes.values():
                podosome.relative_volume = 1.0
            return

        for podosome in podosomes.values():
            podosome.relative_volume = float((podosome.volume - min_vol) / (max_vol - min_vol))

    def _create_label_map_from_dilated_masks(self) -> np.ndarray:
        """
        Create a 3D label map by combining the dilated podosome masks.

        Returns:
        np.ndarray: A 3D label map where each voxel contains a tuple of podosome IDs.
        """
        # Initialize an empty 3D array for the label map with the same shape as masks_3d
        label_map = np.empty_like(self.mask_3d, dtype=object)

        # Iterate over all podosomes and dilated slices
        for podosome_id, podosome in self.get_podosomes().items():
            for z, data in podosome.dilated_slices.items():
                pixels = data.get('pixels', [])  # Get the 'pixels' list, default to empty if not present
                for (y, x) in pixels:
                    if label_map[z, y, x] is None:
                        # Assign the podosome ID as a tuple if this voxel was previously empty
                        label_map[z, y, x] = (podosome_id,)
                    else:
                        # Add the podosome ID to the existing tuple
                        label_map[z, y, x] = tuple(sorted(set(label_map[z, y, x] + (podosome_id,))))

        return label_map

    def _extract_slices_with_masks(self):
        """
        Creates a copy of the podosome dictionary, keeping only slices that contain 'pixels' 
        in the 'slices' and 'dilated_slices' attributes.

        Returns:
            dict: A new podosome dictionary with slices that contain pixel masks.
        """
        podosomes_with_masks = {}

        for podosome_id, podosome in self.get_podosomes().items():

            masked_podosome = copy.deepcopy(podosome)

            # Keep only slices that contain 'pixels'
            masked_podosome.slices = {z: data for z, data in podosome.slices.items() if data.get("pixels")}
            masked_podosome.dilated_slices = {z: data for z, data in podosome.dilated_slices.items() if data.get("pixels")}

            # Add the deep-copied podosome with its filtered slices to the dictionary
            podosomes_with_masks[podosome_id] = masked_podosome

        return podosomes_with_masks

    def _generate_podosome_topographic(self):
        """
        Generates a 3D heatmap where each voxel represents the relative height of a podosome.
        The relative height is based on the original podosome mask, with dilated areas above and below
        assigned values greater than 1.0 or less than 0.0, respectively.

        Parameters:
            original_shape (tuple): Shape of the 3D volume (z, y, x)
            podosome_dict (dict): Dict of podosome objects {id: podosome_object}

        Returns:
            numpy.ndarray: The final 3D heatmap.
        """
        topography_map = np.zeros_like(self.mask_3d, dtype=np.float32)
        count_map = np.zeros_like(self.mask_3d, dtype=np.uint8)

        slice_reduced_podosomes = self._extract_slices_with_masks()

        for podosome in slice_reduced_podosomes.values():
            original_z_layers = sorted(podosome.slices.keys())
            num_layers = len(original_z_layers)

            if num_layers == 0:
                continue

            podosome.relative_height_map = {}
            for i, z in enumerate(original_z_layers):
                podosome.relative_height_map[z] = (i + 1) / num_layers

        for podosome in slice_reduced_podosomes.values():
            slices = podosome.dilated_slices
            z_layers = sorted(slices.keys())

            if not hasattr(podosome, "relative_height_map"):
                continue

            original_z_layers = sorted(podosome.slices.keys())
            if not original_z_layers:
                continue

            min_z, max_z = original_z_layers[0], original_z_layers[-1]
            num_layers = len(original_z_layers)

            for z in z_layers:
                if z in podosome.relative_height_map:
                    relative_height = podosome.relative_height_map[z]
                elif z > max_z:
                    relative_height = 1.0 + (z - max_z) / num_layers
                elif z < min_z:
                    relative_height = -((min_z - z) / num_layers)
                else:
                    relative_height = podosome.relative_height_map[min_z]

                for (y, x) in slices[z]["pixels"]:
                    topography_map[z, y, x] += relative_height
                    count_map[z, y, x] += 1

        dilated_mask = count_map > 0
        topography_map[dilated_mask] /= count_map[dilated_mask]

        return topography_map

class PodosomeDetector:
    def __init__(self, image_array, diameter=10, channel=-1):
        self._single_channel_tensor = self._get_tensor(image_array, channel)
        self.diameter = diameter
        #self.masks = None
        self._manager: Optional[PodosomeManager] = None

    def detect(self, flow_threshold_3d=0.4, cellprob_threshold_3d=0.0):

        def _taper_out_masks(mask, max_dilation_steps=10, bottom_limit=0.1):

            # Convert mask areas to ones
            mask = mask.astype(bool).astype(float)
            
            # Calculate the values to be distributed
            values = np.geomspace(1, bottom_limit, max_dilation_steps + 1)
            
            # Initialize the tapered mask
            tapered_mask = np.copy(mask)
            
            # Create a kernel for dilation
            kernel = np.ones((3, 3), np.uint8)
            
            # Iteratively dilate the mask and apply the decreased values
            for step in range(1, max_dilation_steps + 1):
                # Dilate the mask
                dilated_mask = cv2.dilate(tapered_mask, kernel, iterations=1)
                
                # Remove the previous dilated mask from the current dilated mask
                new_mask = np.where(tapered_mask > 0, 0, dilated_mask)
                
                # Apply the current decreased value
                new_mask = np.where(new_mask > 0, values[step], new_mask)
                
                # Update the tapered mask
                tapered_mask = np.maximum(tapered_mask, new_mask)
            
            # Ensure the bottom limit is applied to all pixels outside the original mask
            tapered_mask = np.where(tapered_mask == 0, 0, tapered_mask)
            
            return tapered_mask

        def _dim_the_lights_on_3D(image_3d, tapered_array, adjust_mean=True):
            """
            Adjusts the brightness of a 3D image using a tapered mask.
            
            Parameters:
                image_3d (np.ndarray): 3D array representing image slices.
                tapered_array (np.ndarray): 2D mask to be applied to each slice.
                adjust_mean (bool): If True, scales intensity based on max mean slice; otherwise, applies mask as is.

            Returns:
                np.ndarray: The adjusted 3D image.
            """
            intensity_mean = np.mean(image_3d, axis=(1, 2))
            
            if adjust_mean:
                intensity_mean_index = np.max(intensity_mean)
                intensity_mean = intensity_mean / intensity_mean_index
            else:
                intensity_mean = np.ones_like(intensity_mean)  # Keep scaling uniform
            
            original_min = np.min(tapered_array)
            original_max = np.max(tapered_array)
            
            assert intensity_mean.shape[0] == image_3d.shape[0]
            
            output_array = np.zeros_like(image_3d)
            
            for z in range(image_3d.shape[0]):
                new_max = intensity_mean[z]
                mask = np.interp(tapered_array, [original_min, original_max], [original_min, new_max])
                output_array[z] = image_3d[z] * mask
            
            return output_array
            
        image_3d = self._single_channel_tensor

        image_2d = self._find_slice_with_highest_intensity(image_3d)

        image_2d = self._equalize_image_clahe(image_2d, normalize_output=True)

        masks_2d = CellPoseDetector(image_2d, do_3d=False, diameter=10, 
                                       model="podosomes", detection_channel=1, cellprob_threshold=-3.0, 
                                       gpu=True).detect()
        
        tapered_mask = _taper_out_masks(masks_2d)

        image_3d_normalized = _dim_the_lights_on_3D(image_3d, tapered_mask, adjust_mean=False)

        masks_3d = CellPoseDetector(image_3d_normalized, do_3d=True, diameter=10, 
                                      model="podosomes", detection_channel=1, 
                                      gpu=True, flow_threshold=flow_threshold_3d, cellprob_threshold=cellprob_threshold_3d).detect()
                            
        return masks_3d
    
    @property
    def podosomes(self) -> Dict[int, Podosome]:
        if self._manager is None:
            raise ValueError("No masks have been detected yet. Call `detect` first.")
        return self._manager.get_podosomes()
    
    @property
    def masks(self) -> np.ndarray:
        if self._manager is None:
            raise ValueError("No masks have been detected yet. Call `detect` first.")
        return self._manager.masks
    
    @property
    def dilated_masks(self) -> np.ndarray:
        if self._manager is None:
            raise ValueError("No masks have been detected yet. Call `detect` first.")
        return self._manager.dilated_masks
    
    @property
    def label_map(self) -> np.ndarray:
        if self._manager is None:
            raise ValueError("No masks have been detected yet. Call `detect` first.")
        return self._manager.label_map
    
    @property
    def topographic_map(self) -> np.ndarray:
        if self._manager is None:
            raise ValueError("No masks have been detected yet. Call `detect` first.")
        return self._manager.topographic_map
    
    def _get_tensor(self, array, channel):
        """
        Extracts a specific channel tensor from the image data.

        Parameters:
        -----------
        channel : int
            The channel index to extract from the image data.

        Returns:
        --------
        numpy.ndarray
            The extracted tensor for the specified channel.

        Notes:
        ------
        - Handles both 5D and 4D image data shapes.
        - Returns a 3D tensor for the specified channel.
        """
        if len(array.shape) == 5:
            image_tensor = array[0, :, channel, :, :]
        elif len(array.shape) == 4:
            image_tensor = array[:, channel, :, :]
        else:
            image_tensor = array

        return image_tensor
    
    def _find_slice_with_highest_intensity(self, z_stack):
        """
        Finds the 2D array (slice) in the z-stack with the highest overall intensity.

        Parameters:
        z_stack (numpy.ndarray): A 3D numpy array representing the z-stack of 2D arrays.

        Returns:
        numpy.ndarray: The 2D array (slice) with the highest overall intensity.
        """
        # Calculate the sum of intensities for each slice
        intensity_sums = np.sum(z_stack, axis=(1, 2))
        
        # Find the index of the slice with the highest intensity sum
        max_intensity_index = np.argmax(intensity_sums)

        # Return the slice with the highest intensity
        return z_stack[max_intensity_index]

    def _equalize_image_clahe(self, image, clip_limit=20.0, tile_grid_size=(4, 4), normalize_output=False):
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

def assign_colors_based_on_volume(podosomes: dict):
    """
    Assigns a color to each podosome based on its volume, mapping min volume to green and max to red.
    
    Parameters:
        podosomes (dict): Dictionary of Podosome objects.

    Returns:
        None (modifies podosomes in-place)
    """
    # Get all volume values
    volumes = np.array([p.volume for p in podosomes.values()])
    
    if len(volumes) == 0:
        return  # No podosomes to process
    
    min_vol, max_vol = volumes.min(), volumes.max()

    if min_vol == max_vol:
        # If all podosomes have the same volume, assign them all red
        for podosome in podosomes.values():
            podosome.color = (255, 0, 0)
        return

    # Assign color based on volume position
    for podosome in podosomes.values():
        norm_value = (podosome.volume - min_vol) / (max_vol - min_vol)  # Normalize to [0, 1]
        r = int(255 * (1 - norm_value))  # Red decreases as volume increases
        g = int(255 * norm_value)        # Green increases as volume increases
        podosome.color = (r, g, 0) 

    return podosomes

def plot_signals(signals: list):
    """
    Plots podosome-associated signals with RelativeSignalHeight on the y-axis and DistanceToBorder on the x-axis.
    
    Parameters:
        signals (list): List of dictionaries containing:
                        - 'PodosomeAssociated': Boolean indicating if the signal is associated with a podosome.
                        - 'DistanceToBorder': Float or None, the shortest distance to the border.
                        - 'RelativeSignalHeight': Float or None, the relative signal height from the heatmap.
    """
    # Extract data for podosome-associated signals
    x_values = []
    y_values = []
    
    for signal in signals:
        if signal.get("PodosomeAssociated", False):
            distance = signal.get("DistanceToBorder")
            height = signal.get("RelativeSignalHeight")
            
            # Only include signals with valid distance and height values
            if distance is not None and height is not None:
                x_values.append(distance)
                y_values.append(height)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.7, edgecolors='k')
    
    # Add labels and title
    plt.xlabel("Distance to Border")
    plt.ylabel("Relative Signal Height")
    plt.title("Podosome-Associated Signals: Relative Height vs. Distance to Border")

    # Force y-axis to be in the range [0, 1]
    plt.ylim(0, 1.1)

    # Show the plot
    plt.grid(True)
    plt.show()

class SignalDetector:
    def __init__(self, image_input: np.ndarray, channel: int = 0):
        self._image_data = image_input
        self._single_channel_tensor = self._get_tensor(channel)
        projection_tensor = Projection(self._single_channel_tensor)
        self._maximum_projection = projection_tensor.projection(projection_type="maximum")
        self._average_projection = projection_tensor.projection(projection_type="average")
        self._max_value = np.max(self._maximum_projection)
        self._ninety_third_percentile = np.percentile(self._maximum_projection, 93)
        self._thresholding_intervals = self._get_thresholding_intervals()
        self._thresholded_stack = self._get_thresholded_stack()
        self._maximum_projection = np.max(self._thresholded_stack, axis=0)
        self._signals = None

    def _get_tensor(self, channel):

        if len(self._image_data.shape) == 5:
            image_tensor = self._image_data[0, :, channel, :, :]
        elif len(self._image_data.shape) == 4:
            image_tensor = self._image_data[:, channel, :, :]
        
        return image_tensor

    def _get_thresholded_stack(self):
        """
        Applies a threshold to the maximum projection of the image stack.

        Returns:
        --------
        numpy.ndarray
            The thresholded image stack where values below the threshold are set to 0.

        Notes:
        ------
        - Uses a 93rd percentile threshold.
        - Applies a Gaussian blur to the maximum projection before thresholding.
        - Ensures the resulting stack has the same shape as the original image stack.
        """
        threshold = self._ninety_third_percentile
        max_projection = self._maximum_projection
        image_stack = self._single_channel_tensor

        max_projection = cv2.GaussianBlur(max_projection, (3, 3), 0)
        thresholded_image = np.where(max_projection <= threshold, 0, 255)

        assert image_stack.shape[1:] == thresholded_image.shape, "Arrays must be of the same size"

        masked_stack = image_stack.copy()
        for i in range(masked_stack.shape[0]):
            masked_stack[i, thresholded_image == 0] = 0

        return masked_stack
    
    def _get_thresholding_intervals(self):
        """
        Generates a list of thresholding intervals based on the maximum value.

        Returns:
        --------
        list of tuples
            A list of tuples where each tuple represents a thresholding interval.

        Notes:
        ------
        - Intervals start at 40% of the maximum value and increment by 10.
        - The last interval always ends at the maximum value.
        """
        max_value = self._max_value
        start_point = int(max_value * 0.4)
        
        intervals = []
        increment = 10
        while start_point < max_value:
            intervals.append((start_point, max_value))
            start_point += increment

        return intervals

    def detect(self):

        def analyze_contour_region(cnt, image, adjust_min_radius: Optional[int] = None) -> Signal:
            """
            Analyze and calculate key values for a given contour region in an image.
            """
            def largest_inscribed_circle(contour):
                # Find the convex hull of the contour
                hull = cv2.convexHull(contour)
                
                # Initialize variables to store the center and radius of the largest inscribed circle
                center = (0, 0)
                radius = 0
                
                # Iterate over each point on the convex hull
                for i in range(len(hull)):
                    # Get the current point on the convex hull
                    hull_point = tuple(hull[i][0])
                    
                    # Calculate the maximum distance from the current point to any point on the contour
                    distances = np.linalg.norm(contour - hull_point, axis=2)
                    max_distance = np.max(distances)
                    
                    # If the maximum distance is greater than the current radius, update the center and radius
                    if max_distance > radius:
                        radius = max_distance
                        center = hull_point
                
                return center, radius
        
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            (x_inside, y_inside), radius_inside = largest_inscribed_circle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            if adjust_min_radius is not None and radius < adjust_min_radius:
                radius = adjust_min_radius
            
            circle_area = np.pi * (radius ** 2)
            ratio_cnt_circle_area = area / circle_area
            circumference = 2 * np.pi * radius
            ratio_cnt_circle_perimeter = perimeter / circumference

            # Extract the circle region
            mask = np.zeros_like(image)
            cv2.circle(mask, center, radius, 255, -1)
            circle_region = cv2.bitwise_and(image, mask)
            
            # Apply Gaussian blur to the circle region
            blurred_circle_region = cv2.GaussianBlur(circle_region, (5, 5), 0)
            
            # Find local maxima in the blurred circle region
            local_max = cv2.dilate(blurred_circle_region, np.ones((3, 3), np.uint8))
            local_maxima = np.where((blurred_circle_region == local_max) & (blurred_circle_region > 0))
            local_maxima = list(zip(local_maxima[1], local_maxima[0]))

            signal = Signal(
                center=center,
                radius=radius,
                center_inside=(int(x_inside), int(y_inside)),
                radius_inside=int(radius_inside),
                minor=radius,
                area_contour=area,
                area_circle=circle_area,
                area_ratio=ratio_cnt_circle_area,
                perimeter=perimeter,
                perimeter_ratio=ratio_cnt_circle_perimeter,
                orientation=None,
                coordinates=cnt,
                local_maxima=local_maxima,
                is_valid=True
            )

            return signal

        def validate_regions(regions: List[Signal], circularity_threshold: float = None, 
                        allowed_radius: int = None, strict: bool = False, check_local_maxima: bool = True, 
                        exclude_pinched_signals: bool = False) -> List[Dict[str, Any]]:
            """
            Validate each region based on the specified criteria.
            """
            signals = []
            for region in regions:

                if allowed_radius is not None and region.radius >= allowed_radius:
                    region.is_valid = False
                
                if circularity_threshold is not None and region.area_ratio <= circularity_threshold:
                    region.is_valid = False

                if circularity_threshold is not None and region.perimeter_ratio <= circularity_threshold:
                    region.is_valid = False

                if exclude_pinched_signals:
                    contour = region.coordinates
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    region.is_valid = not cv2.isContourConvex(approx)

                if check_local_maxima and len(region.local_maxima) > 1:
                    region.is_valid = False
                else:
                    region.is_valid = True
                
                if not strict or region.is_valid:
                    signals.append(region)

            return signals

        def extract_signals(image: np.ndarray, min_contour_area: int = 4, max_contour_area: int = 30, 
                    circularity_threshold: float = None, color=(255, 255, 255), threshold=0, allowed_radius=8, 
                    adjust_min_radius: int = None, exclude_pinched_signals=True, check_local_maxima=True, strict=True,
                    **kwargs) -> List[Dict[str, any]]:
            """
            Extract blob signals from the image.
            """
            average_projection = self._average_projection
            _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            signals = []
            for contour in contours:

                region = analyze_contour_region(contour, average_projection, adjust_min_radius=adjust_min_radius)
                
                signals.append(region)

            signals = validate_regions(signals, circularity_threshold=circularity_threshold, allowed_radius=allowed_radius, 
                                        exclude_pinched_signals=exclude_pinched_signals, strict=strict)

            if color is not None:
                for region in signals:
                    region.circle_color = color
            
            return signals

        def apply_mask(image: np.ndarray, regions: List[Dict[str, any]], threshold=0, use_circle=True) -> np.ndarray:
            """
            Apply a mask to the image based on the provided regions.
            
            :param image: The input image or image stack.
            :param regions: A list of dictionaries representing regions to mask.
            :param threshold: The threshold value to apply after masking. Only when using circles.
            :param use_circle: A boolean flag to indicate whether to use circles or contours for masking.
            :return: The masked image.
            """
            masked_image = image.copy()
            
            # Ensure the image is in the correct shape
            if masked_image.ndim == 2:
                masked_image = np.expand_dims(masked_image, axis=0)
            
            depth, height, width = masked_image.shape
            
            # Create a blank mask with the same shape as the image
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process each region to create the mask
            for region in regions:
                if use_circle:
                    # Handle circle-based regions
                    x_center, y_center = region.center
                    if region.radius < 5:
                        radius = 5
                    radius = int(region.radius + 5)
                    y, x = np.ogrid[:height, :width]
                    circle_mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
                    mask |= circle_mask.astype(np.uint8)
                else:
                    # Handle contour-based regions
                    contour = region.coordinates
                    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            
            if use_circle:
                for z in range(depth):
                    masked_image[z][mask.astype(bool)] = 0
                masked_image = np.squeeze(masked_image)
                masked_image[masked_image < threshold] = 0
            else:
                masked_image = cv2.bitwise_and(image, mask)
            
            return masked_image

        def find_peaks(image):
            local_max = peak_local_max(image, min_distance=5, threshold_abs=0, exclude_border=False)

            regions = []
            for y,x in local_max:

                signal = Signal(
                    center=(int(x), int(y)),
                    radius=5,
                    center_inside=(int(y), int(x)),
                    radius_inside=5,
                    minor=5,
                    area_contour=79,
                    area_circle=79,
                    area_ratio=1,
                    perimeter=31.415,
                    perimeter_ratio=1,
                    orientation=None,
                    coordinates=None,
                    local_maxima=None,
                    is_valid=True,
                    pinched=None,
                    approximation=None,
                    circle_color=(127, 127, 255)
                )

                regions.append(signal)
            
            return regions
        
        def iteratively_blur(image_input, iterations=1):

            image = image_input.copy()
            for i in range(iterations):
                image = cv2.GaussianBlur(image, (3, 3), 0)
            return image

        def filter_circles(circles: List[Signal], max_radius=None, min_radius=None) -> List[Signal]:
            """
            Filter out overlapping circles.
            """
            def is_inside(circle1: Signal, circle2: Signal) -> bool:
                x1, y1 = circle1.center
                r1 = circle1.radius
                x2, y2 = circle2.center
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                return distance <= r1
            
            if max_radius is not None or min_radius is not None:
                size_restricted = []
                for circle in circles:
                    if (max_radius is None or circle.radius <= max_radius) and (min_radius is None or circle.radius > min_radius):
                        size_restricted.append(circle)
                return size_restricted

            unique_centers = set()
            filtered_circles = []
            for circle in circles:
                center = circle.center
                if center not in unique_centers:
                    unique_centers.add(center)
                    filtered_circles.append(circle)

            filtered_circles.sort(key=lambda c: c.radius, reverse=True)
            solitary_circles = []
            other_circles = copy.deepcopy(filtered_circles)

            for circle in filtered_circles:
                other_circles.pop(0)
                center, radius = circle.center, circle.radius

                is_solitary = True
                
                for other_circle in other_circles:
                    other_center, other_radius = other_circle.center, other_circle.radius
                    if center == other_center and radius == other_radius:
                        continue
                    if is_inside(circle, other_circle):
                        is_solitary = False
                        break
                if is_solitary:
                    solitary_circles.append(circle)
            
            return solitary_circles

        def generate_interval_thresholded_images(image: np.ndarray) -> Generator[np.ndarray, None, None]:
            """
            Generate binary images from the original image.
            """
            blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
            for markers in self._thresholding_intervals:
                binary_image = np.where((blurred_image > markers[1]) | (blurred_image < markers[0]), 0, 255).astype(np.uint8)
                _, thresholded_image = cv2.threshold(binary_image, 254, 255, cv2.THRESH_BINARY)
                yield thresholded_image

        def detect_signals(image: np.ndarray, min_contour_area: int = 4, circularity_threshold: float = None, color=(255, 255, 255),
                    threshold=0, allowed_radius=8, adjust_min_radius: int = None, exclude_pinched_signals=True, check_local_maxima=True,
                    strict=True, use_threshold_intervals=False, use_peak_detection_only=False, signals=None) -> List[Dict[str, any]]:
            """
            Detect signals from the binary images.
            """
            
            params = {
                "min_contour_area": min_contour_area,
                "circularity_threshold": circularity_threshold,
                "color": color,
                "threshold": threshold,
                "allowed_radius": allowed_radius,
                "adjust_min_radius": adjust_min_radius,
                "exclude_pinched_signals": exclude_pinched_signals,
                "check_local_maxima": check_local_maxima,
                "strict": strict,
                "use_threshold_intervals": use_threshold_intervals,
            }

            # Ensure the image is in the correct shape
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)

            depth = image.shape[0]

            if signals is None:
                all_signals = []
            else:
                all_signals = list(signals)

            for i in range(depth):

                if use_peak_detection_only:
                    peaks = find_peaks(image[i])
                    all_signals.extend(peaks)
                elif use_threshold_intervals:
                    for binary_image in generate_interval_thresholded_images(image[i]):
                        signals = extract_signals(binary_image, **params)
                        all_signals.extend(signals)
                else:
                    signals = extract_signals(image[i], **params)
                    all_signals.extend(signals)

            return all_signals

        def find_max_intensity_z_with_circle(detected_signals, image_stack, radius=5):
            """
            Find the z-plane with the maximum intensity around the detected signals, 
            considering only pixels within a circular radius.
            
            Parameters:
            - detected_signals: List of dictionaries with 'x' and 'y' coordinates.
            - image_stack: 3D numpy array representing the z-stack of images (shape: (z, y, x)).
            - radius: The radius around the (x, y) coordinates to consider for intensity summation.

            Returns:
            - Updated list of dictionaries with added 'z' coordinate for each detected signal.
            """
            z_size, y_size, x_size = image_stack.shape
            
            # Create a circular mask with the given radius
            diameter = radius * 2 + 1
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            circular_mask = (x**2 + y**2) <= radius**2
            mask_size = circular_mask.shape
            
            for signal in detected_signals:
                x_center = signal.center[0]
                y_center = signal.center[1]
                
                max_intensity_sum = -np.inf
                best_z = 0
                
                # Loop through each z-slice
                for z in range(z_size):
                    # Define the ROI around the (x_center, y_center) within the radius
                    x_min = max(0, x_center - radius)
                    x_max = min(x_size, x_center + radius + 1)
                    y_min = max(0, y_center - radius)
                    y_max = min(y_size, y_center + radius + 1)
                    
                    # Extract the ROI from the current z-slice
                    roi = image_stack[z, y_min:y_max, x_min:x_max]
                    
                    # Ensure the ROI has the same size as the mask
                    if roi.shape != mask_size:
                        # Adjust the mask to the current ROI size if necessary
                        adjusted_mask = circular_mask[:roi.shape[0], :roi.shape[1]]
                    else:
                        adjusted_mask = circular_mask
                    
                    # Apply the mask to the ROI
                    masked_roi = np.where(adjusted_mask, roi, 0)
                    
                    # Sum the intensity within the circular region
                    intensity_sum = np.sum(masked_roi)
                    
                    # Check if this is the maximum intensity sum
                    if intensity_sum > max_intensity_sum:
                        max_intensity_sum = intensity_sum
                        best_z = z
                signal.z = best_z

            return detected_signals

        def process_signals():
            """
            Process signals according to the specified procedure.
            
            :return: List of detected signal regions.
            """
            ### Initial Signal Detection and Masking
            initial_signals = detect_signals(self._maximum_projection, color=(255, 0, 255), threshold=self._ninety_third_percentile, circularity_threshold=0.7)
            masked_max_projection = apply_mask(self._maximum_projection, initial_signals)
            masked_image_stack = apply_mask(self._thresholded_stack, initial_signals)

            stricter_signals = detect_signals(masked_max_projection, color=(255, 255, 0), threshold=int(self._max_value - 5), strict=True, min_contour_area=2, circularity_threshold=0.7)
            masked_max_projection_stricter = apply_mask(masked_max_projection, stricter_signals)
            masked_image_stack_stricter = apply_mask(masked_image_stack, stricter_signals)
            initial_signals.extend(stricter_signals)

            blurred_max_projection = iteratively_blur(masked_max_projection_stricter, 5)

            blurred_signals = detect_signals(blurred_max_projection, color=(0, 255, 255), threshold=50, strict=True, min_contour_area=2, circularity_threshold=0.7)
            masked_max_projection_blurred = apply_mask(blurred_max_projection, blurred_signals, threshold=50)
            masked_image_stack_blurred = apply_mask(masked_image_stack_stricter, blurred_signals, threshold=50)
            initial_signals.extend(blurred_signals)

            layer_signals = detect_signals(masked_image_stack_blurred, adjust_min_radius=5, color=(0, 255, 0), allowed_radius=None, exclude_pinched_signals=False, use_threshold_intervals=True)
            filtered_layer_signals = filter_circles(layer_signals)
            masked_max_projection_filtered = apply_mask(masked_max_projection_blurred, filtered_layer_signals, threshold=50)
            masked_image_stack_filtered = apply_mask(masked_image_stack_blurred, filtered_layer_signals, threshold=50)
            initial_signals.extend(filtered_layer_signals)

            binarized_max_projection = np.where(masked_max_projection_filtered != 0, 255, masked_max_projection_filtered)
            median_blurred_max_projection = cv2.medianBlur(binarized_max_projection, 5)

            non_strict_signals = detect_signals(median_blurred_max_projection, color=(0, 0, 255), threshold=50, strict=False, min_contour_area=1, circularity_threshold=0.7)
            filtered_non_strict_signals_exempt = filter_circles(non_strict_signals, min_radius=10)
            filtered_non_strict_signals = filter_circles(non_strict_signals, max_radius=10)
            initial_signals.extend(filtered_non_strict_signals)

            average_image = self._average_projection
            masked_average_image = apply_mask(average_image, filtered_non_strict_signals_exempt, use_circle=False)
            blurred_average_image = iteratively_blur(masked_average_image, 5)

            peak_signals = detect_signals(blurred_average_image, use_peak_detection_only=True)
            initial_signals.extend(peak_signals)

            final_signals = find_max_intensity_z_with_circle(initial_signals, self._single_channel_tensor)


            return final_signals
        
        signals = process_signals()

        self._signals = signals

        return signals
    
if __name__ == "__main__":

    pass

        
        
