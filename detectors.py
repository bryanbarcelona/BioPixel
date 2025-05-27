import copy
import math

from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import peak_local_max

from cellpose import models
from data_structures import Podosome, PodosomeCellResult, Signal
from params import DetectionParams, ModelPaths
from utils.io import get_resource_path

mods = ModelPaths()
model_path = get_resource_path(mods.MODEL_MAP["macrophage"])

class CellPoseDetector:
    """Detects cells in an image using the Cellpose model.

    This class handles image preprocessing, detection, and post-processing
    for cell segmentation tasks.
    """

    DEFAULT_MIN_SIZE: int = 15
    CONSTANT_PAD_VALUE: int = 0

    def __init__(
        self,
        image_array: np.ndarray,
        model: str = "macrophage",
        detection_channel: int = 1,
        auxiliary_channel: int = 0,
        diameter: int = 10,
        do_3d: bool = False,
        flow_threshold: float = 0.9,
        cellprob_threshold: float = 0.0,
        filename: Optional[str] = None,
        output_folder: Optional[str] = None,
        gpu: bool = False,
        min_canvas_size: Optional[Tuple[int, int]] = None,
        min_size: int = DEFAULT_MIN_SIZE,
    ) -> None:
        """Initializes the detector with image and configuration parameters.

        Args:
            image_array: Input image as a NumPy array.
            model: Model type for detection. Defaults to "macrophage".
            detection_channel: Primary channel index for detection. Defaults to 1.
            auxiliary_channel: Auxiliary channel index for detection. Defaults to 0.
            diameter: Expected object diameter in pixels. Defaults to 10.
            do_3d: If True, performs 3D detection. Defaults to False.
            flow_threshold: Flow threshold for segmentation. Defaults to 0.9.
            cellprob_threshold: Cell probability threshold. Defaults to 0.0.
            filename: Optional filename for the image. Defaults to None.
            output_folder: Optional folder for output. Defaults to None.
            gpu: If True, uses GPU for computation. Defaults to False.
            min_canvas_size: Minimum canvas size (height, width). If provided,
                smaller images are padded. Defaults to None.
            min_size: Minimum size for detected objects. Defaults to 15.
        """
        self._params = DetectionParams(
            model=model,
            detection_channel=detection_channel,
            auxiliary_channel=auxiliary_channel,
            diameter=diameter,
            do_3d=do_3d,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            filename=filename,
            output_folder=output_folder,
            gpu=gpu,
            min_size=min_size,
        )
        self._original_shape = image_array.shape
        self._min_canvas_size = min_canvas_size
        self._image = self._extract_channel(image_array, detection_channel - 1)
        if min_canvas_size:
            self._image = self._pad_to_min_size(self._image, min_canvas_size)

    def _extract_channel(self, image: np.ndarray, channel: int) -> np.ndarray:
        """Extracts a specific channel from the input image.

        Args:
            image: Input image array.
            channel: Channel index to extract (0-based).

        Returns:
            Extracted channel as a NumPy array.

        Notes:
            - For 4D arrays (c, z, y, x), extracts the specified channel.
            - For 5D arrays (t, c, z, y, x), extracts from the first time point.
            - Other dimensionalities are returned as-is.
        """
        if image.ndim == 4:
            return image[channel, :, :, :]
        if image.ndim == 5:
            return image[0, channel, :, :, :]
        return image

    def _pad_to_min_size(self, image: np.ndarray, min_size: Tuple[int, int]) -> np.ndarray:
        """Pads the image to meet minimum canvas size requirements.

        Args:
            image: Input image array.
            min_size: Minimum canvas size as (height, width).

        Returns:
            Padded image array if needed, else original image.
        """
        height, width = image.shape[-2:]
        min_height, min_width = min_size

        if height >= min_height and width >= min_width:
            return image

        pad_height = (min_height - height) // 2
        pad_width = (min_width - width) // 2
        padding = [(0, 0)] * (image.ndim - 2) + [(pad_height, pad_height), (pad_width, pad_width)]
        return np.pad(image, padding, mode="constant", constant_values=self.CONSTANT_PAD_VALUE)

    def _trim_to_original(self, image: np.ndarray) -> np.ndarray:
        """Trims the image to its original spatial dimensions.

        Args:
            image: Image array to trim.

        Returns:
            Trimmed image array matching original spatial dimensions.
        """
        if not self._min_canvas_size:
            return image

        original_height, original_width = self._original_shape[-2:]
        height, width = image.shape[-2:]
        pad_height = (height - original_height) // 2
        pad_width = (width - original_width) // 2
        slices = [slice(None)] * (image.ndim - 2) + [
            slice(pad_height, pad_height + original_height),
            slice(pad_width, pad_width + original_width),
        ]
        return image[tuple(slices)]

    @property
    def image(self) -> np.ndarray:
        """Processed image tensor after channel extraction and optional padding.

        Returns:
            Processed image as a NumPy array.
        """
        return self._image

    def detect(self, mask_dict: bool = False, **param_overrides: dict) -> np.ndarray:
        """Performs cell detection using the Cellpose model.

        Args:
            mask_dict: If True, returns additional detection data (not implemented).
            **param_overrides: Keyword arguments to override default parameters.

        Returns:
            Detected masks as a NumPy array with uint16 dtype.
        """
        params = copy.deepcopy(self._params)
        params.update_params(**param_overrides)

        model = models.CellposeModel(gpu=params.gpu, pretrained_model=params.model)
        masks, _, _ = model.eval(
            self._image,
            channels=[params.detection_channel, params.auxiliary_channel],
            diameter=params.diameter or model.diam_labels,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
            do_3D=params.do_3d,
            min_size=params.min_size,
        )

        return self._trim_to_original(masks.astype(np.uint16))

class MacrophageDetector:
    """Detects macrophages in an image using Cellpose with specific preprocessing and post-processing."""

    DEFAULT_DIAMETER: int = 700
    DEFAULT_CHANNEL: int = 1
    DEFAULT_FLOW_THRESHOLD: float = 0.4
    DEFAULT_CELLPROB_THRESHOLD: float = 0.0
    DEFAULT_MIN_CANVAS_SIZE: Tuple[int, int] = (3000, 3000)
    MODEL_TYPE: str = "macrophage"

    def __init__(
        self,
        image_array: np.ndarray,
        diameter: int = DEFAULT_DIAMETER,
        channel: int = DEFAULT_CHANNEL,
        only_center_mask: bool = False,
    ) -> None:
        """Initializes the detector with an image and detection parameters.

        Args:
            image_array: Input image as a NumPy array.
            diameter: Expected macrophage diameter in pixels. Defaults to 700.
            channel: Channel index for detection (1-based). Defaults to 1.
            only_center_mask: If True, keeps only the centermost mask. Defaults to False.
        """
        self._image_array = image_array
        self._channel = channel
        self._diameter = diameter
        self.masks = self.detect()
        if only_center_mask:
            self.masks = self._reduce_to_centermost_mask(self.masks)

    @property
    def image(self) -> np.ndarray:
        """Processed image tensor used for detection.

        Returns:
            Processed image as a NumPy array.
        """
        return self._image_array

    def detect(self) -> np.ndarray:
        """Performs macrophage detection using Cellpose.

        Returns:
            Detected masks as a NumPy array with uint16 dtype.
        """
        image_3d = self._extract_channel(self._image_array, self._channel - 1)
        image_2d = self._find_highest_intensity_slice(image_3d)

        detector = CellPoseDetector(
            image_array=image_2d,
            model=self.MODEL_TYPE,
            detection_channel=1,
            diameter=self._diameter,
            do_3d=False,
            flow_threshold=self.DEFAULT_FLOW_THRESHOLD,
            cellprob_threshold=self.DEFAULT_CELLPROB_THRESHOLD,
            gpu=True,
            min_canvas_size=self.DEFAULT_MIN_CANVAS_SIZE,
        )
        return detector.detect()

    def _extract_channel(self, image: np.ndarray, channel: int) -> np.ndarray:
        """Extracts a specific channel from the input image.

        Args:
            image: Input image array.
            channel: Channel index to extract (0-based).

        Returns:
            Extracted channel as a 3D NumPy array.

        Notes:
            - For 5D arrays (t, z, c, y, x), extracts from first time point.
            - For 4D arrays (z, c, y, x), extracts specified channel.
        """
        if image.ndim == 5:
            return image[0, :, channel, :, :]
        if image.ndim == 4:
            return image[:, channel, :, :]
        return image

    def _find_highest_intensity_slice(self, z_stack: np.ndarray) -> np.ndarray:
        """Finds the 2D slice with the highest total intensity in a 3D z-stack.

        Args:
            z_stack: 3D array representing a z-stack of 2D images.

        Returns:
            2D array with the highest total intensity.
        """
        intensity_sums = np.sum(z_stack, axis=(1, 2))
        max_intensity_idx = np.argmax(intensity_sums)
        return z_stack[max_intensity_idx]

    def _reduce_to_centermost_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """Reduces segmentation to the centermost mask or the closest to the center.

        Args:
            segmentation: 2D or 3D labeled mask array.

        Returns:
            Segmentation array with only the centermost mask (uint16 dtype).
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

    def blackout_non_mask_areas(self) -> Generator[np.ndarray, None, None]:
        """Yields images with non-mask areas blacked out for each mask.

        Returns:
            Generator yielding cropped images with non-mask areas set to zero.
        """
        for mask_id in np.unique(self.masks):
            if mask_id == 0:  # Skip background
                continue

            mask = self.masks == mask_id
            y_indices, x_indices = np.where(mask)
            if not y_indices.size or not x_indices.size:
                continue

            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)

            masked_image = np.zeros_like(self._image_array)
            mask_broadcasted = np.broadcast_to(
                mask, self._image_array.shape[:-2] + mask.shape[-2:]
            )
            masked_image[mask_broadcasted] = self._image_array[mask_broadcasted]

            yield masked_image[..., min_y : max_y + 1, min_x : max_x + 1]
 
class PodosomeManager:
    """Manages podosome detection and processing in 3D segmentation masks."""

    DEFAULT_SIZE_UM: float = 1.0
    DEFAULT_RESOLUTION_XY: float = 5.0
    DEFAULT_RESOLUTION_Z: float = 1.0
    DEFAULT_DILATION_ITERATIONS: int = 1
    DEFAULT_MARGIN: int = 30
    DEFAULT_SCALE_FACTOR: float = 2.0

    def __init__(
        self,
        mask_3d: np.ndarray,
        size_um: float = DEFAULT_SIZE_UM,
        resolution_xy: float = DEFAULT_RESOLUTION_XY,
        resolution_z: float = DEFAULT_RESOLUTION_Z,
        dilation_iterations: int = DEFAULT_DILATION_ITERATIONS,
        cell_id: Optional[int] = None,
    ) -> None:
        """Initializes the manager with a 3D segmentation mask and parameters.

        Args:
            mask_3d: 3D segmentation mask (Z, Y, X).
            size_um: Kernel size in micrometers. Defaults to 1.0.
            resolution_xy: XY resolution in pixels per micrometer. Defaults to 5.0.
            resolution_z: Z resolution in pixels per micrometer. Defaults to 1.0.
            dilation_iterations: Number of dilation iterations. Defaults to 1.
            cell_id: Optional identifier for time-series tracking. Defaults to None.
        """
        self.mask_3d = mask_3d
        self.cell_id = cell_id or 0
        self._kernel_xy, self._kernel_z = self._generate_kernels(size_um, resolution_xy, resolution_z)
        self._kernel_offsets_xy = np.argwhere(self._kernel_xy == 1) - (self._kernel_xy.shape[0] // 2, self._kernel_xy.shape[1] // 2)
        self._kernel_offsets_z = np.argwhere(self._kernel_z == 1) - (self._kernel_z.shape[0] // 2, self._kernel_z.shape[1] // 2)
        self.cell_result = self.process()

    def process(self) -> PodosomeCellResult:
        """Processes the 3D mask to detect and analyze podosomes.

        Returns:
            PodosomeCellResult containing processed podosomes and label map.
        """
        podosomes = {}
        unique_ids = np.unique(self.mask_3d)[1:]  # Exclude background (0)
        for podosome_id in unique_ids:
            print(f"Processing podosome ID: {podosome_id}")
            binary_mask = (self.mask_3d == podosome_id).astype(np.uint8)
            dilated_mask = self._process_single_podosome(binary_mask)
            podosomes[podosome_id] = self._create_podosome(podosome_id, binary_mask, dilated_mask)

        cell = PodosomeCellResult(cell_id=self.cell_id, podosomes=podosomes, label_map=self.mask_3d)
        self._calculate_relative_volumes(cell)
        return cell

    def _generate_kernels(self, size_um: float, resolution_xy: float, resolution_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generates elliptical kernels for XY and Z dilation.

        Args:
            size_um: Kernel size in micrometers.
            resolution_xy: XY resolution in pixels per micrometer.
            resolution_z: Z resolution in pixels per micrometer.

        Returns:
            Tuple of XY and Z kernels as NumPy arrays.
        """
        kernel_radius_xy = int(round(size_um * resolution_xy))
        kernel_radius_z = int(round(size_um * resolution_z))
        kernel_size_xy = 2 * kernel_radius_xy + 1
        kernel_size_z = 2 * kernel_radius_z + 1
        return (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_xy, kernel_size_xy)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_z, kernel_size_z)),
        )

    def _process_single_podosome(self, binary_mask: np.ndarray) -> np.ndarray:
        """Processes a single podosome mask through dilation.

        Args:
            binary_mask: Binary mask for a single podosome.

        Returns:
            Dilated mask as a NumPy array.
        """
        cropped_mask, coords = self._extract_mask_with_margin(binary_mask, self.DEFAULT_MARGIN)
        if cropped_mask is None:
            return np.zeros_like(binary_mask)
        dilated_mask = self._create_dilated_mask_3d(cropped_mask)
        return self._reinsert_dilated_mask(binary_mask.shape, dilated_mask, coords)

    def _create_podosome(self, podosome_id: int, binary_mask: np.ndarray, dilated_mask: np.ndarray) -> Podosome:
        """Creates a Podosome object from binary and dilated masks.

        Args:
            podosome_id: Unique identifier for the podosome.
            binary_mask: Original binary mask.
            dilated_mask: Dilated binary mask.

        Returns:
            Populated Podosome object.
        """
        podosome = Podosome(id=podosome_id)
        for z in range(binary_mask.shape[0]):
            for mask, dilated in [(binary_mask, False), (dilated_mask, True)]:
                slice_data = (mask[z] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(slice_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area = cv2.contourArea(contours[0]) if contours else 0
                y, x = np.where(slice_data > 0)
                podosome._add_slice_data(
                    z=z,
                    contour=contours[0] if contours else None,
                    pixels=list(zip(y, x)),
                    area=area,
                    dilated=dilated,
                )
        if podosome.slices:
            podosome._calculate_centroid()
            podosome._calculate_bounding_box(dilated=False)
            podosome._calculate_bounding_box(dilated=True)
        return podosome

    def _calculate_relative_volumes(self, cell: PodosomeCellResult) -> None:
        """Normalizes podosome volumes within the cell.

        Args:
            cell: PodosomeCellResult containing podosomes to process.
        """
        volumes = [p.volume for p in cell.podosomes.values()]
        if not volumes:
            return
        min_vol, max_vol = min(volumes), max(volumes)
        for podosome in cell.podosomes.values():
            podosome.relative_volume = (
                (podosome.volume - min_vol) / (max_vol - min_vol) if max_vol != min_vol else 1.0
            )

    def _extract_mask_with_margin(self, mask: np.ndarray, margin: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Extracts a 3D mask with a margin for dilation.

        Args:
            mask: 3D binary mask (Z, Y, X).
            margin: Margin size in pixels for XY dimensions.

        Returns:
            Tuple of cropped mask and starting coordinates (min_y, min_x), or (None, None) if empty.
        """
        non_zero_y = np.any(mask, axis=(0, 2))
        non_zero_x = np.any(mask, axis=(0, 1))
        if not non_zero_y.any() or not non_zero_x.any():
            return None, None
        min_y, max_y = np.where(non_zero_y)[0][[0, -1]]
        min_x, max_x = np.where(non_zero_x)[0][[0, -1]]
        shape_z, shape_y, shape_x = mask.shape
        min_y, max_y = max(min_y - margin, 0), min(max_y + margin, shape_y - 1)
        min_x, max_x = max(min_x - margin, 0), min(max_x + margin, shape_x - 1)
        return mask[:, min_y:max_y + 1, min_x:max_x + 1], (min_y, min_x)

    def _reinsert_dilated_mask(self, original_shape: Tuple[int, int, int], dilated_mask: np.ndarray, start_coords: Tuple[int, int]) -> np.ndarray:
        """Reinserts a dilated mask into the original 3D volume.

        Args:
            original_shape: Shape of the original 3D volume (Z, Y, X).
            dilated_mask: Dilated cropped mask.
            start_coords: Starting coordinates (min_y, min_x) for reinsertion.

        Returns:
            Reinserted mask as a NumPy array.
        """
        reinserted_mask = np.zeros(original_shape, dtype=dilated_mask.dtype)
        min_y, min_x = start_coords
        _, dy, dx = dilated_mask.shape
        reinserted_mask[:, min_y:min_y + dy, min_x:min_x + dx] = dilated_mask
        return reinserted_mask

    def _expand_mask_cv2(self, input_mask: np.ndarray, scale_factor: float = DEFAULT_SCALE_FACTOR) -> np.ndarray:
        """Expands a 2D binary mask by scaling its contour.

        Args:
            input_mask: 2D binary mask (1 for mask, 0 for background).
            scale_factor: Factor to scale the contour. Defaults to 2.0.

        Returns:
            Expanded mask as a NumPy array.
        """
        input_mask = (input_mask / 255).astype(np.uint8)
        contours, _ = cv2.findContours(input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(input_mask)
        contour = contours[0]
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        moments = cv2.moments(approx_contour)
        if moments["m00"] == 0:
            return np.zeros_like(input_mask)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        expanded_contour = np.array([[
            (point[0][0] - center_x) * scale_factor + center_x,
            (point[0][1] - center_y) * scale_factor + center_y
        ] for point in approx_contour])
        expanded_mask = np.zeros_like(input_mask)
        if not expanded_mask.flags['C_CONTIGUOUS']:
            expanded_mask = np.ascontiguousarray(expanded_mask)
        cv2.drawContours(expanded_mask, [expanded_contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        return expanded_mask

    def _create_dilated_mask_3d(self, segmentation_3d: np.ndarray) -> np.ndarray:
        """Creates a 3D dilated binary mask with anisotropic dilation.

        Args:
            segmentation_3d: 3D segmentation mask.

        Returns:
            Dilated 3D binary mask as a NumPy array.
        """
        binary_mask = np.where(segmentation_3d > 0, 255, 0).astype(np.uint8)
        dilated_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        for z in range(binary_mask.shape[0]):
            dilated_mask[z] = self._expand_mask_cv2(binary_mask[z])
        dilated_mask_yz = np.transpose(dilated_mask, (1, 0, 2))
        for y in range(dilated_mask_yz.shape[0]):
            dilated_mask_yz[y] = cv2.dilate(dilated_mask_yz[y], self._kernel_z, iterations=self.DEFAULT_DILATION_ITERATIONS)
        dilated_mask_yz = np.transpose(dilated_mask_yz, (1, 0, 2))
        dilated_mask_xz = np.transpose(dilated_mask, (2, 1, 0))
        for x in range(dilated_mask_xz.shape[0]):
            dilated_mask_xz[x] = cv2.dilate(dilated_mask_xz[x], self._kernel_z, iterations=self.DEFAULT_DILATION_ITERATIONS)
        dilated_mask_xz = np.transpose(dilated_mask_xz, (2, 1, 0))
        return np.maximum.reduce([dilated_mask, dilated_mask_yz, dilated_mask_xz])

    def get_centroids(self) -> Dict[int, Tuple[float, float, float]]:
        """Returns centroids of all podosomes for tracking.

        Returns:
            Dictionary mapping podosome IDs to their centroids.
        """
        return {p.id: p.centroid for p in self.cell_result.podosomes.values() if p.centroid}

    def get_podosomes(self) -> Dict[int, Podosome]:
        """Returns the dictionary of podosome objects.

        Returns:
            Dictionary mapping podosome IDs to Podosome objects.
        """
        return self.cell_result.podosomes

    @property
    def masks(self) -> np.ndarray:
        """Binary masks for all podosomes.

        Returns:
            Binary mask array (255 for mask, 0 for background) as uint8.
        """
        return np.where(self.mask_3d > 0, 255, 0).astype(np.uint8)

    @property
    def dilated_masks(self) -> np.ndarray:
        """Dilated masks for all podosomes.

        Returns:
            Dilated binary mask array as uint8.
        """
        return self._create_dilated_mask_3d(self.mask_3d)

    @property
    def label_map(self) -> np.ndarray:
        """Label map for all podosomes.

        Returns:
            3D array where each voxel contains a tuple of podosome IDs.
        """
        return self._create_label_map_from_dilated_masks()

    @property
    def topographic_map(self) -> np.ndarray:
        """Topographic map for all podosomes based on relative heights.

        Returns:
            3D heatmap array as float32.
        """
        return self._generate_podosome_topographic()

    def filter_podosomes(self, distance_threshold: float = 5.0, volume_threshold: float = 15.0) -> None:
        """Filters podosomes based on nearest-neighbor distance or volume.

        Args:
            distance_threshold: Minimum distance to nearest neighbor to keep a podosome.
            volume_threshold: Minimum volume to keep a podosome.
        """
        podosome_ids = list(self.cell_result.podosomes.keys())
        if not podosome_ids:
            return
        centroids = np.array([self.cell_result.podosomes[i].centroid for i in podosome_ids if self.cell_result.podosomes[i].centroid], dtype=np.float64)
        volumes = np.array([self.cell_result.podosomes[i].volume for i in podosome_ids], dtype=np.float64)
        if len(podosome_ids) > 1 and len(centroids) == len(podosome_ids):
            diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(distances, np.inf)
            nearest_distances = np.min(distances, axis=1)
        else:
            nearest_distances = np.full(len(podosome_ids), np.inf)
        keep_mask = (nearest_distances > distance_threshold) | (volumes >= volume_threshold)
        self.cell_result.podosomes = {pid: self.cell_result.podosomes[pid] for i, pid in enumerate(podosome_ids) if keep_mask[i]}

    def _create_label_map_from_dilated_masks(self) -> np.ndarray:
        """Creates a 3D label map from dilated podosome masks.

        Returns:
            3D array where each voxel contains a tuple of podosome IDs.
        """
        label_map = np.empty_like(self.mask_3d, dtype=object)
        for podosome_id, podosome in self.get_podosomes().items():
            for z, data in podosome.dilated_slices.items():
                for y, x in data.pixels:
                    if label_map[z, y, x] is None:
                        label_map[z, y, x] = (podosome_id,)
                    else:
                        label_map[z, y, x] = tuple(sorted(set(label_map[z, y, x] + (podosome_id,))))
        return label_map

    def _extract_slices_with_masks(self) -> Dict[int, Podosome]:
        """Extracts podosomes with non-empty slices.

        Returns:
            Dictionary of podosomes with only non-empty slices.
        """
        podosomes_with_masks = {}
        for podosome_id, podosome in self.get_podosomes().items():
            masked_podosome = copy.deepcopy(podosome)
            masked_podosome.slices = {z: data for z, data in podosome.slices.items() if data.pixels}
            masked_podosome.dilated_slices = {z: data for z, data in podosome.dilated_slices.items() if data.pixels}
            podosomes_with_masks[podosome_id] = masked_podosome
        return podosomes_with_masks

    def _generate_podosome_topographic(self) -> np.ndarray:
        """Generates a 3D topographic heatmap based on podosome relative heights.

        Returns:
            3D heatmap array as float32.
        """
        topography_map = np.zeros_like(self.mask_3d, dtype=np.float32)
        count_map = np.zeros_like(self.mask_3d, dtype=np.uint8)
        slice_reduced_podosomes = self._extract_slices_with_masks()

        for podosome in slice_reduced_podosomes.values():
            original_z_layers = sorted(podosome.slices.keys())
            if not original_z_layers:
                continue
            num_layers = len(original_z_layers)
            podosome.relative_height_map = {z: (i + 1) / num_layers for i, z in enumerate(original_z_layers)}

        for podosome in slice_reduced_podosomes.values():
            if not hasattr(podosome, "relative_height_map") or not podosome.slices:
                continue
            z_layers = sorted(podosome.dilated_slices.keys())
            original_z_layers = sorted(podosome.slices.keys())
            min_z, max_z = original_z_layers[0], original_z_layers[-1]
            num_layers = len(original_z_layers)

            for z in z_layers:
                relative_height = (
                    podosome.relative_height_map.get(z, 0.0) or
                    (1.0 + (z - max_z) / num_layers if z > max_z else -(min_z - z) / num_layers)
                )
                for y, x in podosome.dilated_slices[z].pixels:
                    topography_map[z, y, x] += relative_height
                    count_map[z, y, x] += 1

        dilated_mask = count_map > 0
        topography_map[dilated_mask] /= count_map[dilated_mask]
        return topography_map

class PodosomeDetector:
    """Detects podosomes in a 3D image using Cellpose with intensity-based preprocessing."""

    DEFAULT_DIAMETER: int = 10
    DEFAULT_CHANNEL: int = -1
    DEFAULT_FLOW_THRESHOLD_3D: float = 0.4
    DEFAULT_CELLPROB_THRESHOLD_3D: float = 0.0
    DEFAULT_MAX_DILATION_STEPS: int = 10
    DEFAULT_BOTTOM_LIMIT: float = 0.1
    DEFAULT_CLIP_LIMIT: float = 20.0
    DEFAULT_TILE_GRID_SIZE: tuple[int, int] = (4, 4)
    MODEL_TYPE: str = "podosomes"

    def __init__(
        self,
        image_array: np.ndarray,
        diameter: int = DEFAULT_DIAMETER,
        channel: int = DEFAULT_CHANNEL,
    ) -> None:
        """Initializes the detector with an image and parameters.

        Args:
            image_array: Input image as a NumPy array (3D, 4D, or 5D).
            diameter: Expected podosome diameter in pixels. Defaults to 10.
            channel: Channel index for detection (0-based, -1 for last). Defaults to -1.
        """
        self._image_tensor = self._extract_channel(image_array, channel)
        self.diameter = diameter
        self._manager: Optional[object] = None

    def detect(
        self,
        flow_threshold_3d: float = DEFAULT_FLOW_THRESHOLD_3D,
        cellprob_threshold_3d: float = DEFAULT_CELLPROB_THRESHOLD_3D,
    ) -> np.ndarray:
        """Detects podosomes in the 3D image using a two-stage Cellpose process.

        Args:
            flow_threshold_3d: Flow threshold for 3D detection. Defaults to 0.4.
            cellprob_threshold_3d: Cell probability threshold for 3D detection. Defaults to 0.0.

        Returns:
            3D mask array with detected podosomes (uint16 dtype).
        """
        image_3d = self._image_tensor
        image_2d = self._find_highest_intensity_slice(image_3d)
        image_2d = self._equalize_image_clahe(image_2d, normalize_output=True)

        # Perform 2D detection
        masks_2d = self._run_cellpose_2d(image_2d)
        tapered_mask = self._taper_out_masks(masks_2d)
        image_3d_normalized = self._adjust_3d_intensity(image_3d, tapered_mask, adjust_mean=False)

        # Perform 3D detection
        masks_3d = self._run_cellpose_3d(image_3d_normalized, flow_threshold_3d, cellprob_threshold_3d)
        return masks_3d

    def _extract_channel(self, image: np.ndarray, channel: int) -> np.ndarray:
        """Extracts a specific channel from the input image.

        Args:
            image: Input image array (3D, 4D, or 5D).
            channel: Channel index to extract (0-based, -1 for last).

        Returns:
            Extracted 3D channel as a NumPy array.

        Notes:
            - For 5D arrays (t, z, c, y, x), extracts from first time point.
            - For 4D arrays (z, c, y, x), extracts specified channel.
            - For other shapes, returns array as-is.
        """
        if image.ndim == 5:
            return image[0, :, channel, :, :]
        if image.ndim == 4:
            return image[:, channel, :, :]
        return image

    def _find_highest_intensity_slice(self, z_stack: np.ndarray) -> np.ndarray:
        """Finds the 2D slice with the highest total intensity in a 3D z-stack.

        Args:
            z_stack: 3D array of 2D image slices.

        Returns:
            2D array with the highest total intensity.
        """
        intensity_sums = np.sum(z_stack, axis=(1, 2))
        max_intensity_idx = np.argmax(intensity_sums)
        return z_stack[max_intensity_idx]

    def _equalize_image_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = DEFAULT_CLIP_LIMIT,
        tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
        normalize_output: bool = False,
    ) -> np.ndarray:
        """Applies CLAHE to enhance image contrast.

        Args:
            image: Input image (2D grayscale or 3D BGR).
            clip_limit: Threshold for contrast limiting. Defaults to 20.0.
            tile_grid_size: Grid size for CLAHE. Defaults to (4, 4).
            normalize_output: If True, normalizes output to [0, 255]. Defaults to False.

        Returns:
            Equalized image as a NumPy array (2D).
        """
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2:
            image = np.squeeze(image)
        image = np.clip(image, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized_image = clahe.apply(image)
        if normalize_output:
            equalized_image = np.where(equalized_image == 0, 0, equalized_image)
            equalized_image = cv2.normalize(equalized_image, None, -21, 255, cv2.NORM_MINMAX)
        return equalized_image

    def _taper_out_masks(
        self,
        mask: np.ndarray,
        max_dilation_steps: int = DEFAULT_MAX_DILATION_STEPS,
        bottom_limit: float = DEFAULT_BOTTOM_LIMIT,
    ) -> np.ndarray:
        """Creates a tapered mask with decreasing values around mask edges.

        Args:
            mask: Input mask array (2D, binary or labeled).
            max_dilation_steps: Number of dilation steps. Defaults to 10.
            bottom_limit: Minimum value for outermost dilation. Defaults to 0.1.

        Returns:
            Tapered mask with values from 1 to bottom_limit.
        """
        mask = mask.astype(bool).astype(float)
        values = np.geomspace(1, bottom_limit, max_dilation_steps + 1)
        tapered_mask = np.copy(mask)
        kernel = np.ones((3, 3), np.uint8)

        for step in range(1, max_dilation_steps + 1):
            dilated_mask = cv2.dilate(tapered_mask, kernel, iterations=1)
            new_mask = np.where(tapered_mask > 0, 0, dilated_mask)
            new_mask = np.where(new_mask > 0, values[step], new_mask)
            tapered_mask = np.maximum(tapered_mask, new_mask)

        return np.where(tapered_mask == 0, 0, tapered_mask)

    def _adjust_3d_intensity(
        self,
        image_3d: np.ndarray,
        tapered_mask: np.ndarray,
        adjust_mean: bool = False,
    ) -> np.ndarray:
        """Adjusts 3D image intensity using a tapered mask.

        Args:
            image_3d: 3D image array.
            tapered_mask: 2D mask for intensity adjustment.
            adjust_mean: If True, scales based on max mean slice intensity. Defaults to False.

        Returns:
            Adjusted 3D image array.
        """
        intensity_mean = np.mean(image_3d, axis=(1, 2))
        if adjust_mean:
            max_mean = np.max(intensity_mean)
            intensity_mean = intensity_mean / max_mean if max_mean != 0 else np.ones_like(intensity_mean)
        else:
            intensity_mean = np.ones_like(intensity_mean)

        original_min, original_max = np.min(tapered_mask), np.max(tapered_mask)
        output_array = np.zeros_like(image_3d)

        for z in range(image_3d.shape[0]):
            new_max = intensity_mean[z]
            mask = np.interp(tapered_mask, [original_min, original_max], [original_min, new_max])
            output_array[z] = image_3d[z] * mask

        return output_array

    def _run_cellpose_2d(self, image_2d: np.ndarray) -> np.ndarray:
        """Runs 2D Cellpose detection on a single slice.

        Args:
            image_2d: 2D image array.

        Returns:
            2D mask array (uint16 dtype).
        """
        detector = CellPoseDetector(
            image_array=image_2d,
            model=self.MODEL_TYPE,
            detection_channel=1,
            diameter=self.diameter,
            do_3d=False,
            cellprob_threshold=-3.0,
            gpu=True,
        )
        return detector.detect()

    def _run_cellpose_3d(
        self,
        image_3d: np.ndarray,
        flow_threshold: float,
        cellprob_threshold: float,
    ) -> np.ndarray:
        """Runs 3D Cellpose detection on the normalized image.

        Args:
            image_3d: 3D image array.
            flow_threshold: Flow threshold for detection.
            cellprob_threshold: Cell probability threshold for detection.

        Returns:
            3D mask array (uint16 dtype).
        """
        detector = CellPoseDetector(
            image_array=image_3d,
            model=self.MODEL_TYPE,
            detection_channel=1,
            diameter=self.diameter,
            do_3d=True,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            gpu=True,
        )
        return detector.detect()

    @property
    def podosomes(self) -> Dict[int, object]:
        """Podosome objects from the manager.

        Returns:
            Dictionary of podosome objects.

        Raises:
            ValueError: If detect() has not been called.
        """
        if self._manager is None:
            raise ValueError("No masks detected. Call `detect` first.")
        return self._manager.get_podosomes()

    @property
    def masks(self) -> np.ndarray:
        """Binary masks for all podosomes.

        Returns:
            Binary mask array (255 for mask, 0 for background, uint8 dtype).

        Raises:
            ValueError: If detect() has not been called.
        """
        if self._manager is None:
            raise ValueError("No masks detected. Call `detect` first.")
        return self._manager.masks

    @property
    def dilated_masks(self) -> np.ndarray:
        """Dilated masks for all podosomes.

        Returns:
            Dilated binary mask array (uint8 dtype).

        Raises:
            ValueError: If detect() has not been called.
        """
        if self._manager is None:
            raise ValueError("No masks detected. Call `detect` first.")
        return self._manager.dilated_masks

    @property
    def label_map(self) -> np.ndarray:
        """Label map for all podosomes.

        Returns:
            3D array with tuples of podosome IDs.

        Raises:
            ValueError: If detect() has not been called.
        """
        if self._manager is None:
            raise ValueError("No masks detected. Call `detect` first.")
        return self._manager.label_map

    @property
    def topographic_map(self) -> np.ndarray:
        """Topographic map for all podosomes.

        Returns:
            3D heatmap array (float32 dtype).

        Raises:
            ValueError: If detect() has not been called.
        """
        if self._manager is None:
            raise ValueError("No masks detected. Call `detect` first.")
        return self._manager.topographic_map


def assign_colors_based_on_volume(podosomes: Dict[int, object]) -> Dict[int, object]:
    """Assigns RGB colors to podosomes based on volume (min: green, max: red).

    Args:
        podosomes: Dictionary of podosome objects with volume and color attributes.

    Returns:
        Modified podosomes dictionary (updated in-place).
    """
    volumes = np.array([p.volume for p in podosomes.values()])
    if not volumes.size:
        return podosomes

    min_vol, max_vol = volumes.min(), volumes.max()
    if min_vol == max_vol:
        for podosome in podosomes.values():
            podosome.color = (255, 0, 0)  # Red for uniform volumes
        return podosomes

    for podosome in podosomes.values():
        norm_value = (podosome.volume - min_vol) / (max_vol - min_vol) if max_vol != min_vol else 1.0
        r = int(255 * norm_value)
        g = int(255 * (1 - norm_value))
        podosome.color = (r, g, 0)

    return podosomes


class SignalDetector:
    """Detects signals in a 3D image stack using thresholding and contour analysis."""

    DEFAULT_CHANNEL: int = 0
    DEFAULT_GAUSSIAN_KSIZE: tuple[int, int] = (3, 3)
    DEFAULT_PERCENTILE: float = 93.0
    DEFAULT_THRESHOLD_FACTOR: float = 0.4
    DEFAULT_INCREMENT: int = 10
    DEFAULT_MIN_CONTOUR_AREA: int = 4
    DEFAULT_MAX_CONTOUR_AREA: int = 30
    DEFAULT_ALLOWED_RADIUS: int = 8
    DEFAULT_CIRCULARITY_THRESHOLD: float = 0.7
    DEFAULT_THRESHOLD: int = 50
    DEFAULT_MIN_RADIUS: int = 5
    DEFAULT_BLUR_ITERATIONS: int = 5
    DEFAULT_MEDIAN_KSIZE: int = 5

    def __init__(self, image_input: np.ndarray, channel: int = DEFAULT_CHANNEL) -> None:
        """Initializes the detector with an image and channel.

        Args:
            image_input: Input image array (4D or 5D).
            channel: Channel index to process (0-based). Defaults to 0.
        """
        self._image_data = image_input
        self._image_tensor = self._extract_channel(channel)
        self._max_projection = np.max(self._image_tensor, axis=0)
        self._avg_projection = np.mean(self._image_tensor, axis=0)
        self._max_value = np.max(self._max_projection)
        self._percentile_threshold = np.percentile(self._max_projection, self.DEFAULT_PERCENTILE)
        self._threshold_intervals = self._generate_threshold_intervals()
        self._thresholded_stack = self._apply_threshold()
        self._max_projection = np.max(self._thresholded_stack, axis=0)
        self._signals: Optional[List[Signal]] = None

    def _extract_channel(self, channel: int) -> np.ndarray:
        """Extracts a specific channel from the input image.

        Args:
            channel: Channel index to extract (0-based).

        Returns:
            3D image tensor for the specified channel.

        Notes:
            - For 5D arrays (t, z, c, y, x), extracts from first time point.
            - For 4D arrays (z, c, y, x), extracts specified channel.
        """
        if self._image_data.ndim == 5:
            return self._image_data[0, :, channel, :, :]
        if self._image_data.ndim == 4:
            return self._image_data[:, channel, :, :]
        return self._image_data

    def _apply_threshold(self) -> np.ndarray:
        """Applies a percentile-based threshold to the image stack.

        Returns:
            Thresholded 3D image stack (values below threshold set to 0).

        Notes:
            - Uses Gaussian blur on maximum projection before thresholding.
            - Threshold is based on the 93rd percentile of the maximum projection.
        """
        threshold = self._percentile_threshold
        max_projection = cv2.GaussianBlur(self._max_projection, self.DEFAULT_GAUSSIAN_KSIZE, 0)
        thresholded_image = np.where(max_projection <= threshold, 0, 255).astype(np.uint8)
        assert self._image_tensor.shape[1:] == thresholded_image.shape, "Shape mismatch"
        masked_stack = self._image_tensor.copy()
        for i in range(masked_stack.shape[0]):
            masked_stack[i, thresholded_image == 0] = 0
        return masked_stack

    def _generate_threshold_intervals(self) -> List[Tuple[int, int]]:
        """Generates thresholding intervals based on the maximum projection value.

        Returns:
            List of tuples representing threshold intervals (start, max_value).

        Notes:
            - Starts at 40% of max value, increments by 10.
            - Last interval ends at max value.
        """
        start_point = int(self._max_value * self.DEFAULT_THRESHOLD_FACTOR)
        intervals = []
        while start_point < self._max_value:
            intervals.append((start_point, self._max_value))
            start_point += self.DEFAULT_INCREMENT
        return intervals

    def _analyze_contour_region(
        self,
        contour: np.ndarray,
        image: np.ndarray,
        adjust_min_radius: Optional[int] = None,
    ) -> Signal:
        """Analyzes a contour region to compute signal properties.

        Args:
            contour: Contour array from cv2.findContours.
            image: Input image for intensity analysis.
            adjust_min_radius: Minimum radius to enforce. Defaults to None.

        Returns:
            Signal object with computed properties.
        """
        def largest_inscribed_circle(contour: np.ndarray) -> Tuple[Tuple[int, int], float]:
            hull = cv2.convexHull(contour)
            center, radius = (0, 0), 0
            for i in range(len(hull)):
                hull_point = tuple(hull[i][0])
                distances = np.linalg.norm(contour - hull_point, axis=2)
                max_distance = np.max(distances)
                if max_distance > radius:
                    radius = max_distance
                    center = hull_point
            return center, radius

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        (x_inside, y_inside), radius_inside = largest_inscribed_circle(contour)

        if adjust_min_radius is not None and radius < adjust_min_radius:
            radius = adjust_min_radius

        circle_area = np.pi * radius ** 2
        area_ratio = area / circle_area if circle_area != 0 else 0
        circumference = 2 * np.pi * radius
        perimeter_ratio = perimeter / circumference if circumference != 0 else 0

        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        circle_region = cv2.bitwise_and(image, mask)
        blurred_region = cv2.GaussianBlur(circle_region, (5, 5), 0)
        local_max = cv2.dilate(blurred_region, np.ones((3, 3), np.uint8))
        local_maxima = np.where((blurred_region == local_max) & (blurred_region > 0))
        local_maxima = list(zip(local_maxima[1], local_maxima[0]))

        return Signal(
            center=center,
            radius=radius,
            center_inside=(int(x_inside), int(y_inside)),
            radius_inside=int(radius_inside),
            minor=radius,
            area_contour=area,
            area_circle=circle_area,
            area_ratio=area_ratio,
            perimeter=perimeter,
            perimeter_ratio=perimeter_ratio,
            orientation=None,
            coordinates=contour,
            local_maxima=local_maxima,
            is_valid=True,
        )

    def _validate_regions(
        self,
        regions: List[Signal],
        circularity_threshold: Optional[float] = None,
        allowed_radius: Optional[int] = None,
        strict: bool = False,
        check_local_maxima: bool = True,
        exclude_pinched_signals: bool = False,
    ) -> List[Signal]:
        """Validates signal regions based on specified criteria.

        Args:
            regions: List of Signal objects to validate.
            circularity_threshold: Minimum area/perimeter ratio. Defaults to None.
            allowed_radius: Maximum allowed radius. Defaults to None.
            strict: If True, only valid regions are kept. Defaults to False.
            check_local_maxima: If True, checks for single local maximum. Defaults to True.
            exclude_pinched_signals: If True, excludes non-convex contours. Defaults to False.

        Returns:
            Filtered list of Signal objects.
        """
        validated = []
        for region in regions:
            if allowed_radius is not None and region.radius >= allowed_radius:
                region.is_valid = False
            if circularity_threshold is not None:
                if region.area_ratio <= circularity_threshold or region.perimeter_ratio <= circularity_threshold:
                    region.is_valid = False
            if exclude_pinched_signals:
                epsilon = 0.04 * cv2.arcLength(region.coordinates, True)
                approx = cv2.approxPolyDP(region.coordinates, epsilon, True)
                region.is_valid = not cv2.isContourConvex(approx)
            if check_local_maxima and len(region.local_maxima) > 1:
                region.is_valid = False
            if not strict or region.is_valid:
                validated.append(region)
        return validated

    def _extract_signals(
        self,
        image: np.ndarray,
        min_contour_area: int = DEFAULT_MIN_CONTOUR_AREA,
        max_contour_area: int = DEFAULT_MAX_CONTOUR_AREA,
        circularity_threshold: Optional[float] = None,
        color: Tuple[int, int, int] = (255, 255, 255),
        threshold: int = 0,
        allowed_radius: int = DEFAULT_ALLOWED_RADIUS,
        adjust_min_radius: Optional[int] = None,
        exclude_pinched_signals: bool = True,
        check_local_maxima: bool = True,
        strict: bool = True,
    ) -> List[Signal]:
        """Extracts signal regions from a binary image.

        Args:
            image: 2D binary or grayscale image.
            min_contour_area: Minimum contour area. Defaults to 4.
            max_contour_area: Maximum contour area. Defaults to 30.
            circularity_threshold: Minimum circularity ratio. Defaults to None.
            color: RGB color for signals. Defaults to (255, 255, 255).
            threshold: Threshold for binarization. Defaults to 0.
            allowed_radius: Maximum allowed radius. Defaults to 8.
            adjust_min_radius: Enforced minimum radius. Defaults to None.
            exclude_pinched_signals: If True, excludes non-convex contours. Defaults to True.
            check_local_maxima: If True, checks for single local maximum. Defaults to True.
            strict: If True, only valid regions are kept. Defaults to True.

        Returns:
            List of Signal objects.
        """
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
        signals = [
            self._analyze_contour_region(cnt, self._avg_projection, adjust_min_radius)
            for cnt in contours
        ]
        signals = self._validate_regions(
            signals,
            circularity_threshold=circularity_threshold,
            allowed_radius=allowed_radius,
            exclude_pinched_signals=exclude_pinched_signals,
            check_local_maxima=check_local_maxima,
            strict=strict,
        )
        for signal in signals:
            signal.circle_color = color
        return signals

    def _apply_mask(
        self,
        image: np.ndarray,
        regions: List[Signal],
        threshold: int = 0,
        use_circle: bool = True,
    ) -> np.ndarray:
        """Applies a mask to an image based on signal regions.

        Args:
            image: Input image (2D or 3D).
            regions: List of Signal objects for masking.
            threshold: Threshold for masked regions when using circles. Defaults to 0.
            use_circle: If True, uses circular masks; otherwise, uses contours. Defaults to True.

        Returns:
            Masked image array.
        """
        masked_image = image.copy()
        if masked_image.ndim == 2:
            masked_image = np.expand_dims(masked_image, axis=0)
        depth, height, width = masked_image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        for region in regions:
            if use_circle:
                x_center, y_center = region.center
                radius = max(int(region.radius + 5), self.DEFAULT_MIN_RADIUS)
                y, x = np.ogrid[:height, :width]
                circle_mask = (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2
                mask |= circle_mask.astype(np.uint8)
            else:
                cv2.drawContours(mask, [region.coordinates], -1, 255, thickness=cv2.FILLED)

        if use_circle:
            for z in range(depth):
                masked_image[z][mask.astype(bool)] = 0
            masked_image = np.squeeze(masked_image)
            masked_image[masked_image < threshold] = 0
        else:
            masked_image = cv2.bitwise_and(image, mask)

        return masked_image

    def _find_peaks(self, image: np.ndarray) -> List[Signal]:
        """Detects peak signals using local maxima.

        Args:
            image: 2D image array.

        Returns:
            List of Signal objects based on peak locations.
        """
        local_max = peak_local_max(image, min_distance=5, threshold_abs=0, exclude_border=False)
        return [
            Signal(
                center=(int(x), int(y)),
                radius=5,
                center_inside=(int(x), int(y)),
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
                circle_color=(127, 127, 255),
            )
            for y, x in local_max
        ]

    def _iteratively_blur(self, image: np.ndarray, iterations: int = DEFAULT_BLUR_ITERATIONS) -> np.ndarray:
        """Applies iterative Gaussian blur to an image.

        Args:
            image: Input image array.
            iterations: Number of blur iterations. Defaults to 5.

        Returns:
            Blurred image array.
        """
        blurred = image.copy()
        for _ in range(iterations):
            blurred = cv2.GaussianBlur(blurred, self.DEFAULT_GAUSSIAN_KSIZE, 0)
        return blurred

    def _filter_circles(
        self,
        circles: List[Signal],
        max_radius: Optional[int] = None,
        min_radius: Optional[int] = None,
    ) -> List[Signal]:
        """Filters circles based on radius and overlap.

        Args:
            circles: List of Signal objects representing circles.
            max_radius: Maximum allowed radius. Defaults to None.
            min_radius: Minimum allowed radius. Defaults to None.

        Returns:
            Filtered list of non-overlapping Signal objects.
        """
        def is_inside(circle1: Signal, circle2: Signal) -> bool:
            x1, y1 = circle1.center
            x2, y2 = circle2.center
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return distance <= circle1.radius

        if max_radius is not None or min_radius is not None:
            size_restricted = [
                c for c in circles
                if (max_radius is None or c.radius <= max_radius) and
                   (min_radius is None or c.radius > min_radius)
            ]
            circles = size_restricted

        unique_centers = set()
        filtered = []
        for circle in circles:
            if circle.center not in unique_centers:
                unique_centers.add(circle.center)
                filtered.append(circle)

        filtered.sort(key=lambda c: c.radius, reverse=True)
        solitary = []
        others = copy.deepcopy(filtered)

        for circle in filtered:
            others.pop(0)
            is_solitary = all(not is_inside(circle, other) for other in others)
            if is_solitary:
                solitary.append(circle)

        return solitary

    def _generate_interval_thresholded_images(self, image: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Generates binary images using threshold intervals.

        Args:
            image: 2D input image.

        Yields:
            Binary thresholded images (uint8).
        """
        blurred = cv2.GaussianBlur(image, self.DEFAULT_GAUSSIAN_KSIZE, 0)
        for lower, upper in self._threshold_intervals:
            binary = np.where((blurred > upper) | (blurred < lower), 0, 255).astype(np.uint8)
            _, thresholded = cv2.threshold(binary, 254, 255, cv2.THRESH_BINARY)
            yield thresholded

    def _detect_signals(
        self,
        image: np.ndarray,
        min_contour_area: int = DEFAULT_MIN_CONTOUR_AREA,
        circularity_threshold: Optional[float] = None,
        color: Tuple[int, int, int] = (255, 255, 255),
        threshold: int = 0,
        allowed_radius: int = DEFAULT_ALLOWED_RADIUS,
        adjust_min_radius: Optional[int] = None,
        exclude_pinched_signals: bool = True,
        check_local_maxima: bool = True,
        strict: bool = True,
        use_threshold_intervals: bool = False,
        use_peak_detection_only: bool = False,
        signals: Optional[List[Signal]] = None,
    ) -> List[Signal]:
        """Detects signals from an image using contour or peak analysis.

        Args:
            image: Input image (2D or 3D).
            min_contour_area: Minimum contour area. Defaults to 4.
            circularity_threshold: Minimum circularity ratio. Defaults to None.
            color: RGB color for signals. Defaults to (255, 255, 255).
            threshold: Threshold for binarization. Defaults to 0.
            allowed_radius: Maximum allowed radius. Defaults to 8.
            adjust_min_radius: Enforced minimum radius. Defaults to None.
            exclude_pinched_signals: If True, excludes non-convex contours. Defaults to True.
            check_local_maxima: If True, checks for single local maximum. Defaults to True.
            strict: If True, only valid regions are kept. Defaults to True.
            use_threshold_intervals: If True, uses threshold intervals. Defaults to False.
            use_peak_detection_only: If True, uses peak detection only. Defaults to False.
            signals: Existing signals to extend. Defaults to None.

        Returns:
            List of detected Signal objects.
        """
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        depth = image.shape[0]
        all_signals = list(signals) if signals else []

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
        }

        for i in range(depth):
            if use_peak_detection_only:
                all_signals.extend(self._find_peaks(image[i]))
            elif use_threshold_intervals:
                for binary_image in self._generate_interval_thresholded_images(image[i]):
                    all_signals.extend(self._extract_signals(binary_image, **params))
            else:
                all_signals.extend(self._extract_signals(image[i], **params))

        return all_signals

    def _find_max_intensity_z_with_circle(
        self,
        signals: List[Signal],
        image_stack: np.ndarray,
        radius: int = DEFAULT_MIN_RADIUS,
    ) -> List[Signal]:
        """Assigns z-coordinates to signals based on maximum intensity within a circular region.

        Args:
            signals: List of Signal objects with x, y coordinates.
            image_stack: 3D image stack (z, y, x).
            radius: Radius for intensity summation. Defaults to 5.

        Returns:
            Signals with updated z-coordinates.
        """
        z_size, y_size, x_size = image_stack.shape
        diameter = radius * 2 + 1
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        circular_mask = (x ** 2 + y ** 2) <= radius ** 2
        mask_size = circular_mask.shape

        for signal in signals:
            x_center, y_center = signal.center
            max_intensity_sum = -np.inf
            best_z = 0

            for z in range(z_size):
                x_min = max(0, x_center - radius)
                x_max = min(x_size, x_center + radius + 1)
                y_min = max(0, y_center - radius)
                y_max = min(y_size, y_center + radius + 1)
                roi = image_stack[z, y_min:y_max, x_min:x_max]
                adjusted_mask = circular_mask[:roi.shape[0], :roi.shape[1]] if roi.shape != mask_size else circular_mask
                masked_roi = np.where(adjusted_mask, roi, 0)
                intensity_sum = np.sum(masked_roi)
                if intensity_sum > max_intensity_sum:
                    max_intensity_sum = intensity_sum
                    best_z = z
            signal.z = best_z

        return signals

    def detect(self) -> List[Signal]:
        """Detects signals through iterative thresholding, masking, and peak detection.

        Returns:
            List of detected Signal objects.
        """
        signals = []
        max_projection = self._max_projection
        image_stack = self._thresholded_stack

        # Initial detection
        initial_signals = self._detect_signals(
            max_projection,
            color=(255, 0, 255),
            threshold=self._percentile_threshold,
            circularity_threshold=self.DEFAULT_CIRCULARITY_THRESHOLD,
        )
        masked_max = self._apply_mask(max_projection, initial_signals)
        masked_stack = self._apply_mask(image_stack, initial_signals)
        signals.extend(initial_signals)

        # Stricter detection
        stricter_signals = self._detect_signals(
            masked_max,
            color=(255, 255, 0),
            threshold=int(self._max_value - 5),
            strict=True,
            min_contour_area=2,
            circularity_threshold=self.DEFAULT_CIRCULARITY_THRESHOLD,
        )
        masked_max_strict = self._apply_mask(masked_max, stricter_signals)
        masked_stack_strict = self._apply_mask(masked_stack, stricter_signals)
        signals.extend(stricter_signals)

        # Blurred detection
        blurred_max = self._iteratively_blur(masked_max_strict)
        blurred_signals = self._detect_signals(
            blurred_max,
            color=(0, 255, 255),
            threshold=self.DEFAULT_THRESHOLD,
            strict=True,
            min_contour_area=2,
            circularity_threshold=self.DEFAULT_CIRCULARITY_THRESHOLD,
        )
        masked_max_blurred = self._apply_mask(blurred_max, blurred_signals, threshold=self.DEFAULT_THRESHOLD)
        masked_stack_blurred = self._apply_mask(masked_stack_strict, blurred_signals, threshold=self.DEFAULT_THRESHOLD)
        signals.extend(blurred_signals)

        # Threshold interval detection
        layer_signals = self._detect_signals(
            masked_stack_blurred,
            adjust_min_radius=self.DEFAULT_MIN_RADIUS,
            color=(0, 255, 0),
            allowed_radius=None,
            exclude_pinched_signals=False,
            use_threshold_intervals=True,
        )
        filtered_layer_signals = self._filter_circles(layer_signals)
        masked_max_filtered = self._apply_mask(masked_max_blurred, filtered_layer_signals, threshold=self.DEFAULT_THRESHOLD)
        masked_stack_filtered = self._apply_mask(masked_stack_blurred, filtered_layer_signals, threshold=self.DEFAULT_THRESHOLD)
        signals.extend(filtered_layer_signals)

        # Non-strict detection
        binarized_max = np.where(masked_max_filtered != 0, 255, masked_max_filtered)
        median_blurred_max = cv2.medianBlur(binarized_max.astype(np.uint8), self.DEFAULT_MEDIAN_KSIZE)
        non_strict_signals = self._detect_signals(
            median_blurred_max,
            color=(0, 0, 255),
            threshold=self.DEFAULT_THRESHOLD,
            strict=False,
            min_contour_area=1,
            circularity_threshold=self.DEFAULT_CIRCULARITY_THRESHOLD,
        )
        filtered_non_strict_exempt = self._filter_circles(non_strict_signals, min_radius=10)
        filtered_non_strict = self._filter_circles(non_strict_signals, max_radius=10)
        signals.extend(filtered_non_strict)

        # Peak detection on average projection
        masked_avg = self._apply_mask(self._avg_projection, filtered_non_strict_exempt, use_circle=False)
        blurred_avg = self._iteratively_blur(masked_avg)
        peak_signals = self._detect_signals(blurred_avg, use_peak_detection_only=True)
        signals.extend(peak_signals)

        # Assign z-coordinates
        final_signals = self._find_max_intensity_z_with_circle(signals, self._image_tensor)
        self._signals = final_signals
        return final_signals
    
if __name__ == "__main__":

    pass
        
        
