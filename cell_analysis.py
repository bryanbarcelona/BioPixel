from typing import Any, List, Tuple, Dict, Optional
import cv2
import numpy as np
from numba import jit, prange
from data_structures import PodosomeCellResult, Signal, Podosome
from detectors import PodosomeDetector, PodosomeManager, SignalDetector

class PLACellAnalyzer:
    """Analyzes Proximity Ligation Assay (PLA) signals in cellular microscopy data.  

    PLA detects protein-protein interactions as fluorescent puncta. This class:  
    - Identifies PLA signals in a specified channel.  
    - Maps signals to podosomes (if detected in another channel).  
    - Quantifies colocalization and spatial statistics.  

    Example:  
        >>> analyzer = PLACellAnalyzer(image_3d, podosome_channel=1, signal_channel=2)  
        >>> analyzer.run()  
        >>> print(analyzer.podosome_associated_count)  
    """

    def __init__(
        self,
        image_data: np.ndarray,
        control: bool = False,
        podosome_channel: int = -1,
        signal_channel: int = 0
    ) -> None:
        """Initializes the analyzer with raw image data and channel settings.  

        Args:  
            image_data: 3D array (Z, Y, X) of multichannel microscopy data.  
            control: If True, marks signals as control samples (default: False).  
            podosome_channel: Channel for podosome detection (-1 = auto-select).  
            signal_channel: Channel for PLA signal detection (default: 0).  
        """
        self.image_data = image_data
        self.control = control
        self.podosome_channel = podosome_channel
        self.signal_channel = signal_channel
        
        self._signals: List[Signal] = []
        self.individual_podosomes: Dict[int, Podosome] = {}
        self.dilated_podosome_mask: Optional[np.ndarray] = None
        self.label_map: Optional[np.ndarray] = None
        self.topographic_map: Optional[np.ndarray] = None

    def run(self) -> None:
        """Run the complete analysis pipeline."""
        self._detect_podosomes()
        self._detect_signals()
        self._assign_control_status()
        self._map_signals_to_podosomes()

    def _detect_podosomes(self) -> None:
        """Detect podosomes in the image data and initialize related properties."""
        podosome_mask = PodosomeDetector(self.image_data, channel=self.podosome_channel).detect()
        podosome_manager = PodosomeManager(
            mask_3d=podosome_mask,
            size_um=1,
            resolution_xy=5,
            resolution_z=1
        )
        self.individual_podosomes = podosome_manager.cell_result.podosomes
        self.dilated_podosome_mask = podosome_manager.dilated_masks
        self.label_map = podosome_manager.label_map
        self.topographic_map = podosome_manager.topographic_map

    def _detect_signals(self) -> None:
        """Detect signals in the specified channel."""
        self._signals = SignalDetector(self.image_data, channel=self.signal_channel).detect()

    @property
    def podosome_associated_count(self) -> int:
        """Count of signals associated with podosomes."""
        return sum(signal.podosome_associated for signal in self._signals)

    @property
    def non_podosome_associated_count(self) -> int:
        """Count of signals not associated with podosomes."""
        return len(self._signals) - self.podosome_associated_count

    @property
    def podosome_associated_signals(self) -> List[Signal]:
        """List of signals associated with podosomes."""
        return [signal for signal in self._signals if signal.podosome_associated]
    
    @property
    def signals(self) -> List[Signal]:
        """List of all detected signals."""
        return self._signals

    def _assign_control_status(self) -> None:
        """Assign control status to all detected signals."""
        for signal in self._signals:
            signal.is_control = self.control

    def _map_signals_to_podosomes(self) -> None:
        """Map each signal to its closest podosome and assign spatial properties."""
        self._assign_podosome_association()
        self._assign_closest_podosome()
        self._assign_signal_heights()

    def _assign_podosome_association(self) -> None:
        """Determine which signals are associated with podosomes based on spatial overlap."""
        if self.label_map is None:
            return

        depth, height, width = self.label_map.shape

        for signal in self._signals:
            x, y = signal.center
            z = signal.z
            
            if 0 <= z < depth and 0 <= y < height and 0 <= x < width:
                signal.podosome_associated = self.label_map[z, y, x] is not None
            else:
                signal.podosome_associated = False

            signal.podosome_label = self.label_map[z, y, x] if signal.podosome_associated else None

    def _assign_closest_podosome(self) -> None:
        """Find the closest podosome for each associated signal and calculate distance."""
        for signal in self._signals:
            if not signal.podosome_associated or not signal.podosome_label:
                continue

            signal_x, signal_y, signal_z = signal.center[0], signal.center[1], signal.z
            podosome_ids = signal.podosome_label

            min_distance = float("inf")
            closest_podosome_id = None

            for podosome_id in podosome_ids:
                podosome = self.individual_podosomes.get(podosome_id)
                if podosome is None:
                    continue

                distance = np.linalg.norm(np.array(signal.center + (signal_z,)) - 
                          np.array(podosome.centroid))
                if distance < min_distance:
                    min_distance = distance
                    closest_podosome_id = podosome_id

            if closest_podosome_id is not None:
                signal.podosome_label = closest_podosome_id 
                signal.distance_to_podosome = min_distance
            else:
                signal.podosome_associated = False

    def _assign_signal_heights(self) -> None:
        """Assign relative height values from the topographic map to each signal."""
        if self.topographic_map is None:
            return

        for signal in self._signals:
            x, y = signal.center
            z = signal.z
            
            if (0 <= z < self.topographic_map.shape[0] and 
                0 <= y < self.topographic_map.shape[1] and 
                0 <= x < self.topographic_map.shape[2]):
                signal.relative_signal_height = self.topographic_map[z, y, x]
            else:
                signal.relative_signal_height = None

class PodosomeProfileAnalyzer:
    """Analyzes radial intensity profiles of podosomes from 3D microscopy data.

    This class extracts and processes radial intensity profiles around podosome centroids
    across multiple z-slices, providing averaged profile data for quantitative analysis.

    Attributes:
        image_data: 4D numpy array (Z, Y, X, Channels) of microscopy data
        podosome_cell: Processed PodosomeCellResult object containing podosome data
        channel_index: Index of channel to analyze (default: 0)
        podosome_count: Number of podosomes detected
        radial_profiles_count: Total number of radial profiles generated
    """

    def __init__(
        self,
        image_data: np.ndarray,
        podosome_cell: 'PodosomeCellResult',
        channel_index: int = 0
    ) -> None:
        """Initializes the analyzer with image data and podosome information.

        Args:
            image_data: 4D numpy array (Z, Y, X, Channels) of microscopy data
            podosome_cell: PodosomeCellResult object containing podosome coordinates
            channel_index: Channel index to analyze (default: 0)
        """
        self.image_data = image_data
        self.podosome_cell = podosome_cell
        self.channel_index = channel_index
        self.podosome_count: int = len(podosome_cell.podosomes)
        self.radial_profiles_count: int = 0

    def analyze_podosomes(self) -> np.ndarray:
        """Analyzes all podosomes and returns averaged radial intensity profiles.

        Processes each podosome to extract radial profiles, then averages them.

        Returns:
            np.ndarray: Averaged radial intensity profiles (shape: [num_samples,])
        
        Note:
            Skips podosomes without valid contours and logs warnings
        """
        channel_data = self.extract_channel(self.channel_index)
        profiles = []

        for podosome_id, podosome in self.podosome_cell.podosomes.items():
            profile = self._process_single_podosome(podosome, channel_data)
            if profile is not None:
                profiles.append(profile)

        return np.mean(np.array(profiles), axis=0) if profiles else np.array([])

    def extract_channel(self, channel_index: int) -> np.ndarray:
        """Extracts specified channel from 4D image data.

        Args:
            channel_index: Index of channel to extract

        Returns:
            np.ndarray: 3D array (Z, Y, X) of single-channel data
        """
        return self.image_data[..., channel_index]

    def _process_single_podosome(
        self,
        podosome: 'Podosome',
        channel_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """Processes a single podosome to extract radial profiles.

        Args:
            podosome: Podosome object containing coordinate data
            channel_data: 3D array (Z, Y, X) of single-channel image data

        Returns:
            Optional[np.ndarray]: Interpolated radial profiles or None if invalid
        """
        contour = self._create_master_contour(podosome, channel_data.shape)
        if contour is None:
            return None

        profiles = self._get_all_z_profiles(
            channel_data,
            contour,
            podosome.centroid
        )
        profiles = self._filter_empty_slices(profiles, podosome)
        self.radial_profiles_count += profiles.shape[0] * 360
        return self._interpolate_z(profiles)

    def _create_master_contour(
        self,
        podosome: 'Podosome',
        image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Creates a 2D contour from podosome slices across z-stack.

        Args:
            podosome: Podosome object with dilated_slices data
            image_shape: Shape of the target image (Z, Y, X)

        Returns:
            Optional[np.ndarray]: Largest contour or None if none found
        """
        height, width = image_shape[1], image_shape[2]
        mask = np.zeros((height, width), dtype="uint8")

        for slice_data in podosome.dilated_slices.values():
            if not slice_data.get('pixels'):
                continue
            for y, x in slice_data['pixels']:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print(f"Warning: No contours found for podosome {podosome.id}. Skipping.")
            return None

        return max(contours, key=cv2.contourArea)

    def _get_all_z_profiles(
        self,
        image_data: np.ndarray,
        contour: np.ndarray,
        centroid: Tuple[float, float],
        num_directions: int = 360,
        num_samples: int = 100
    ) -> np.ndarray:
        """Extracts radial profiles from all z-slices for a podosome.

        Args:
            image_data: 3D array (Z, Y, X) of image data
            contour: Podosome contour points
            centroid: (x,y) coordinates of podosome center
            num_directions: Number of angular directions (default: 360)
            num_samples: Number of radial samples (default: 100)

        Returns:
            np.ndarray: Array of radial profiles (shape: [z_slices, num_directions, num_samples])
        """
        all_profiles = []
        for z in range(image_data.shape[0]):
            profiles = self._get_slice_profiles(
                contour,
                centroid,
                image_data,
                z,
                num_directions,
                num_samples
            )
            all_profiles.append(profiles)
        return np.stack(all_profiles)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _get_slice_profiles(
        contour: np.ndarray,
        centroid: Tuple[float, float],
        image_data: np.ndarray,
        z: int,
        num_directions: int,
        num_samples: int
    ) -> np.ndarray:
        """Numba-optimized radial profile extraction for single z-slice.

        Args:
            contour: Podosome contour points
            centroid: (x,y) coordinates of podosome center
            image_data: 3D array (Z, Y, X) of image data
            z: Current z-slice index
            num_directions: Number of angular directions
            num_samples: Number of radial samples

        Returns:
            np.ndarray: Radial profiles (shape: [num_directions, num_samples])
        """
        height, width = image_data.shape[1], image_data.shape[2]
        profiles = np.zeros((num_directions, num_samples), dtype=np.float64)
        distances = np.array([
            np.sqrt((p[0][0] - centroid[0])**2 + (p[0][1] - centroid[1])**2)
            for p in contour
        ])
        max_dist = np.min(distances) if len(distances) > 0 else 0

        if max_dist <= 0:
            return profiles

        old_dists = np.linspace(0, max_dist, num_samples)

        for angle in prange(num_directions):
            theta = np.radians(angle)
            dx, dy = np.cos(theta), np.sin(theta)
            radial_values = np.zeros(num_samples)

            for j in prange(num_samples):
                dist = (j / (num_samples - 1)) * max_dist
                x = int(centroid[0] + dist * dx)
                y = int(centroid[1] + dist * dy)
                if 0 <= x < width and 0 <= y < height:
                    radial_values[j] = image_data[z, y, x]

            if np.any(radial_values):
                profiles[angle, :] = np.interp(
                    old_dists,
                    np.linspace(0, max_dist, len(radial_values)),
                    radial_values
                )

        return profiles

    def _filter_empty_slices(
        self,
        profiles: np.ndarray,
        podosome: 'Podosome'
    ) -> np.ndarray:
        """Filters out empty z-slices from profile data.

        Args:
            profiles: Array of radial profiles
            podosome: Podosome object with slice data

        Returns:
            np.ndarray: Filtered profiles with empty slices removed
        """
        empty_keys = [
            k for k, v in podosome.dilated_slices.items()
            if not v["pixels"]
        ]
        mask = ~np.isin(np.arange(profiles.shape[0]), empty_keys)
        return profiles[mask]

    def _interpolate_z(
        self,
        data: np.ndarray,
        target_z: int = 100
    ) -> np.ndarray:
        """Interpolates profile data to target z-resolution.

        Args:
            data: Input profile data
            target_z: Desired number of z-slices (default: 100)

        Returns:
            np.ndarray: Interpolated profiles (shape: [target_z, num_directions])
        """
        original_z = data.shape[0]
        new_z = np.linspace(0, original_z-1, target_z)
        return np.array([
            np.interp(new_z, np.arange(original_z), data[:, i])
            for i in range(data.shape[1])
        ]).T
    
if __name__ == "__main__":

    pass
