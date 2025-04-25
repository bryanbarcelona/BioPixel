#from cell_detector import PodosomeDetector, SignalDetector
from detectors import SignalDetector, PodosomeDetector, PodosomeManager
from data_structures import Signal, PLACellResult, PodosomeCellResult
from image_tensors import ImageReader
import numpy as np
from typing import List, Dict, Any, Tuple
from visualization import Visualization
from numba import jit, prange
import cv2

class PLACellAnalyzer:
    def __init__(self, image_data: np.ndarray, control: bool = False, podosome_channel: int = -1, signal_channel: int = 0):
        self.image_data = image_data
        self.control = control
        self.podosome_channel = podosome_channel
        self.signal_channel = signal_channel
        
        self._signals = []
        self.individual_podosomes = {}
        self.dilated_podosome_mask = None
        self.label_map = None

    def run(self):
        self._detect_podosomes()
        self._detect_signals()
        self._assign_control_status()
        self._map_signals_to_podosomes()

    # def _detect_podosomes(self):
    #     detector = PodosomeDetector(self.image_data, channel=self.podosome_channel)
    #     detector.detect()
    #     detector._manager.filter_podosomes()
    #     self.podosomes = detector
    #     self.individual_podosomes = detector.podosomes
    #     self.dilated_podosome_mask = detector.dilated_masks
    #     self.label_map = detector.label_map

    def _detect_podosomes(self):
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

    def _detect_signals(self):
        self._signals = SignalDetector(self.image_data, channel=self.signal_channel).detect()

    @property
    def podosome_associated_count(self) -> int:
        return sum(1 for signal in self._signals if signal.podosome_associated)

    @property
    def non_podosome_associated_count(self) -> int:
        return sum(1 for signal in self._signals if not signal.podosome_associated)

    @property
    def podosome_associated_signals(self) -> List[Signal]:
        return [signal for signal in self._signals if signal.podosome_associated]
    
    @property
    def signals(self) -> List[Signal]:
        return self._signals

    def _assign_control_status(self):
        """Assign control status to signals."""
        for signal in self._signals:
            signal.is_control = self.control

    def _map_signals_to_podosomes(self):
        """
        Maps each signal to its closest podosome based on centroid distance.
        """
        self._assign_podosome_association()
        self._assign_closest_podosome()
        self._assign_signal_heights()

    def _assign_podosome_association(self):
        """
        Validates if signals are inside a binary 3D mask.
        
        Parameters:
            mask (np.ndarray): A 3D binary mask with shape (Z, H, W), where 0 is background and 255 is mask.
            signals (list): List of dictionaries containing:
                            - 'Center': tuple (x, y) for the signal's coordinates.
                            - 'z': Integer for the z-layer index.
        
        Returns:
            list: Updated list of signals with 'PodosomeAssociated' key as a boolean.
        """
        depth, height, width = self.label_map.shape

        for signal in self._signals:
            x, y = signal.center
            z = signal.z
            
            if 0 <= z < depth and 0 <= y < height and 0 <= x < width:
                signal.podosome_associated = self.label_map[z, y, x] is not None
            else:
                signal.podosome_associated = False

            if signal.podosome_associated:
                signal.podosome_label = self.label_map[z, y, x]
            else:
                signal.podosome_label = None

    def _assign_closest_podosome(self):
        """
        Updates self._signals by assigning each podosome-associated signal to the closest podosome
        based on centroid distance. Stores the closest podosome ID and the computed distance.
        """
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

                podo_x, podo_y, podo_z = podosome.centroid
                distance = np.sqrt((signal_x - podo_x) ** 2 + (signal_y - podo_y) ** 2 + (signal_z - podo_z) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_podosome_id = podosome_id

            if closest_podosome_id is not None:
                signal.podosome_label = closest_podosome_id 
                signal.distance_to_podosome = min_distance
            else:
                print(f"Irregular Behavior: No valid podosome found for signal at X: {signal_x}, Y: {signal_y}, Z: {signal_z}")
                signal.podosome_associated = False

    def _assign_signal_heights(self):
        """
        Assigns the relative signal height from a 3D topographic map to each signal based on its coordinates.
        
        Parameters:
            topographic_map (np.ndarray): A 3D array with shape (Z, H, W) containing relative height values.
            signals (list): List of dictionaries containing:
                            - 'Center': tuple (x, y) for the signal's coordinates.
                            - 'z': Integer for the z-layer index.
        
        Returns:
            list: Updated list of signals with 'RelativeSignalHeight' key added.
        """

        topography = self.topographic_map

        for signal in self._signals:
            x, y = signal.center
            z = signal.z
            
            if 0 <= z < topography.shape[0] and 0 <= y < topography.shape[1] and 0 <= x < topography.shape[2]:
                signal.relative_signal_height = topography[z, y, x]
            else:
                signal.relative_signal_height = None

class PodosomeProfileAnalyzer:
    """
    Analyzes radial intensity profiles of podosomes from microscopy data.
    
    Attributes:
        image_data: 3D microscopy data (Z, Y, X, Channels).
        podosome_cell: Processed PodosomeCell object containing podosome data.
        channel_index: Index of the channel to analyze.
    """
    def __init__(self, image_data: np.ndarray, podosome_cell: PodosomeCellResult, channel_index: int = 0):
        self.image_data = image_data
        self.podosome_cell = podosome_cell
        self.channel_index = channel_index
        self.podosome_count: int = len(podosome_cell.podosomes)
        self.radial_profiles_count: int = 0

    def analyze_podosomes(self) -> np.ndarray:
        """
        Analyze all podosomes in the specified channel data.
        
        Returns:
            Averaged radial intensity profiles across all podosomes.
        """
        channel_data = self.extract_channel(self.channel_index)
        print(channel_data.shape, channel_data.dtype, type(channel_data))
        profiles = []
        #for idx, podosome in enumerate(self.podosome_cell.podosomes):
        for podosome_id, podosome in self.podosome_cell.podosomes.items():
            # if idx % 10 == 0:
            #     print(f"Processing podosome {idx}...")
            profile = self._process_single_podosome(podosome, channel_data)
            if profile is not None:
                profiles.append(profile)

        return np.mean(np.array(profiles), axis=0)

    def extract_channel(self, channel_index: int) -> np.ndarray:
        return self.image_data[0, :, channel_index, :, :]

    def _process_single_podosome(self, podosome, channel_data: np.ndarray) -> np.ndarray:
        contour = self._create_master_contour(podosome, channel_data.shape)
        if contour is None:
            return None
        profiles = self._get_all_z_profiles(channel_data, contour, podosome.centroid)
        profiles = self._filter_empty_slices(profiles, podosome)
        self.radial_profiles_count += profiles.shape[0] * 360
        return self._interpolate_z(profiles)

    def _create_master_contour(self, podosome, image_shape: Tuple) -> np.ndarray:
        height, width = image_shape[1], image_shape[2]
        mask = np.zeros((height, width), dtype="uint8")
        for z, slice_data in podosome.dilated_slices.items():
            if not slice_data.get('pixels'):
                continue
            for y, x in slice_data['pixels']:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: No contours found for podosome {podosome.id}. Skipping.")
            return None

        return contours[0]

    def _get_all_z_profiles(self, image_data: np.ndarray, contour: np.ndarray,
                          centroid: Tuple, num_directions: int = 360, 
                          num_samples: int = 100) -> np.ndarray:
        all_profiles = []
        for z in range(image_data.shape[0]):
            profiles = self._get_slice_profiles(contour, centroid, image_data, z, 
                                              num_directions, num_samples)            
            all_profiles.append(profiles)
        return np.mean(np.stack(all_profiles), axis=1)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _get_slice_profiles(contour: np.ndarray, centroid: Tuple,
                            image_data: np.ndarray, z: int, num_directions: int,
                            num_samples: int) -> np.ndarray:
        height, width = image_data.shape[1], image_data.shape[2]
        profiles = np.zeros((num_directions, num_samples), dtype=np.float64)
        distances = np.array([np.sqrt((p[0][0] - centroid[0])**2 + (p[0][1] - centroid[1])**2) for p in contour])
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
                profiles[angle, :] = np.interp(old_dists, np.linspace(0, max_dist, len(radial_values)), radial_values)
        return profiles

    def _filter_empty_slices(self, profiles: np.ndarray, podosome) -> np.ndarray:
        empty_keys = [k for k, v in podosome.dilated_slices.items() if not v["pixels"]]
        mask = ~np.isin(np.arange(profiles.shape[0]), empty_keys)
        return profiles[mask]

    def _interpolate_z(self, data: np.ndarray, target_z: int = 100) -> np.ndarray:
        original_z = data.shape[0]
        new_z = np.linspace(0, original_z-1, target_z)
        return np.array([np.interp(new_z, np.arange(original_z), data[:, i]) 
                        for i in range(data.shape[1])]).T
    
if __name__ == "__main__":

    pass
