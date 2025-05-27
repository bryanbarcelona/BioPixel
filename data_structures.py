import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

@dataclass
class Signal:
    """Represents a detected signal in microscopy data with spatial and analysis attributes.

    Attributes:
        center: (x,y) coordinates of signal center
        z: Z-layer index
        radius: Estimated radius of signal
        center_inside: (x,y) of inner center (if applicable)
        radius_inside: Inner radius (if applicable)
        minor: Minor axis length (for elliptical signals)
        circle_color: RGB color for visualization
        area_contour: Area based on contour
        area_circle: Area based on circular approximation
        area_ratio: area_contour / area_circle
        perimeter: Contour perimeter length
        perimeter_ratio: perimeter / (2π√(area/π))
        orientation: Angle of major axis (degrees)
        coordinates: Array of all pixel coordinates
        local_maxima: List of (x,y) local maxima coordinates
        pinched: Whether signal appears pinched
        approximation: Contour approximation accuracy
        is_valid: Whether signal passed validation checks
        podosome_associated: Whether signal is associated with a podosome
        podosome_label: ID of associated podosome
        distance_to_podosome: Distance to nearest podosome (µm)
        relative_signal_height: Normalized height in topographic map
        is_control: Whether signal is from control sample
    """
    center: Optional[Tuple[int, int]] = None
    z: Optional[int] = None
    radius: Optional[int] = None
    center_inside: Optional[Tuple[int, int]] = None
    radius_inside: Optional[int] = None
    minor: Optional[int] = None
    circle_color: Optional[Tuple[int, int, int]] = None
    area_contour: Optional[float] = None
    area_circle: Optional[float] = None
    area_ratio: Optional[float] = None
    perimeter: Optional[float] = None
    perimeter_ratio: Optional[float] = None
    orientation: Optional[float] = None
    coordinates: Optional[np.ndarray] = None
    local_maxima: Optional[List[Tuple[int, int]]] = None
    pinched: Optional[bool] = None
    approximation: Optional[float] = None
    is_valid: Optional[bool] = None
    podosome_associated: Optional[bool] = None
    podosome_label: Optional[int] = None
    distance_to_podosome: Optional[float] = None
    relative_signal_height: Optional[float] = None
    is_control: Optional[bool] = None


@dataclass
class PLACellResult:
    """Contains analysis results for a single cell in a Proximity Ligation Assay (PLA) experiment.

    Attributes:
        file_idx: Index of source file
        scene_idx: Microscope scene index
        cell_idx: Cell identifier
        donor_id: Biological donor identifier
        is_control: Whether cell is a control sample
        signals: List of detected signals
        file_path: Path to source image file
    """
    file_idx: Optional[int] = 0
    scene_idx: Optional[int] = 0
    cell_idx: Optional[int] = 0
    donor_id: Optional[int] = 0
    is_control: Optional[bool] = False
    signals: List[Signal] = field(default_factory=list)
    file_path: Optional[Path] = None

    @property
    def podosome_associated_signals(self) -> List[Signal]:
        """Returns signals associated with podosomes."""
        return [s for s in self.signals if s.podosome_associated]

    @property
    def non_podosome_signals(self) -> List[Signal]:
        """Returns signals not associated with podosomes."""
        return [s for s in self.signals if not s.podosome_associated]

    @property
    def total_signal_count(self) -> int:
        """Returns total number of signals."""
        return len(self.signals)

    @property
    def podosome_associated_count(self) -> int:
        """Returns count of podosome-associated signals."""
        return sum(1 for s in self.signals if s.podosome_associated)

    @property
    def non_podosome_count(self) -> int:
        """Returns count of non-podosome signals."""
        return sum(1 for s in self.signals if not s.podosome_associated)


@dataclass 
class PLAExperimentResult:
    """Aggregates results from multiple cells in a PLA experiment.

    Attributes:
        experiment_path: Path to experiment data
        cells: List of analyzed cell results
    """
    experiment_path: Optional[Path] = None
    cells: List[PLACellResult] = field(default_factory=list)

    @property
    def total_cells(self) -> int:
        """Returns total number of analyzed cells."""
        return len(self.cells)

    @property
    def total_signal_count(self) -> int:
        """Returns total signal count across all cells."""
        return sum(cell.total_signal_count for cell in self.cells)

    @property
    def podosome_associated_count(self) -> int:
        """Returns total podosome-associated signals."""
        return sum(cell.podosome_associated_count for cell in self.cells)

    @property
    def control_cells(self) -> List[PLACellResult]:
        """Returns all control cell results."""
        return self.get_cells(is_control=True)

    @property
    def treatment_cells(self) -> List[PLACellResult]:
        """Returns all treatment cell results."""
        return self.get_cells(is_control=False)

    def get_cells(self, is_control: Optional[bool] = None, donor_id: Optional[str] = None) -> List[PLACellResult]:
        """Filters cells by control status and/or donor ID.
        
        Args:
            is_control: Filter by control status if not None
            donor_id: Filter by donor ID if not None
            
        Returns:
            Filtered list of cell results
        """
        return [
            cell for cell in self.cells
            if (is_control is None or cell.is_control == is_control)
            and (donor_id is None or cell.donor_id == donor_id)
        ]

    def get_signals(
        self,
        *,
        is_control: Optional[bool] = None,
        donor_id: Optional[str] = None,
        podosome_associated: Optional[bool] = None
    ) -> List[Signal]:
        """Gets signals matching specified filters.
        
        Args:
            is_control: Filter by control status
            donor_id: Filter by donor ID
            podosome_associated: Filter by podosome association
            
        Returns:
            List of matching signals
        """
        cells = self.get_cells(is_control=is_control, donor_id=donor_id)
        signals = [s for cell in cells for s in cell.signals]
        if podosome_associated is not None:
            signals = [s for s in signals if s.podosome_associated == podosome_associated]
        return signals

    def count_signals(self, **kwargs) -> int:
        """Counts signals matching given filters."""
        return len(self.get_signals(**kwargs))

    def signal_count_by_donor(self, **kwargs) -> Dict[str, int]:
        """Counts signals per donor ID.
        
        Returns:
            Dictionary mapping donor IDs to signal counts
        """
        counts = {}
        for cell in self.get_cells(**kwargs):
            counts[cell.donor_id] = counts.get(cell.donor_id, 0) + len(
                [s for s in cell.signals 
                 if "podosome_associated" not in kwargs 
                 or s.podosome_associated == kwargs["podosome_associated"]]
            )
        return counts

    def update(self, cell_result: PLACellResult) -> None:
        """Adds a cell result to the experiment.
        
        Args:
            cell_result: Cell result to add
        """
        self.cells.append(cell_result)

    @classmethod
    def from_cell_results(
        cls, 
        cell_results: List[PLACellResult], 
        experiment_path: Optional[Path] = None
    ) -> "PLAExperimentResult":
        """Creates experiment result from list of cell results.
        
        Args:
            cell_results: List of cell results to include
            experiment_path: Path to experiment data
            
        Returns:
            New PLAExperimentResult instance
        """
        return cls(experiment_path=experiment_path, cells=cell_results)


@dataclass
class Podosome:
    """Represents a single podosome with spatial and morphological data.

    Attributes:
        id: Unique identifier
        slices: Original slice data {z: {'contour': [], 'pixels': [], 'area': float}}
        dilated_slices: Dilated slice data (same structure)
        bounding_box: (min_y, min_x, max_y, max_x, height, width)
        dilated_bounding_box: Bounding box for dilated podosome
        volume: Total volume in µm³
        relative_volume: Volume normalized to cell size
        centroid: (x,y,z) coordinates of center
        color: RGB color for visualization
    """
    id: int
    slices: Dict[int, Dict[str, object]] = field(default_factory=dict)
    dilated_slices: Dict[int, Dict[str, object]] = field(default_factory=dict)
    bounding_box: Optional[Tuple[int, int, int, int, int, int]] = None
    dilated_bounding_box: Optional[Tuple[int, int, int, int, int, int]] = None
    volume: float = 0.0
    relative_volume: Optional[float] = None
    centroid: Optional[Tuple[float, float, float]] = None
    color: Tuple[int, int, int] = field(
        default_factory=lambda: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    )

    def _add_slice_data(
        self,
        z: int,
        contour: List[Tuple[int, int]],
        pixels: List[Tuple[int, int]], 
        area: float,
        dilated: bool = False
    ) -> None:
        """Adds slice data to podosome.
        
        Args:
            z: Z-layer index
            contour: List of (y,x) contour points
            pixels: List of (y,x) pixel coordinates  
            area: Area of slice
            dilated: Whether data is from dilated podosome
        """
        target = self.dilated_slices if dilated else self.slices
        target[z] = {'contour': contour, 'pixels': pixels, 'area': area}
        
        if not dilated:
            self.volume += area

    def _calculate_bounding_box(self, dilated: bool = False) -> None:
        """Calculates bounding box from slice data.
        
        Args:
            dilated: Whether to use dilated slices
        """
        slices = self.dilated_slices if dilated else self.slices
        if not slices:
            return

        coords = [
            (y, x) 
            for z_data in slices.values() 
            for y, x in z_data.get('pixels', [])
        ]
        
        if not coords:
            return
            
        y_coords, x_coords = zip(*coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_x, max_x = min(x_coords), max(x_coords)
        
        box = (
            min_y, min_x, 
            max_y, max_x,
            max_y - min_y + 1,
            max_x - min_x + 1
        )
        
        if dilated:
            self.dilated_bounding_box = box
        else:
            self.bounding_box = box

    def _calculate_centroid(self) -> None:
        """Calculates centroid from all pixel coordinates."""
        coords = [
            (y, x, z)
            for z, z_data in self.slices.items()
            for y, x in z_data.get('pixels', [])
        ]
        
        if not coords:
            self.centroid = None
            return
            
        y, x, z = zip(*coords)
        self.centroid = (
            sum(x) / len(x),
            sum(y) / len(y), 
            sum(z) / len(z)
        )

    def __str__(self) -> str:
        return (
            f"Podosome(id={self.id}, slices={len(self.slices)}, "
            f"volume={self.volume:.2f}, centroid={self.centroid})"
        )


@dataclass
class PodosomeCellResult:
    """Contains podosome analysis results for a single cell.

    Attributes:
        cell_id: Unique cell identifier
        podosomes: List of detected podosomes
        label_map: 3D label map array
    """
    cell_id: int
    podosomes: List[Podosome] = field(default_factory=list)
    label_map: Optional[np.ndarray] = None

    @property
    def podosome_count(self) -> int:
        """Returns number of podosomes in cell."""
        return len(self.podosomes)

    @property
    def total_volume(self) -> float:
        """Returns total podosome volume in cell."""
        return sum(p.volume for p in self.podosomes)

    def filter_by_volume(self, min_volume: float) -> List[Podosome]:
        """Filters podosomes by minimum volume.
        
        Args:
            min_volume: Minimum volume threshold
            
        Returns:
            List of podosomes meeting threshold
        """
        return [p for p in self.podosomes if p.volume >= min_volume]


@dataclass
class PodosomeExperimentResult:
    """Aggregates podosome results from multiple cells in an experiment.

    Attributes:
        cells: List of cell results
        experiment_id: Experiment identifier
        metadata: Additional experiment metadata
    """
    cells: List[PodosomeCellResult] = field(default_factory=list)
    experiment_id: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def all_podosomes(self) -> List[Podosome]:
        """Returns all podosomes across all cells."""
        return [p for cell in self.cells for p in cell.podosomes]

    @property
    def total_cells(self) -> int:
        """Returns total number of cells."""
        return len(self.cells)

    @property
    def total_podosome_count(self) -> int:
        """Returns total podosome count."""
        return sum(cell.podosome_count for cell in self.cells)

    @property
    def total_volume(self) -> float:
        """Returns total podosome volume."""
        return sum(p.volume for p in self.all_podosomes)

    @property
    def average_volume_per_podosome(self) -> float:
        """Returns average podosome volume."""
        return self.total_volume / self.total_podosome_count if self.total_podosome_count else 0.0

    def filter_podosomes_by_volume(self, min_volume: float) -> List[Podosome]:
        """Filters podosomes by minimum volume."""
        return [p for p in self.all_podosomes if p.volume >= min_volume]

    def podosome_count_by_cell(self) -> Dict[int, int]:
        """Returns podosome counts per cell ID."""
        return {cell.cell_id: cell.podosome_count for cell in self.cells}

    def total_volume_by_cell(self) -> Dict[int, float]:
        """Returns total podosome volume per cell ID."""
        return {cell.cell_id: cell.total_volume for cell in self.cells}

    @classmethod
    def from_cells(
        cls,
        cells: List[PodosomeCellResult],
        experiment_id: Optional[int] = None
    ) -> "PodosomeExperimentResult":
        """Creates experiment result from cell results.
        
        Args:
            cells: List of cell results
            experiment_id: Experiment identifier
            
        Returns:
            New experiment result instance
        """
        return cls(cells=cells, experiment_id=experiment_id)

    def __str__(self) -> str:
        """Returns formatted summary of experiment results."""
        return (
            f"PodosomeExperimentResult("
            f"cells={self.total_cells}, "
            f"podosomes={self.total_podosome_count}, "
            f"total_volume={self.total_volume:.2f})"
        )


@dataclass
class PodosomeProfileResult:
    """Stores radial intensity profile results from podosome analysis.

    Attributes:
        file_path: Source image file path
        profile_name: Identifier for profile set
        all_profiles: List of radial profile arrays
        podosome_count: Total podosomes analyzed
        radial_profiles_count: Total radial profiles generated
        cell_count: Number of cells analyzed
    """
    file_path: str
    profile_name: str
    all_profiles: List[np.ndarray] = field(default_factory=list)
    podosome_count: int = 0
    radial_profiles_count: int = 0
    cell_count: int = 0

    @property
    def mean_radial_profile(self) -> np.ndarray:
        """Returns average of all radial profiles."""
        return np.mean(self.all_profiles, axis=0) if self.all_profiles else np.array([])

    def update_from_cell(self, analyzer, profiles: np.ndarray) -> None:
        """Updates results with data from a single cell analysis.
        
        Args:
            analyzer: Cell analyzer instance
            profiles: Radial profiles from cell
        """
        self.cell_count += 1
        self.podosome_count += analyzer.podosome_count
        self.radial_profiles_count += analyzer.radial_profiles_count
        self.all_profiles.append(profiles)