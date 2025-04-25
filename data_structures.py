from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import random

@dataclass
class Signal:
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
    file_idx: Optional[int] = 0
    scene_idx: Optional[int] = 0
    cell_idx: Optional[int] = 0
    donor_id: Optional[int] = 0
    is_control: Optional[bool] = False
    signals: List[Signal] = field(default_factory=list)
    file_path: Optional[Path] = None

    @property
    def podosome_associated_signals(self) -> List[Signal]:
        return [s for s in self.signals if s.podosome_associated]

    @property
    def non_podosome_signals(self) -> List[Signal]:
        return [s for s in self.signals if not s.podosome_associated]

    @property
    def total_signal_count(self) -> int:
        return sum(1 for _ in self.signals)

    @property
    def podosome_associated_count(self) -> int:
        return sum(1 for s in self.signals if s.podosome_associated)

    @property
    def non_podosome_count(self) -> int:
        return sum(1 for s in self.signals if not s.podosome_associated)
    
@dataclass
class PLAExperimentResult:
    experiment_path: Optional[Path] = None
    cells: List[PLACellResult] = field(default_factory=list)

    # ----------- Cell Grouping Methods -----------

    def get_cells(self, is_control: Optional[bool] = None, donor_id: Optional[str] = None) -> List["PLACellResult"]:
        return [
            cell for cell in self.cells
            if (is_control is None or cell.is_control == is_control)
            and (donor_id is None or cell.donor_id == donor_id)
        ]

    def get_signals(self, *, is_control: Optional[bool] = None, donor_id: Optional[str] = None, podosome_associated: Optional[bool] = None) -> List["Signal"]:
        cells = self.get_cells(is_control=is_control, donor_id=donor_id)
        signals = [s for cell in cells for s in cell.signals]
        if podosome_associated is not None:
            signals = [s for s in signals if s.podosome_associated == podosome_associated]
        return signals

    def count_signals(self, **kwargs) -> int:
        return len(self.get_signals(**kwargs))

    def signal_count_by_donor(self, **kwargs) -> dict:
        donor_counts = {}
        for cell in self.get_cells(**kwargs):
            donor_id = cell.donor_id
            signals = [s for s in cell.signals]
            if "podosome_associated" in kwargs:
                signals = [s for s in signals if s.podosome_associated == kwargs["podosome_associated"]]
            donor_counts.setdefault(donor_id, 0)
            donor_counts[donor_id] += len(signals)
        return donor_counts

    # Keep the convenience properties
    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def total_signal_count(self) -> int:
        return self.count_signals()

    @property
    def podosome_associated_count(self) -> int:
        return self.count_signals(podosome_associated=True)

    @property
    def control_cells(self) -> List["PLACellResult"]:
        return self.get_cells(is_control=True)

    @property
    def treatment_cells(self) -> List["PLACellResult"]:
        return self.get_cells(is_control=False)

    # ----------- Aggregation / Build Methods -----------
    def update(self, cell_result: PLACellResult):
        """Add a single cell result to the experiment."""
        self.cells.append(cell_result)

    @classmethod
    def from_cell_results(cls, cell_results: List[PLACellResult], experiment_path: Optional[Path] = None) -> "PLAExperimentResult":
        """Factory method to build an experiment result from multiple cell results."""
        return cls(experiment_path=experiment_path, cells=cell_results)

@dataclass
class Podosome:
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

    def _add_slice_data(self, z: int, contour: List[tuple], pixels: List[tuple], area: float, dilated: bool = False) -> None:
        target_dict = self.dilated_slices if dilated else self.slices
        target_dict[z] = {
            'contour': contour,
            'pixels': pixels,
            'area': area
        }

        if not dilated:
            self.volume += area

    def _calculate_bounding_box(self, dilated: bool = False) -> None:
        target_dict = self.dilated_slices if dilated else self.slices
        if not target_dict:
            print(f"Warning: No {'dilated' if dilated else 'original'} slices found.")
            return

        min_y = min_x = float('inf')
        max_y = max_x = float('-inf')

        for data in target_dict.values():
            pixels = data.get('pixels', [])
            if not pixels:
                continue

            y_coords = [y for y, x in pixels]
            x_coords = [x for y, x in pixels]

            min_y = min(min_y, min(y_coords))
            max_y = max(max_y, max(y_coords))
            min_x = min(min_x, min(x_coords))
            max_x = max(max_x, max(x_coords))

        if min_y == float('inf') or min_x == float('inf') or max_y == float('-inf') or max_x == float('-inf'):
            print("Warning: No valid pixels found for bounding box.")
            return
        
        height = max_y - min_y + 1
        width = max_x - min_x + 1

        if dilated:
            self.dilated_bounding_box = (min_y, min_x, max_y, max_x, height, width)
        else:
            self.bounding_box = (min_y, min_x, max_y, max_x, height, width)

    def _calculate_centroid(self):
        all_coords = [(y, x, z) for z, data in self.slices.items() for (y, x) in data["pixels"]]
        
        if not all_coords:
            self.centroid = None
            return
        
        y_vals, x_vals, z_vals = zip(*all_coords)
        self.centroid =  (
            float(sum(x_vals)) / len(x_vals),
            float(sum(y_vals)) / len(y_vals),
            float(sum(z_vals)) / len(z_vals),
        )

    def __str__(self):
        return (
            f"Podosome #{self.id} | "
            f"Slices: {len(self.slices)} | "
            f"Dilated: {len(self.dilated_slices)} | "
            f"Volume: {self.volume:.2f} | "
            f"Centroid: {self.centroid}"
        )

    __repr__ = __str__

@dataclass
class PodosomeCellResult:
    """Podosomes from a single cell (or ROI)."""
    cell_id: int
    podosomes: List[Podosome] = field(default_factory=list)
    label_map: np.ndarray = None  # Optional 3D label map for this cell

    # ----------- Aggregated Properties -----------
    @property
    def podosome_count(self) -> int:
        return len(self.podosomes)

    @property
    def total_volume(self) -> float:
        return sum(p.volume for p in self.podosomes)

    @property
    def density(self) -> float:
        """Podosomes per unit volume (if cell volume is known)."""
        ...

    def filter_by_volume(self, min_volume: float) -> List[Podosome]:
        return [p for p in self.podosomes if p.volume >= min_volume]

@dataclass
class PodosomeExperimentResult:
    """
    Aggregates and processes data from multiple PodosomeCell objects in an experiment.
    """
    cells: List[PodosomeCellResult] = field(default_factory=list)
    experiment_id: Optional[int] = None  # Identifier for the experiment
    metadata: Optional[Dict[str, object]] = field(default_factory=dict)

    def add_cell(self, cell: PodosomeCellResult) -> None:
        """Add a new PodosomeCell to the experiment."""
        self.cells.append(cell)

    @property
    def all_podosomes(self) -> List[Podosome]:
        """Returns a flattened list of all podosomes across all cells."""
        return [podosome for cell in self.cells for podosome in cell.podosomes]

    @property
    def total_cells(self) -> int:
        """Returns the total number of cells in the experiment."""
        return len(self.cells)

    @property
    def total_podosome_count(self) -> int:
        """Returns the total number of podosomes across all cells."""
        return sum(cell.podosome_count for cell in self.cells)

    @property
    def total_volume(self) -> float:
        """Returns the total volume of all podosomes across all cells."""
        return sum(p.volume for p in self.all_podosomes)

    @property
    def average_volume_per_podosome(self) -> float:
        """Returns the average volume of podosomes across all cells."""
        if not self.all_podosomes:
            return 0.0
        return self.total_volume / len(self.all_podosomes)

    @property
    def average_density(self) -> float:
        """Returns the average podosome density per cell (if cell volume is known)."""
        densities = [cell.density for cell in self.cells if cell.density is not None]
        return sum(densities) / len(densities) if densities else 0.0

    def filter_podosomes_by_volume(self, min_volume: float) -> List[Podosome]:
        """Returns podosomes with volumes above the specified threshold."""
        return [p for p in self.all_podosomes if p.volume >= min_volume]

    def filter_podosomes_by_distance(self, distance_threshold: float) -> List[Podosome]:
        """Returns podosomes filtered by nearest-neighbor distance."""
        # Implement logic similar to PodosomeManager.filter_podosomes
        pass

    def podosome_count_by_cell(self) -> Dict[int, int]:
        """Returns a dictionary of podosome counts per cell ID."""
        return {cell.cell_id: cell.podosome_count for cell in self.cells}

    def total_volume_by_cell(self) -> Dict[int, float]:
        """Returns a dictionary of total podosome volumes per cell ID."""
        return {cell.cell_id: cell.total_volume for cell in self.cells}

    def relative_volume_distribution(self) -> Dict[int, List[float]]:
        """Returns a dictionary of relative volume distributions per cell ID."""
        return {cell.cell_id: [p.relative_volume for p in cell.podosomes] for cell in self.cells}

    @classmethod
    def from_cells(cls, cells: List[PodosomeCellResult], experiment_id: Optional[int] = None) -> "PodosomeExperimentResult":
        """Factory method to create an instance from a list of PodosomeCell objects."""
        experiment = cls(experiment_id=experiment_id)
        for cell in cells:
            experiment.add_cell(cell)
        return experiment

    def to_dict(self) -> Dict:
        """Serializes the experiment result into a dictionary for export."""
        return {
            "experiment_id": self.experiment_id,
            "total_cells": self.total_cells,
            "total_podosome_count": self.total_podosome_count,
            "total_volume": self.total_volume,
            "average_volume_per_podosome": self.average_volume_per_podosome,
            "average_density": self.average_density,
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "podosome_count": cell.podosome_count,
                    "total_volume": cell.total_volume,
                    "density": cell.density,
                }
                for cell in self.cells
            ],
        }

    def __str__(self):
        """
        Returns a concise summary of the experiment result, including key metrics.
        """
        total_cells = self.total_cells
        total_podosomes = self.total_podosome_count
        total_volume = self.total_volume
        avg_volume_per_podosome = self.average_volume_per_podosome
        avg_density = self.average_density

        return (
            f"PodosomeExperimentResult(experiment_id={self.experiment_id})\n"
            f"  Total Cells: {total_cells}\n"
            f"  Total Podosomes: {total_podosomes}\n"
            f"  Total Volume: {total_volume:.2f}\n"
            f"  Avg Volume per Podosome: {avg_volume_per_podosome:.2f}\n"
            f"  Avg Density: {avg_density:.2f}\n"
        )

    def __repr__(self):
        """
        Returns a developer-friendly representation of the experiment result.
        """
        return (
            f"PodosomeExperimentResult("
            f"experiment_id={self.experiment_id}, "
            f"total_cells={self.total_cells}, "
            f"total_podosomes={self.total_podosome_count}, "
            f"total_volume={self.total_volume:.2f})"
        )

@dataclass
class PodosomeProfileResult:
    file_path: str
    profile_name: str
    all_profiles: List[np.ndarray] = field(default_factory=list)
    podosome_count: int = 0
    radial_profiles_count: int = 0
    cell_count: int = 0
    
    @property 
    def mean_radial_profile(self) -> np.ndarray:
        profiles = np.array(self.all_profiles)
        return np.mean(profiles, axis=0)
    
    def update_from_cell(self, analyzer, profiles: np.ndarray):
        self.cell_count += 1
        self.podosome_count += analyzer.podosome_count
        self.radial_profiles_count += analyzer.radial_profiles_count
        self.all_profiles.append(profiles)