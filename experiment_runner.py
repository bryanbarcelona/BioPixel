import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cell_analysis import PLACellAnalyzer, PodosomeProfileAnalyzer
from data_structures import PLAExperimentResult, PLACellResult, PodosomeProfileResult
from detectors import MacrophageDetector, PodosomeDetector, PodosomeManager
from image_tensors import ImageReader
from utils.io import ExperimentHandler
from visualization import Visualizer as viz

class PLAExperimentRunner:
    """Manages the execution of PLA experiments across image files."""

    SUPPORTED_EXTENSIONS: set[str] = {".lif", ".tif", ".czi", ".nd"}
    CONTROL_KEYWORDS: set[str] = {"control", "ctrl", "test_control", "negative", "baseline", "untreated"}

    def __init__(self, image_path: str) -> None:
        """Initializes the runner with an image path.

        Args:
            image_path: Path to a single image file or directory containing images.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        self.image_paths: List[str] = self._resolve_filepaths(image_path)
        self.iteration_tracker: dict[str, int] = {"file": 0, "scene": 0, "cell": 0}
        self.donor_count: int = 0
        self.cell_count: int = 0
        self.experiment_result: Optional[PLAExperimentResult] = None
        self.is_control: bool = False
        self.base_directory: str = ""

    def _resolve_filepaths(self, path: str) -> List[str]:
        """Resolves file paths for processing.

        Args:
            path: Path to a file or directory.

        Returns:
            List of valid image file paths.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Path '{path}' does not exist.")

        self.base_directory = str(path_obj.parent if path_obj.is_file() else path_obj)
        if path_obj.is_file():
            return [str(path_obj)]
        return [
            str(file)
            for file in path_obj.glob("*")
            if file.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

    def run(self, use_cache: bool = True, force_recompute: bool = False) -> PLAExperimentResult:
        """Runs the PLA experiment.

        Args:
            use_cache: If True, attempts to load results from cache. Defaults to True.
            force_recompute: If True, ignores cache and recomputes. Defaults to False.

        Returns:
            PLAExperimentResult containing experiment outcomes.
        """
        cache_path = self._get_cache_path()
        if use_cache and cache_path.exists() and not force_recompute:
            self.experiment_result = self._load_from_cache(cache_path)
        else:
            self.experiment_result = PLAExperimentResult()
            self.donor_count += 1
            self._process_files()
            self._save_to_cache(cache_path)
        return self.experiment_result

    def _process_files(self) -> None:
        """Processes each image file."""
        for file_idx, file_path in enumerate(self.image_paths, start=1):
            self._set_control_status(file_path)
            self.iteration_tracker.update({"file": file_idx, "scene": 0, "cell": 0})
            self._process_scenes(file_path)

    def _process_scenes(self, file_path: str) -> None:
        """Processes scenes within an image file.

        Args:
            file_path: Path to the image file.
        """
        images = ImageReader(file_path).image_data()
        for scene_idx, scene in enumerate(images, start=1):
            self.iteration_tracker.update({"scene": scene_idx, "cell": 0})
            self._process_cells(scene, file_path)

    def _process_cells(self, scene: np.ndarray, file_path: str) -> None:
        """Processes cells within a scene.

        Args:
            scene: Image data for the scene.
            file_path: Path to the image file.
        """
        cells = MacrophageDetector(scene, channel=-1, only_center_mask=False).blackout_non_mask_areas()
        for cell_idx, cell in enumerate(cells, start=1):
            self.iteration_tracker["cell"] = cell_idx
            self.cell_count += 1

            analyzer = PLACellAnalyzer(cell, control=self.is_control)
            analyzer.run()

            cell_result = PLACellResult(
                file_idx=self.iteration_tracker["file"],
                scene_idx=self.iteration_tracker["scene"],
                cell_idx=cell_idx,
                donor_id=self.donor_count,
                is_control=analyzer.control,
                signals=analyzer.signals,
                file_path=Path(file_path),
            )
            self.experiment_result.update(cell_result)

    def _set_control_status(self, file_path: str) -> None:
        """Determines if the file is a control based on its name.

        Args:
            file_path: Path to the image file.
        """
        basename = os.path.splitext(os.path.basename(file_path).lower())[0]
        self.is_control = any(keyword in basename for keyword in self.CONTROL_KEYWORDS)

    def _get_cache_path(self) -> Path:
        """Generates the cache file path.

        Returns:
            Path to the cache file.
        """
        folder_name = Path(self.base_directory).name
        return Path(self.base_directory) / f"{folder_name}.bpx"

    def _save_to_cache(self, path: Path) -> None:
        """Saves experiment results to cache.

        Args:
            path: Path to the cache file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.experiment_result, f)

    def _load_from_cache(self, path: Path) -> PLAExperimentResult:
        """Loads experiment results from cache.

        Args:
            path: Path to the cache file.

        Returns:
            Loaded PLAExperimentResult.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

class PodosomeExperimentRunner:
    """Manages execution of podosome experiments across image files and channels."""

    def __init__(self, experiment: dict) -> None:
        """Initializes the runner with experiment configuration.

        Args:
            experiment: Dictionary containing experiment settings, including pickle path and channels.
        """
        self.experiment: dict = experiment
        self.pickle_path: Path = Path(experiment["pickle"])

    def run(self) -> Dict[int, PodosomeProfileResult]:
        """Runs the podosome experiment, using cache if available.

        Returns:
            Dictionary mapping channel indices to PodosomeProfileResult objects.
        """
        if self.pickle_path.exists():
            print(f"Loading cached result from {self.pickle_path}")
            with open(self.pickle_path, "rb") as f:
                return pickle.load(f)

        results: Dict[int, PodosomeProfileResult] = {
            idx: PodosomeProfileResult(
                file_path=self.experiment["channels"][idx]["plot_path"],
                profile_name=self.experiment["channels"][idx]["name"],
            )
            for idx in self.experiment["channels"]
        }

        for file_path in self.experiment["files"]:
            print(f"Processing file: {file_path}")
            for scene in ImageReader(file_path).image_data():
                for cell in MacrophageDetector(scene, channel=0, only_center_mask=True).blackout_non_mask_areas():
                    podosome_mask = PodosomeDetector(cell, channel=0).detect()
                    podosome_cell = PodosomeManager(
                        mask_3d=podosome_mask,
                        size_um=1,
                        resolution_xy=5,
                        resolution_z=1,
                    ).cell_result

                    for channel_idx in results:
                        analyzer = PodosomeProfileAnalyzer(
                            image_data=cell,
                            podosome_cell=podosome_cell,
                            channel_index=channel_idx,
                        )
                        profiles = analyzer.analyze_podosomes()
                        results[channel_idx].update_from_cell(analyzer, profiles)

        with open(self.pickle_path, "wb") as f:
            pickle.dump(results, f)

        return results
    
def pla_analysis_run(experiment_path: str, output_path: str = None, **kwargs):
    """Run PLA analysis with flexible output directory support.
    
    Args:
        experiment_path: Input directory path (can contain ~, ., ..)
        output_path: Optional output directory (defaults to experiment_path)
    """
    experiment_path = Path(experiment_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve() if output_path else experiment_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    experiment_runner = PLAExperimentRunner(experiment_path)
    experiment_runner.run()
    result = experiment_runner.experiment_result

    viz.plot_podosome_associated_signals(
        result,
        save_path=output_path / "podosome_associated_signals.png",
        kind="kde_smooth"
    )

    # Print results
    print("\n--- Experiment Summary ---")
    print(f"Total Cells Processed: {result.total_cells}")
    print(f"Total Control Cells: {len(result.control_cells)}")
    print(f"Total Treatment Cells: {len(result.treatment_cells)}")
    print(f"Total Signals Detected: {result.total_signal_count}")
    print(f"Podosome-Associated Signals: {result.podosome_associated_count}")
    print(f"Non-Podosome Signals: {result.total_signal_count - result.podosome_associated_count}")

    print("\n--- Signal Counts by Group ---")
    print(f"Control Cells - Total Signals: {result.count_signals(is_control=True)}")
    print(f"Control Cells - Podosome-Associated Signals: {result.count_signals(is_control=True, podosome_associated=True)}")
    print(f"Control Cells - Non-Podosome Signals: {result.count_signals(is_control=True, podosome_associated=False)}")
    print(f"Treatment Cells - Total Signals: {result.count_signals(is_control=False)}")
    print(f"Treatment Cells - Podosome-Associated Signals: {result.count_signals(is_control=False, podosome_associated=True)}")
    print(f"Treatment Cells - Non-Podosome Signals: {result.count_signals(is_control=False, podosome_associated=False)}")

    return result

def podosome_profile_run(experiment_path: str, output_path: str = None, **kwargs):
    """Run podosome profiling with flexible output directory support.
    
    Args:
        experiment_path: Input directory path (can contain ~, ., ..)
        output_path: Optional output directory (defaults to experiment_path)
    """
    experiment_path = Path(experiment_path).expanduser().resolve()
    
    experiments = ExperimentHandler(experiment_path)
    colormap_list = ['afmhot', 'mako', 'viridis', 'plasma', 'inferno', 'cividis']
    
    for experiment in experiments.each_file_is_one_experiment:
        results = PodosomeExperimentRunner(experiment).run()
        
        for idx, key in enumerate(results):
            result = results[key]
            cmap = colormap_list[idx % len(colormap_list)]
            
            viz.plot_profiles(
                result.mean_radial_profile,
                save_path=result.file_path,
                protein=result.profile_name,
                cmap=cmap,
                podosome_count=result.podosome_count,
                radial_count=result.radial_profiles_count
            )

    return results

if __name__ == "__main__":

    test = pla_analysis_run(r"D:\Microscopy Testing\20250408 FULL RUN DREB VS EB3", output_path=r"D:\Microscopy Testing\20250408 FULL RUN DREB VS EB3\output")
    
    podosome_profile_run(r"D:\Microscopy Testing\20250406 FULL PODOPROFILE RUN", output_path=r"D:\Microscopy Testing\20250406 FULL PODOPROFILE RUN\output")

