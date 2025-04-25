from image_tensors import ImageReader
from detectors import MacrophageDetector, PodosomeDetector, PodosomeManager
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import os
from data_structures import PLAExperimentResult, PLACellResult, PodosomeProfileResult
from cell_analysis import PLACellAnalyzer, PodosomeProfileAnalyzer
import pickle
from utils.io import ExperimentHandler
from visualization import Visualizer as viz

class PLAExperimentRunner:
    def __init__(self, image_path: str):
        self.image_paths: List[str] = self._resolve_filepaths(image_path)
        self.iteration_tracker = {'file': 0, 'scene': 0, 'cell': 0}
        self.donor_count = 0
        self.cell_count = 0
        self.experiment_result: Optional[PLAExperimentResult] = None
        self.experimental_control = False

    def _resolve_filepaths(self, path: str) -> List[str]:
        path = Path(path).expanduser().resolve()

        if path.is_file():
            self.base_directory = str(path.parent)
            return [str(path)]
        elif path.is_dir():
            self.base_directory = str(path)
            return [
                str(file)
                for file in path.glob("*")
                if file.suffix in {".lif", ".tif", ".czi", ".nd"}
            ]
        else:
            raise FileNotFoundError(f"Path '{path}' does not exist.")

    def run(self, use_cache: bool = True, force_recompute: bool = False) -> PLAExperimentResult:
        cache_path = self._get_cache_path()

        if use_cache and cache_path.exists() and not force_recompute:
            self.experiment_result = self._load_from_cache(cache_path)
        else:
            self.experiment_result = PLAExperimentResult()
            self.donor_count += 1
            self._iterate_files()
            self._save_to_cache(cache_path)

        return self.experiment_result

    def _iterate_files(self):
        for file_idx, file in enumerate(self.image_paths, start=1):
            self._set_control_status(file)
            self.iteration_tracker['file'] = file_idx
            self.iteration_tracker['scene'] = 0
            self.iteration_tracker['cell'] = 0

            self._iterate_scenes(file)

    def _iterate_scenes(self, file: str):
        images = ImageReader(file).image_data()

        for scene_idx, scene in enumerate(images, start=1):
            self.iteration_tracker['scene'] = scene_idx
            self.iteration_tracker['cell'] = 0

            self._iterate_cells(scene, file)

    def _iterate_cells(self, scene, file):
        cells = MacrophageDetector(scene, channel=-1, only_center_mask=False).blackout_non_mask_areas()

        for cell_idx, cell in enumerate(cells, start=1):
            self.iteration_tracker['cell'] = cell_idx
            self.cell_count += 1

            analyzer = PLACellAnalyzer(cell, control=self.experimental_control)  # Set control logic appropriately
            analyzer.run()

            cell_result = PLACellResult(
                file_idx=self.iteration_tracker['file'],
                scene_idx=self.iteration_tracker['scene'],
                cell_idx=cell_idx,
                donor_id=self.donor_count,  # Add logic to assign donor ID if needed
                is_control=analyzer.control,
                signals=analyzer.signals,
                file_path=Path(file)
            )

            self.experiment_result.update(cell_result)

    def _set_control_status(self, file) -> None:
        control_keywords = {"control", "ctrl", "test_control", "negative", "baseline", "untreated"}
        
        basename, _ = os.path.splitext(os.path.basename(file).lower())
        
        self.experimental_control = any(keyword in basename for keyword in control_keywords)

    def _get_cache_path(self) -> Path:
        folder_name = Path(self.base_directory).name
        return Path(self.base_directory) / f"{folder_name}.bpx"

    def _save_to_cache(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.experiment_result, f)

    def _load_from_cache(self, path: Path) -> PLAExperimentResult:
        with open(path, "rb") as f:
            return pickle.load(f)

class PodosomeExperimentRunner:
    def __init__(self, experiment: dict):
        self.experiment = experiment
        self.pickle_path = Path(experiment["pickle"])  # Assuming pickle file path is provided

    def run(self):
        # If the pickle already exists, load it directly to avoid recalculating
        if self.pickle_path.exists():
            print(f"Loading cached result from {self.pickle_path}")
            with open(self.pickle_path, "rb") as f:
                return pickle.load(f)

        # Initialize the result objects for each channel
        results = {
            channel_index: PodosomeProfileResult(file_path=self.experiment["channels"][channel_index]["plot_path"], profile_name=self.experiment["channels"][channel_index]["name"])
            for channel_index in self.experiment["channels"].keys()
        }

        # Process each file in the experiment
        for file_path in self.experiment["files"]:
            print(f"Processing file: {file_path}")

            for scene in ImageReader(file_path).image_data():  # Assuming ImageReader works like this
                for cell in MacrophageDetector(scene, channel=0, only_center_mask=True).blackout_non_mask_areas():
                    # Detect podosome mask
                    podosome_mask = PodosomeDetector(cell, channel=0).detect()
                    podosome_cell = PodosomeManager(
                        mask_3d=podosome_mask,
                        size_um=1,
                        resolution_xy=5,
                        resolution_z=1
                    ).cell_result

                    # Now process each channel for the current cell
                    for channel_index in results.keys():
                        # Create an analyzer for the current channel
                        analyzer = PodosomeProfileAnalyzer(
                            image_data=cell,  # Pass the full scene
                            podosome_cell=podosome_cell,  # Pass the podosome cell
                            channel_index=channel_index  # Pass the channel index
                        )

                        # Now analyze the podosomes and get the profile for this channel in this cell
                        profiles = analyzer.analyze_podosomes()

                        # Update the corresponding result with the profile for this channel
                        results[channel_index].update_from_cell(analyzer, profiles)

        # After processing all files, pickle the results for each channel
        with open(self.pickle_path, "wb") as f:
            pickle.dump(results, f)

        return results
    
def test_PLA_run():

    image_path = r"D:\Microscopy Testing\20250323 FULL RUN"
    image_path = r"D:\Microscopy Testing\20250404 TINY RUN"
    experiment_runner = PLAExperimentRunner(image_path)
    experiment_runner.run()

    # Access the result
    result = experiment_runner.experiment_result

    viz.plot_podosome_associated_signals(result, save_path=os.path.join(image_path, "podosome_associated_signals.png"), kind="kde_smooth")
    # --- Experiment Summary ---
    print("\n--- Experiment Summary ---")
    print(f"Total Cells Processed: {result.total_cells}")
    print(f"Total Control Cells: {len(result.control_cells)}")
    print(f"Total Treatment Cells: {len(result.treatment_cells)}")

    print(f"Total Signals Detected: {result.total_signal_count}")
    print(f"Podosome-Associated Signals: {result.podosome_associated_count}")
    print(f"Non-Podosome Signals: {result.total_signal_count - result.podosome_associated_count}")

    # --- By Group ---
    print("\n--- Signal Counts by Group ---")
    print(f"Control Cells - Total Signals: {result.count_signals(is_control=True)}")
    print(f"Control Cells - Podosome-Associated Signals: {result.count_signals(is_control=True, podosome_associated=True)}")
    print(f"Control Cells - Non-Podosome Signals: {result.count_signals(is_control=True, podosome_associated=False)}")

    print(f"Treatment Cells - Total Signals: {result.count_signals(is_control=False)}")
    print(f"Treatment Cells - Podosome-Associated Signals: {result.count_signals(is_control=False, podosome_associated=True)}")
    print(f"Treatment Cells - Non-Podosome Signals: {result.count_signals(is_control=False, podosome_associated=False)}")

def test_Podosome_run():

    experiments = ExperimentHandler(r"D:\Microscopy Testing\20250406 FULL PODOPROFILE RUN")
    experiments = ExperimentHandler(r"D:\Microscopy Testing\20250420 TINY PODOPROFILE RUN")


    for experiment in experiments.each_file_is_one_experiment:
        colormap_list = ['afmhot', 'mako', 'viridis', 'plasma', 'inferno', 'cividis']
        
        results = PodosomeExperimentRunner(experiment).run()

        for idx, key in enumerate(results):
            result = results[key]

            cmap = colormap_list[idx % len(colormap_list)]

            mean_radial_profile = result.mean_radial_profile
            save_path = result.file_path
            protein = result.profile_name
            podosome_count = result.podosome_count
            radial_count = result.radial_profiles_count

            viz.plot_profiles(
                mean_radial_profile,
                save_path=save_path,
                protein=protein,
                cmap=cmap,
                podosome_count=podosome_count,
                radial_count=radial_count
            )

if __name__ == "__main__":

    #test = test_PLA_run()
    
    test_Podosome_run()

