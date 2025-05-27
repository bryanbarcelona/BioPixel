from pathlib import Path
import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Generator, Union
from utils.io import get_resource_path

class ModelPaths:
    """
    A class containing constant paths to various model directories.

    Attributes:
        BASE_PATH (Path): The base path to the models directory.
        MODEL_MAP (dict): A dictionary mapping model names to their corresponding paths.

    Available model options:
        - "macrophage": Path to the macrophage detection model.
        - "podosomes": Path to the podocyte detection model.
        - New models may be added in the future. Please refer to the MODEL_MAP dictionary for the latest available models.
    """

    BASE_PATH = Path("models")

    MODEL_MAP = {
        "macrophage": BASE_PATH / "macropose2",
        "podosomes": BASE_PATH / "podotect",
    }

@dataclass
class DetectionParams:
    # Parameters with default values
    model: str = "macrophage"
    detection_channel: int = 1
    auxiliary_channel: int = 0
    diameter: int = 10
    do_3d: bool = False
    flow_threshold: float = 0.9
    cellprob_threshold: float = 0.0
    filename: Optional[str] = None
    output_folder: Optional[str] = None
    gpu: bool = False
    model_path: Optional[str] = field(init=False)
    cuda_status: bool = field(init=False)
    cuda_device: Optional[int] = field(init=False)
    cuda_device_name: Optional[str] = field(init=False)
    min_size: int = 15

    def __post_init__(self):
        # Set the model path based on the model name
        self.model_path = self._get_model_path(self.model)
        self._check_cuda()

        # If CUDA is available, set GPU to True regardless of user input
        if self.cuda_status:
            self.gpu = True

    def _get_model_path(self, model: str) -> str:
        """
        Get the path to the CellPose model based on the model name.
        """
        if os.path.exists(model):
            return model
        model_name = model.lower() if model else "macrophage"
        model_relative_path = ModelPaths.MODEL_MAP.get(model_name, ModelPaths.MODEL_MAP["macrophage"])
        return get_resource_path(model_relative_path)

    def _check_cuda(self):
        """
        Check if CUDA is available and log the status.
        """
        if torch.cuda.is_available():
            self.cuda_status = True
            self.cuda_device = torch.cuda.current_device()
            self.cuda_device_name = torch.cuda.get_device_name(self.cuda_device)
        else:
            self.cuda_status = False
            self.cuda_device = None
            self.cuda_device_name = None

    def update_params(self, **kwargs):
        """Update parameters using the provided overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                pass