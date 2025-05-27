import glob
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from czifile import CziFile
from oiffile import OifFile
from readlif.reader import LifFile
import tifffile
from tifffile import TiffFile

from utils import czi_metadata, image_tools, io


@dataclass
class Metadata:
    """Stores metadata for an image."""
    image_name: Optional[str] = None
    Info: Optional[str] = None
    x_resolution: float = 0.0
    y_resolution: float = 0.0
    slices: int = 0
    x_size: int = 0
    y_size: int = 0
    channels: int = 0
    frames: int = 0
    time_dim: float = 0.0
    end: float = 0.0
    begin: float = 0.0
    Ranges: Optional[Tuple] = None
    min: float = 0.0
    max: float = 0.0
    _spacing: float = 0.0

    @property
    def z_range(self) -> float:
        """Calculates the z-range from begin to end."""
        return float(abs(self.begin - self.end))

    @property
    def spacing(self) -> float:
        """Calculates or returns the z-spacing."""
        return self._spacing if self._spacing > 0 else float(self.z_range / (self.slices - 1) if self.slices > 1 else 0.0)

    @spacing.setter
    def spacing(self, value: float) -> None:
        """Sets the z-spacing value."""
        self._spacing = value

class MetadataManager:
    """Manages metadata for an image, providing access and update functionality."""

    def __init__(self) -> None:
        """Initializes with an empty Metadata object."""
        self._metadata = Metadata()

    @property
    def metadata(self) -> Metadata:
        """Returns the current Metadata object."""
        return self._metadata

    def update(self, **kwargs: Any) -> None:
        """Updates metadata fields with provided key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)

    def __getitem__(self, key: str) -> Any:
        """Retrieves a metadata value by key."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets a metadata value by key."""
        self._metadata.update({key: value})

    def get(self, key: str) -> Any:
        """Retrieves a metadata value by key, raising an error if not found."""
        if hasattr(self._metadata, key):
            return getattr(self._metadata, key)
        raise AttributeError(f"'Metadata' object has no attribute '{key}'")

    def reset_metadata(self) -> None:
        """Resets metadata to a new empty Metadata instance."""
        self._metadata = Metadata()

    def __repr__(self) -> str:
        """Returns a string representation of the metadata."""
        return repr(self._metadata)

    @property
    def imagej_compatible_metadata(self) -> Dict[str, Any]:
        """Generates ImageJ-compatible metadata dictionary."""
        return {
            "axes": "TZCYX",
            "spacing": self._metadata.spacing,
            "unit": "micron",
            "hyperstack": True,
            "mode": "color",
            "channels": self._metadata.channels,
            "frames": self._metadata.frames,
            "Info": self._metadata.Info,
            "slices": self._metadata.slices,
            "Ranges": self._metadata.Ranges,
            "min": self._metadata.min,
            "max": self._metadata.max,
            "metadata": "ImageJ=1.53c\n",
            "x_resolution": self._metadata.x_resolution,
            "y_resolution": self._metadata.y_resolution,
        }
    
class BaseImageReader(ABC):
    """Abstract base class for reading and processing image files."""

    def __init__(self, image_path: str, override_pixel_size_um: Optional[float] = None) -> None:
        """Initializes the image reader.

        Args:
            image_path: Path to the image file.
            override_pixel_size_um: Optional pixel size in micrometers to override default. Defaults to None.
        """
        self._image_path: str = image_path
        self._image_dir: str = os.path.dirname(image_path)
        self._image_name: str = os.path.splitext(os.path.basename(image_path))[0]
        self._metadata = MetadataManager()
        self._configurations: Dict = {}
        self._override_pixel_size_um: Optional[float] = override_pixel_size_um
        self._image_data: List[np.ndarray] = []
        self._associated_files: List[str] = []
        self._initialize_folders()

    def _initialize_folders(self) -> None:
        """Sets up default directories for original and output files."""
        base_dir = os.path.dirname(self._image_path)
        self._originals_repo: str = os.path.join(base_dir, "Original Data")
        self._output_folder: str = os.path.join(base_dir, "TIF")

    def _gather_associated_files(self, search_pattern: str) -> None:
        """Collects associated files matching the search pattern.

        Args:
            search_pattern: Glob pattern to match associated files.

        Raises:
            ValueError: If the main image file or no associated files are found.
        """
        extension = os.path.splitext(os.path.basename(self._image_path))[1].replace(".", "").upper()
        if not os.path.exists(self._image_path):
            raise ValueError(f"{extension} file not found: {self._image_path}")

        self._associated_files = glob.glob(search_pattern)
        if not self._associated_files:
            raise ValueError(f"No associated files found for {extension} file: {self._image_name}")

    @abstractmethod
    def _process_image_and_metadata(self) -> List[Tuple[str, np.ndarray]]:
        """Processes the image and its metadata.

        Returns:
            List of tuples containing file paths and corresponding image data.
        """
        pass

    def _initialize_image_data(self) -> None:
        """Initializes the image data list from processed images."""
        self._image_data = [image[1] for image in self._process_image_and_metadata()]

    @property
    def number_of_images(self) -> int:
        """Returns the number of images."""
        return len(self._image_data)

    @property
    def metadata(self) -> MetadataManager:
        """Returns the metadata manager."""
        return self._metadata

    @property
    def image_path(self) -> str:
        """Returns the image file path."""
        return self._image_path

    @property
    def image_directory(self) -> str:
        """Returns the image directory."""
        return self._image_dir

    def image_data(self) -> Iterator[np.ndarray]:
        """Yields image data from processed images."""
        for _, image_data in self._process_image_and_metadata():
            yield image_data

    @property
    def filename(self) -> str:
        """Returns the image filename without extension."""
        return self._image_name

    @property
    def associated_files(self) -> List[str]:
        """Returns a copy of the associated files list."""
        return self._associated_files.copy()

    @property
    def originals_repo(self) -> str:
        """Returns the path to the originals repository."""
        return self._originals_repo

    @originals_repo.setter
    def originals_repo(self, value: str) -> None:
        """Sets the originals repository path."""
        self._originals_repo = value if os.path.isabs(value) and ":" in value else os.path.join(
            os.path.dirname(self._image_path), value
        )

    @property
    def output_folder(self) -> str:
        """Returns the output folder path."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value: str) -> None:
        """Sets the output folder path."""
        self._output_folder = value if os.path.isabs(value) and ":" in value else os.path.join(
            os.path.dirname(self._image_path), value
        )

    @property
    def configurations(self) -> Dict:
        """Returns the configurations dictionary."""
        return self._configurations

    @configurations.setter
    def configurations(self, new_configurations: Dict) -> None:
        """Updates the configurations dictionary."""
        self._configurations.update(new_configurations)

    def info_string(self, image_data: np.ndarray) -> str:
        """Generates a metadata info string for the image.

        Args:
            image_data: Image data array.

        Returns:
            Formatted metadata string.
        """
        time_count, z_size, channel_count, y_size, x_size = image_data.shape
        bits_per_pixel = image_data.dtype.itemsize * 8
        pixel_type = str(image_data.dtype)
        byte_order = image_data.dtype.byteorder

        if byte_order in ("=", "|"):
            byte_order_metadata = "true" if sys.byteorder == "little" else "false"
        elif byte_order == "<":
            byte_order_metadata = "true"
        elif byte_order == ">":
            byte_order_metadata = "false"
        else:
            byte_order_metadata = "Unknown"

        leading_string = (
            f"BitsPerPixel = {bits_per_pixel}\r\n"
            f"DimensionOrder = TZCYX\r\n"
            f"IsInterleaved = false\r\n"
            f"LittleEndian = {byte_order_metadata}\r\n"
            f"PixelType = {pixel_type}\r\n"
            f"SizeC = {channel_count}\r\n"
            f"SizeT = {time_count}\r\n"
            f"SizeX = {x_size}\r\n"
            f"SizeY = {y_size}\r\n"
            f"SizeZ = {z_size}\r\n"
        )

        settings = dict(self._configurations)
        stack = [(None, settings)]
        parent_keys = []
        lines = []

        while stack:
            key, value = stack.pop()
            if key is not None:
                parent_keys.append(key)

            if isinstance(value, dict):
                stack.append((None, None))
                stack.extend(sorted(value.items(), reverse=True))
            else:
                if value is not None:
                    lines.append(f"{''.join(f'[{k}]' for k in parent_keys)} = {value}\r\n")
                if parent_keys:
                    parent_keys.pop()

        return leading_string + "".join(lines)

    def save_to_tif(self) -> List[str]:
        """Saves images as TIFF files with metadata.

        Returns:
            List of paths to saved TIFF files.
        """
        tif_paths = []
        for tif_filepath, image_data in self._process_image_and_metadata():
            self.metadata["Info"] = self.info_string(image_data)
            image_tools.save_tif(image_data, tif_filepath, self.metadata.imagej_compatible_metadata)
            tif_paths.append(tif_filepath)
        return tif_paths

    def store_originals(self) -> None:
        """Moves associated files to the originals repository."""
        file_paths = (
            [self._associated_files] if isinstance(self._associated_files, str) else self._associated_files
        )
        os.makedirs(self._originals_repo, exist_ok=True)

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            output_path = os.path.join(self._originals_repo, filename)
            print(f"Moved {file_path} to {output_path}")
            shutil.move(file_path, output_path)

class LifImageReader(BaseImageReader):
    """Reader for LIF (Leica Image File) format images."""

    def __init__(self, lif_file_path: str, override_pixel_size_um: Optional[float] = None) -> None:
        """Initializes the LIF image reader.

        Args:
            lif_file_path: Path to the LIF file.
            override_pixel_size_um: Optional pixel size in micrometers to override default. Defaults to None.
        """
        super().__init__(lif_file_path, override_pixel_size_um)
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}.*")
        self._lif_file = LifFile(lif_file_path)

    @property
    def number_of_series(self) -> int:
        """Returns the number of image series in the LIF file."""
        return self._lif_file.num_images

    def _process_image_and_metadata(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Processes LIF image data and metadata.

        Yields:
            Tuple containing the TIFF file path and corresponding image data array (TZCYX).
        """
        tif_directory = f"{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name}"
        for series_index in range(self.number_of_series):
            lif = self._lif_file.get_image(img_n=series_index)
            info = lif.info
            self.configurations = dict(info["settings"])

            self._metadata.update(
                image_name=f"{tif_directory} Series {series_index + 1}.tif",
                x_resolution=info["scale_n"].get(1, 0.0),
                y_resolution=info["scale_n"].get(2, 0.0),
                slices=info["dims_n"].get(3, 0),
                x_size=info["dims_n"].get(1, 0),
                y_size=info["dims_n"].get(2, 0),
                frames=info["dims_n"].get(4, 0),
                time_dim=info["scale_n"].get(4, 0.0),
                channels=info["channels"],
                end=float(info["settings"]["End"]) * 1e6,
                begin=float(info["settings"]["Begin"]) * 1e6,
            )

            image_data = []
            for c in range(self._metadata.channels):
                channel_data = [
                    np.array(lif.get_frame(z=z, c=c), dtype=np.uint8)
                    for z in range(self._metadata.slices)
                ]
                channel_data = np.stack(channel_data, axis=0)
                image_data.append(channel_data)

            image_data = np.stack(image_data, axis=1)
            if image_data.ndim == 4:
                image_data = np.expand_dims(image_data, axis=0)

            ranges = image_tools.get_ranges(image_data)
            self._metadata.update(**ranges)
            self._metadata["Info"] = self.info_string(image_data)

            yield self._metadata.image_name, image_data

class OibImageReader(BaseImageReader):
    """Reader for OIB (Olympus Image Binary) format images."""

    def __init__(self, oib_file_path: str, override_pixel_size_um: Optional[float] = None) -> None:
        """Initializes the OIB image reader.

        Args:
            oib_file_path: Path to the OIB file.
            override_pixel_size_um: Optional pixel size in micrometers to override default. Defaults to None.
        """
        super().__init__(oib_file_path, override_pixel_size_um)
        self._oib_file = OifFile(oib_file_path)
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}.*")
        self._initialize_image_data()

    def close_oib_file(self) -> None:
        """Closes the OIB file if open."""
        if self._oib_file:
            self._oib_file.close()
            self._oib_file = None

    def _process_image_and_metadata(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Processes OIB image data and metadata.

        Yields:
            Tuple containing the TIFF file path and corresponding image data array (TZCYX).
        """
        axis_info = {
            self._oib_file.mainfile["Axis 0 Parameters Common"]["AxisCode"]: self._oib_file.mainfile["Axis 0 Parameters Common"]["MaxSize"],
            self._oib_file.mainfile["Axis 1 Parameters Common"]["AxisCode"]: self._oib_file.mainfile["Axis 1 Parameters Common"]["MaxSize"],
            self._oib_file.mainfile["Axis 2 Parameters Common"]["AxisCode"]: self._oib_file.mainfile["Axis 2 Parameters Common"]["MaxSize"],
            self._oib_file.mainfile["Axis 3 Parameters Common"]["AxisCode"]: self._oib_file.mainfile["Axis 3 Parameters Common"]["MaxSize"],
            self._oib_file.mainfile["Axis 4 Parameters Common"]["AxisCode"]: self._oib_file.mainfile["Axis 4 Parameters Common"]["MaxSize"],
            "X Conversion": round(self._oib_file.mainfile["Reference Image Parameter"]["WidthConvertValue"], 4),
            "Y Conversion": round(self._oib_file.mainfile["Reference Image Parameter"]["HeightConvertValue"], 4),
            "End": self._oib_file.mainfile["Axis 3 Parameters Common"]["EndPosition"],
            "Start": self._oib_file.mainfile["Axis 3 Parameters Common"]["StartPosition"],
        }

        self.configurations = dict(self._oib_file.mainfile)
        image_data = self._oib_file.asarray()

        for dim in range(5):
            if image_data.shape[dim] != axis_info["CZTYX"[dim]] and axis_info["CZTYX"[dim]] == 0:
                axis_info["CZTYX"[dim]] = 1
                image_data = np.expand_dims(image_data, axis=dim)

        image_data = image_data.transpose(2, 1, 0, 3, 4)[:, :, ::-1, :, :]

        tif_directory = f"{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name}.tif"
        metadata_attrs = {
            "image_name": tif_directory,
            "Info": None,
            "x_resolution": 1.0 / axis_info["X Conversion"] if axis_info["X Conversion"] != 0 else 0.0,
            "y_resolution": 1.0 / axis_info["Y Conversion"] if axis_info["Y Conversion"] != 0 else 0.0,
            "slices": axis_info["Z"],
            "x_size": axis_info["X"],
            "y_size": axis_info["Y"],
            "frames": axis_info["T"],
            "time_dim": 1.0,
            "channels": axis_info["C"],
            "end": axis_info["End"],
            "begin": axis_info["Start"],
        }

        self._metadata.update(**metadata_attrs)
        ranges = image_tools.get_ranges(image_data)
        self._metadata.update(**ranges)
        self._metadata["Info"] = self.info_string(image_data)

        yield self._metadata.image_name, image_data

    def store_originals(self) -> None:
        """Closes the OIB file and moves associated files to the originals repository."""
        self.close_oib_file()
        super().store_originals()
        
class NdImageReader(BaseImageReader):
    """Reader for ND (Nikon Dimensions) format images."""

    def __init__(self, nd_image_path: str, override_pixel_size_um: Optional[float] = None) -> None:
        """Initializes the ND image reader.

        Args:
            nd_image_path: Path to the ND file.
            override_pixel_size_um: Optional pixel size in micrometers to override default. Defaults to None.
        """
        super().__init__(nd_image_path, override_pixel_size_um)
        self._nd_image = None
        self._stage_count: int = 1
        self._image_tree: List = []
        self._parse_nd_file()
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}*")
        self._extract_metadata_from_associated_file()
        self._determine_image_file_tree()

    def _parse_nd_file(self) -> None:
        """Parses the ND file and extracts metadata."""
        with open(self._image_path, "r") as f:
            lines = f.readlines()

        attributes = {}
        for line in lines:
            line = line.replace('"', '').strip()
            if line:
                key, *values = line.split(',')
                key = key.strip()
                value = values[0].strip() if len(values) == 1 else '.'.join(values).strip()
                try:
                    attributes[key] = int(value) if value.isdigit() else float(value) if '.' in value and value.replace('.', '').isdigit() else value
                except ValueError:
                    attributes[key] = value

        self.configurations = attributes

        dimensions = {
            "frames": ("DoTimelapse", "NTimePoints"),
            "channels": ("DoWave", "NWavelengths"),
            "slices": ("DoZSeries", "NZSteps"),
            "stage": ("DoStage", "NStagePositions"),
            "end": ("DoZSeries", "ZStepSize"),
        }

        metadata_attributes = {}
        for dim_name, (flag_key, size_key) in dimensions.items():
            size = 1
            if flag_key in attributes and size_key in attributes:
                if attributes[flag_key].upper() == "TRUE":
                    size = attributes.get(size_key, 1)
            metadata_attributes[dim_name] = size

        if "end" in attributes and "z_size" in attributes:
            attributes["end"] = attributes["end"] * (attributes["z_size"] - 1)

        self._stage_count = metadata_attributes["stage"]
        self._metadata.update(**metadata_attributes)

    def _extract_metadata_from_associated_file(self) -> None:
        """Extracts metadata from the first associated TIF or STK file."""
        image_files = [f for f in self._associated_files if f.endswith((".tif", ".stk"))]
        if not image_files:
            return

        with TiffFile(image_files[0]) as tif:
            ifd_tags = tif.pages[0].tags
            relevant_tags = (
                256, 257, 258, 277, 259, 262, 282, 283, 296, 305, 33628,
            )

            extracted_metadata = {}
            for tag in relevant_tags:
                if tag in ifd_tags:
                    value = ifd_tags[tag].value
                    if tag == 256:
                        extracted_metadata["ImageWidth"] = int(value)
                    elif tag == 257:
                        extracted_metadata["ImageLength"] = int(value)
                    elif tag == 258:
                        extracted_metadata["BitsPerSample"] = str(value)
                    elif tag == 277:
                        extracted_metadata["SamplesPerPixel"] = str(value)
                    elif tag == 259:
                        extracted_metadata["Compression"] = "Uncompressed" if value == 1 else None
                    elif tag == 262:
                        extracted_metadata["PhotometricInterpretation"] = "BlackIsZero" if value == 1 else None
                    elif tag == 282:
                        extracted_metadata["XResolution"] = float(value[0])
                    elif tag == 283:
                        extracted_metadata["YResolution"] = float(value[0])
                    elif tag == 296:
                        extracted_metadata["ResolutionUnit"] = "Centimeter" if value == 3 else None
                    elif tag == 305:
                        extracted_metadata["Software"] = str(value)
                    elif tag == 33628:
                        extracted_metadata["DateTime"] = value["CreateTime"].strftime("%Y:%m:%d %H:%M:%S")
                        x_cal = value["XCalibration"]
                        y_cal = value["YCalibration"]
                        if x_cal == 0 and self._override_pixel_size_um is None:
                            extracted_metadata["XCalibration"] = 1.0 / (extracted_metadata["XResolution"] / 10000)
                        elif self._override_pixel_size_um is not None:
                            extracted_metadata["XCalibration"] = 1.0 / float(self._override_pixel_size_um)
                        else:
                            extracted_metadata["XCalibration"] = 1.0 / float(x_cal)
                        if y_cal == 0 and self._override_pixel_size_um is None:
                            extracted_metadata["YCalibration"] = 1.0 / (extracted_metadata["YResolution"] / 10000)
                        elif self._override_pixel_size_um is not None:
                            extracted_metadata["YCalibration"] = 1.0 / float(self._override_pixel_size_um)
                        else:
                            extracted_metadata["YCalibration"] = 1.0 / float(y_cal)

            self.configurations = extracted_metadata
            self._metadata.update(
                x_size=extracted_metadata.get("ImageWidth", 0),
                y_size=extracted_metadata.get("ImageLength", 0),
                x_resolution=extracted_metadata.get("XCalibration", 0.0),
                y_resolution=extracted_metadata.get("YCalibration", 0.0),
            )

    def _determine_image_file_tree(self) -> None:
        """Builds the image file tree based on stage, time, and channel indices."""
        channel_count = self._metadata.metadata.channels
        time_count = self._metadata.metadata.frames
        stage_count = self._stage_count

        image_files = self._associated_files.copy()
        nd_file_path, _ = os.path.splitext(self._image_path)
        if self._image_path in image_files:
            image_files.remove(self._image_path)

        for i, file_path in enumerate(image_files):
            image_files[i] = file_path.replace(nd_file_path, "")

        w_numbers, s_numbers, t_numbers = set(), set(), set()
        for file_name in image_files:
            if w_match := re.search(r"_w(\d+)", file_name):
                w_numbers.add(w_match.group(1))
            if s_match := re.search(r"_s(\d+)", file_name):
                s_numbers.add(s_match.group(1))
            if t_match := re.search(r"_t(\d+)", file_name):
                t_numbers.add(t_match.group(1))

        num_unique_w, num_unique_s, num_unique_t = max(len(w_numbers), 1), max(len(s_numbers), 1), max(len(t_numbers), 1)
        if num_unique_w != channel_count or num_unique_s != stage_count or num_unique_t != time_count:
            raise ValueError("Not all files for assembly of TIFF file are available.")

        ordered_image_files = []
        for stage in range(stage_count):
            stage_list = []
            search_pattern = f"_s{stage + 1}"
            stage_files = [f for f in image_files if search_pattern in f] or image_files.copy()
            ordered_image_files.append(stage_list)
            for time in range(time_count):
                time_list = []
                search_pattern = f"_t{(time + 1)}[_.]"
                time_files = [f for f in stage_files if re.search(search_pattern, f)] or stage_files.copy()
                ordered_image_files[stage].append(time_list)
                for channel in range(channel_count):
                    channel_list = []
                    search_pattern = f"_w{channel + 1}"
                    channel_files = [f for f in time_files if search_pattern in f] or time_files.copy()
                    ordered_image_files[stage][time].append(channel_list)
                    ordered_image_files[stage][time][channel].extend(channel_files)

        self._image_tree = self._prefix_nested_list(ordered_image_files, nd_file_path)

    def _prefix_nested_list(self, nested_list: List, prefix: str) -> List:
        """Prefixes file paths in a nested list with the ND file path."""
        for i, item in enumerate(nested_list):
            if isinstance(item, list):
                self._prefix_nested_list(item, prefix)
            elif isinstance(item, str):
                nested_list[i] = prefix + item
        return nested_list

    def _process_image_and_metadata(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Processes ND image data and metadata.

        Yields:
            Tuple containing the TIFF file path and corresponding image data array (TZCYX).
        """
        images = self._image_tree
        filename = self._image_name
        output_folder = f"{self.image_directory}{os.path.sep}TIF"
        for stage in range(len(images)):
            tif_filepath = f"{output_folder}{os.path.sep}{filename} Series {stage + 1}.tif"
            time_image_series = []
            for time in range(len(images[stage])):
                channel_series = []
                for channel in range(len(images[stage][time])):
                    stk_file = tifffile.imread(images[stage][time][channel])
                    if len(stk_file.shape) < 3:
                        stk_file = np.expand_dims(stk_file, axis=0)
                    channel_series.append(stk_file)
                channel_series = np.stack(channel_series, axis=0)
                time_image_series.append(channel_series)
            time_image_series = np.stack(time_image_series, axis=0)
            time_image_series = time_image_series.transpose(0, 2, 1, 3, 4)[:, :, ::-1, :, :]

            ranges = image_tools.get_ranges(time_image_series)
            self._metadata.update(**ranges)
            self._metadata["Info"] = self.info_string(time_image_series)

            yield tif_filepath, time_image_series

class CziImageReader(BaseImageReader):
    """Reader for CZI (Zeiss) format images."""

    def __init__(self, czi_file_path: str, override_pixel_size_um: Optional[float] = None) -> None:
        """Initialize the CZI image reader.

        Args:
            czi_file_path: Path to the CZI file.
            override_pixel_size_um: Optional pixel size in micrometers to override default.
        """
        super().__init__(czi_file_path, override_pixel_size_um)
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}.*")
        self._czi_file = CziFile(czi_file_path)
        self._dimension_map: Dict[str, int] = self._map_dimensions()

    def _map_dimensions(self) -> Dict[str, int]:
        """Map CZI file axes to their sizes.

        Returns:
            Dictionary mapping axis identifiers to their sizes.

        Raises:
            ValueError: If number of axes doesn't match number of dimensions.
        """
        axes_sizes = self._czi_file.shape
        axes_ids = self._czi_file.axes

        if len(axes_sizes) != len(axes_ids):
            raise ValueError("Axes count doesn't match dimensions in CZI file.")

        return dict(zip(axes_ids, axes_sizes))

    def _process_image_and_metadata(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Process CZI image data and metadata.

        Yields:
            Tuple of TIFF file path and image data array (TZCYX).
        """
        tensor = self._czi_file.asarray()

        if "0" in self._dimension_map:
            position = list(self._dimension_map.keys()).index("0")
            tensor = np.squeeze(tensor, axis=position)

        phase_tensors = []
        num_phases = self._dimension_map.get("H", 1)
        if "H" in self._dimension_map:
            position = list(self._dimension_map.keys()).index("H")
            tensor = np.flip(tensor, axis=position)
            num_channels = self._dimension_map["C"]
            channels_per_phase = num_channels // num_phases

            for phase_index in range(num_phases):
                phase_tensor_list = [
                    tensor[phase_index, :, phase_index + offset * num_phases, :, :, :]
                    for offset in range(channels_per_phase)
                ]
                phase_tensor_stack = np.stack(phase_tensor_list, axis=1)
                phase_tensor_stack = np.transpose(phase_tensor_stack, (0, 2, 1, 3, 4))
                phase_tensors.append(phase_tensor_stack)
        else:
            phase_tensors = [tensor]

        for phase_index, phase_tensor in enumerate(phase_tensors):
            self._configurations = czi_metadata.get_metadata_as_dict(
                self._image_path, iteration=phase_index, stepsize=num_phases
            )
            config = self._configurations

            mode = {0: "Confocal", 1: "AiryScan"}
            image_name = (
                f"{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name} {mode[phase_index]}.tif"
                if num_phases == 2
                else f"{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name} Phase {phase_index + 1}.tif"
            )

            self._metadata.update(
                image_name=image_name,
                x_resolution=1.0 / config["Scaling"].get("X", 0.0),
                y_resolution=1.0 / config["Scaling"].get("Y", 0.0),
                slices=config["Dimensions"].get("SizeZ", 0),
                x_size=config["Dimensions"].get("SizeX", 0),
                y_size=config["Dimensions"].get("SizeY", 0),
                frames=config["Dimensions"].get("SizeT", 0),
                time_dim=config["Scaling"].get("T", 1.0),
                channels=config["Dimensions"].get("SizeC", 0),
                end=float(config["Scaling"].get("Z", 0.0) * (config["Dimensions"].get("SizeZ", 0) - 1)),
                begin=0.0,
            )

            ranges = image_tools.get_ranges(phase_tensor)
            self._metadata.update(**ranges)
            self._metadata["Info"] = self.info_string(phase_tensor)

            yield self._metadata.image_name, phase_tensor

class TifImageReader(BaseImageReader):
    """Reader for TIFF format images."""

    def __init__(self, tif_path: str, override_pixel_size_um: Optional[float] = None) -> None:
        """Initialize the TIFF image reader.

        Args:
            tif_path: Path to the TIFF file.
            override_pixel_size_um: Optional pixel size in micrometers to override default.
        """
        super().__init__(tif_path, override_pixel_size_um)
        self._modulated_image_data: Optional[np.ndarray] = None
        self._image_data = next(self._process_image_and_metadata())[1]

    def _recycle_info_to_dict(self, info: str) -> Dict[str, str]:
        """Convert metadata info string to a dictionary.

        Args:
            info: Metadata info string.

        Returns:
            Dictionary of key-value pairs from the info string.
        """
        lines = [line for line in info.splitlines() if "[" in line]
        return {
            item.split(" = ")[0][1:-1]: item.split(" = ")[1]
            for item in lines
        }

    def _process_image_and_metadata(self, array_override: Optional[np.ndarray] = None) -> Iterator[Tuple[str, np.ndarray]]:
        """Process TIFF image data and metadata.

        Args:
            array_override: Optional array to override internal image data.

        Yields:
            Tuple of TIFF file path and image data array (TZCYX).
        """
        all_axes = "TZCYX"
        with tifffile.TiffFile(self._image_path) as tif:
            tif_data = array_override if array_override is not None else tifffile.imread(self._image_path)
            if array_override is None:
                self._image_data = tif_data

            axes = tif.series[0].axes
            metadata = tif.imagej_metadata
            self._configurations = self._recycle_info_to_dict(metadata["Info"])
            self._metadata.update(**metadata)
            self._spacing = metadata["spacing"]

            x_res_numerator, x_res_denominator = tif.pages[0].tags["XResolution"].value
            self._xresolution = float(x_res_numerator) / float(x_res_denominator)
            y_res_numerator, y_res_denominator = tif.pages[0].tags["YResolution"].value
            self._yresolution = float(y_res_numerator) / float(y_res_denominator)

            missing_axes = [i for i, dim in enumerate(all_axes) if dim not in axes]
            for missing_index in missing_axes:
                tif_data = np.expand_dims(tif_data, axis=missing_index)

            self._slice_distance = metadata["spacing"]
            self._channel_count = tif_data.shape[2]
            self._z_size = tif_data.shape[1]
            self._time_count = tif_data.shape[0]
            end = float((self._z_size - 1) * self._slice_distance)

            self._metadata.update(
                image_name=f"{self._output_folder}{os.path.sep}{self._image_name}.tif",
                x_resolution=self._xresolution,
                y_resolution=self._yresolution,
                slices=tif_data.shape[1],
                x_size=tif_data.shape[4],
                y_size=tif_data.shape[3],
                frames=tif_data.shape[0],
                time_dim=1.0,
                channels=tif_data.shape[2],
                end=end,
                begin=0.0,
            )

            ranges = image_tools.get_ranges(tif_data)
            self._metadata.update(**ranges)

            yield self._metadata.image_name, tif_data

    def save_to_tif(
        self,
        array: Optional[np.ndarray] = None,
        filepath: Optional[str] = None,
        colormap: bool = False,
    ) -> List[str]:
        """Save image array to TIFF file with metadata.

        Args:
            array: Optional array to override internal image data.
            filepath: Optional filepath for the TIFF file.
            colormap: Apply colormap to the array if True.

        Returns:
            List of file paths where TIFF files were saved.
        """
        if array is not None and colormap:
            array = image_tools.apply_colormap(array)
        tif_paths = []
        for tif_filepath, image_data in self._process_image_and_metadata(array_override=array):
            if filepath is not None:
                tif_filepath = filepath
            self._metadata["image_name"] = tif_filepath
            self._metadata["Info"] = self.info_string(image_data)
            image_tools.save_tif(image_data, tif_filepath, self._metadata.imagej_compatible_metadata)
            tif_paths.append(tif_filepath)
        return tif_paths

    @property
    def filename(self) -> str:
        """Return the full path to the TIFF file."""
        return self._image_path

class ImageReader:
    """Factory for creating image reader instances based on file extension."""

    SUPPORTED_EXTENSIONS = {"lif", "nd", "oib", "czi", "tif"}

    def __new__(cls, file_path: str, *args, **kwargs) -> BaseImageReader:
        """Create an appropriate image reader based on file extension.

        Args:
            file_path: Path to the image file.
            *args: Additional positional arguments for the reader.
            **kwargs: Additional keyword arguments for the reader.

        Returns:
            Instance of the appropriate image reader class.

        Raises:
            ValueError: If the file extension is not supported.
        """
        extension = file_path.split(".")[-1].lower()
        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}")
        readers = {
            "lif": LifImageReader,
            "nd": NdImageReader,
            "oib": OibImageReader,
            "czi": CziImageReader,
            "tif": TifImageReader,
        }
        return readers[extension](file_path, *args, **kwargs)