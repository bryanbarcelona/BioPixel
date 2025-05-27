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
    Ranges: Optional[tuple] = None
    min: float = 0.0
    max: float = 0.0
    spacing: float = 0.0

    @property
    def z_range(self):
        return float(abs(self.begin - self.end))

    @property
    def spacing(self):
        return float(self.z_range / (self.slices - 1) if self.slices > 1 else 0)
        
    @spacing.setter
    def spacing(self, value):
        self._spacing = value
  
class MetadataManager:
    def __init__(self):
        self._metadata = Metadata()

    @property
    def metadata(self):
        """Returns the current metadata object."""
        return self._metadata

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)
            else:
                pass

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.update(**{key: value})

    def get(self, key):
        if hasattr(self._metadata, key):
            return getattr(self._metadata, key)
        else:
            raise AttributeError(f"'Metadata' object has no attribute '{key}'")

    def reset_metadata(self):
        self._metadata = Metadata()
    
    def __repr__(self):
        return repr(self._metadata)

    @property
    def imagej_compatible_metadata(self):
        metadata={"axes": "TZCYX",  # Specify the order of the axes
                "spacing": self._metadata.spacing,  # Specify the pixel spacing
                "unit": "micron",  # Specify the units of the pixel spacing
                "hyperstack": "true",  # Specify that the data is a hyperstack
                "mode": "color",  # Specify the display mode
                "channels": self._metadata.channels,  # Specify the channel colors
                "frames": self._metadata.frames,  # Specify the number of frames
                "Info": self._metadata.Info,
                "slices": self._metadata.slices,  # Specify the number of slices
                "Ranges": self._metadata.Ranges,
                'min': self._metadata.min, 
                'max': self._metadata.max,
                "metadata": "ImageJ=1.53c\n",  # Add a blank metadata field for ImageJ
                "x_resolution": self._metadata.x_resolution,
                "y_resolution": self._metadata.y_resolution,
            }
        return metadata
    
class BaseImageReader(ABC):
    def __init__(self, image_path, override_pixel_size_um=None):
        self._image_path = image_path
        self._image_dir = os.path.dirname(image_path)
        self._image_name = os.path.splitext(os.path.basename(image_path))[0]
        self._metadata = MetadataManager()
        self._configurations = {}
        self._override_pixel_size_um = override_pixel_size_um
        self._initialize_folders()
        #_, self._image_data = self._process_image_and_metadata()
       
    def _initialize_folders(self):
        base_dir = os.path.dirname(self._image_path)

        # Default target folders
        self._originals_repo = os.path.join(base_dir, 'Original Data\\')
        self._output_folder = os.path.join(base_dir, 'TIF')

    def _gather_associated_files(self, search_pattern):

        extension = os.path.splitext(os.path.basename(self._image_path))[1].replace('.', '').upper()
  
        # Check for file existence
        if not os.path.exists(self._image_path):
            raise ValueError(f"{extension} file not found: {self._image_path}")
        
        self._associated_files = glob.glob(search_pattern)
        
        # Handle case where no associated files are found
        if not self._associated_files:
            raise ValueError(f"No associated files found for {extension} file: {self._image_name}")
        
    @abstractmethod
    def _process_image_and_metadata(self):
        pass

    def _initalize_image_data(self):
        _image_list = []
        for image in self._process_image_and_metadata():
            _image_list.append(image[0])
        self._image_data = _image_list

    @property
    def number_of_images(self):
        return len(self._image_data)
    
    @property
    def metadata(self):
        return self._metadata
    
    @property
    def image_path(self):
        return self._image_path
    
    @property
    def image_directory(self):
        return self._image_dir

    def image_data(self):
        for image_data in self._process_image_and_metadata():
            yield image_data[1]

    @property
    def filename(self):
        return self._image_name

    @property
    def associated_files(self):
        return self._associated_files.copy()
    
    @property
    def originals_repo(self):
        return self._originals_repo

    @originals_repo.setter
    def originals_repo(self, value):
        if os.path.isabs(value) and ':' in value:
            self._originals_repo = value
        else:
            self._originals_repo = os.path.join(os.path.dirname(self._image_path), value)

    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        if os.path.isabs(value) and ':' in value:
            self._output_folder = value
        else:
            self._output_folder = os.path.join(os.path.dirname(self._image_path), value)

    @property
    def configurations(self):
        return self._configurations
    
    @configurations.setter
    def update_configurations(self, new_configurations):
        """Update the configurations dictionary with new values."""
        self.configurations.update(new_configurations)

    def info_string(self, image_data):

        time_count = image_data.shape[0]
        z_size = image_data.shape[1]
        channel_count = image_data.shape[2]
        y_size = image_data.shape[3]
        x_size = image_data.shape[4]

        bits_per_pixel = image_data.dtype.itemsize * 8

        pixel_type = str(image_data.dtype)

        if image_data is not None:
            byte_order = image_data.dtype.byteorder
            if byte_order == '=' or byte_order == '|':
                if sys.byteorder == 'little':
                    byte_order_metadata = 'true'
                elif sys.byteorder == 'big':
                    byte_order_metadata = 'false'
            elif byte_order == '<':
                byte_order_metadata = 'true'
            elif byte_order == '>':
                byte_order_metadata = 'false'
            #elif byte_order == '|':
            #    byte_order_metadata = 'Not applicable'
            else:
                byte_order_metadata = 'Unknown'
        else:
            byte_order_metadata = 'No byte-order determined'
       
        leading_string = (
            f' BitsPerPixel = {bits_per_pixel}\r\n'
            f' DimensionOrder = TZCYX\r\n'
            f' IsInterleaved = false\r\n'
            f' LittleEndian = {byte_order_metadata}\r\n'
            f' PixelType = {pixel_type}\r\n'
            f' SizeC = {channel_count}\r\n'
            f' SizeT = {time_count}\r\n'
            f' SizeX = {x_size}\r\n'
            f' SizeY = {y_size}\r\n'
            f' SizeZ = {z_size}\r\n'
        )

        settings = dict(self._configurations)
        stack = [(None, settings)]
        parent_keys = []
        lines = []

        while stack:
            k, v = stack.pop()
            if k is not None:
                parent_keys.append(k)

            if isinstance(v, dict):
                # Add a marker to indicate level of the dictionary is completed
                stack.append((None, None))
                # Add the items in the dictionary to the stack
                stack.extend(sorted(v.items(), reverse=True))
            else:
                if v is not None:
                    lines.append(f"{''.join([f'[{key}]' for key in parent_keys])} = {v}\r\n")
                # We're done with this key, so remove it from the parent keys
                if parent_keys:
                    parent_keys.pop()

        return leading_string + ''.join(lines)
    
    def save_to_tif(self) -> List[str]:
        tif_paths = []
        for tif_filepath, image_data in self._process_image_and_metadata():
            self.metadata["Info"] = self.info_string(image_data)

            image_tools.save_tif(image_data, tif_filepath, self.metadata.imagej_compatible_metadata)
            tif_paths.append(tif_filepath)
        return tif_paths

    def store_originals(self):
        if isinstance(self._associated_files, str):
            file_paths = [self._associated_files]  # Convert single file path to list
        if isinstance(self._associated_files, list):
            file_paths = self._associated_files

        for file_path in file_paths:
            
            # Check if the directory exists, if not, create it
            if not os.path.exists(os.path.dirname(self.originals_repo)):
                os.makedirs(os.path.dirname(self.originals_repo))
            
            print(os.path.dirname(file_path))
            print(file_path)
            
            filename = os.path.basename(file_path)
            output_path = os.path.join(self.originals_repo, filename)
            print(output_path)
            print(f"Moved {file_path} to {output_path}")
            shutil.move(file_path, output_path)  

class LifImageReader(BaseImageReader):
    def __init__(self, lif_file_path, override_pixel_size_um=None):
        super().__init__(lif_file_path)
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}.*")
        self._lif_file = LifFile(lif_file_path)
    
    @property
    def number_of_series(self):
        return self._lif_file.num_images
    
    def _process_image_and_metadata(self):

        tif_directory = f'{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name}'
        for series_index in range(self.number_of_series):

            lif = self._lif_file.get_image(img_n=series_index)
            info = lif.info
            self.update_configurations = dict(info['settings'])

            self._metadata.update(
                image_name = f'{tif_directory} Series {series_index + 1}.tif',
                #image_name = f'{self._output_folder}\\{self._image_name} Series {series_index + 1}.tif',
                #Info = self.info_string_generator(lif),
                x_resolution=info["scale_n"].get(1, 0),
                y_resolution=info["scale_n"].get(2, 0),
                slices=info["dims_n"].get(3, 0),
                x_size=info["dims_n"].get(1, 0),
                y_size=info["dims_n"].get(2, 0),
                frames=info["dims_n"].get(4, 0),
                time_dim=info["scale_n"].get(4, 0),
                channels=info["channels"],
                end=float(info['settings']['End']) * 1e6,
                begin=float(info['settings']['Begin']) * 1e6,
            )

            # Create an empty list to hold image data
            image_data = []

            # Loop over the slices and channels and convert each Pillow object to a NumPy array
            for c in range(self._metadata["channels"]):
                channel_data = [np.array(lif.get_frame(z=z, c=c), dtype=np.uint8) for z in range(self._metadata["slices"])]

                # Stack the channel data along the z-axis to create a 3D numpy array with dimensions Z x Y x X
                channel_data = np.stack(channel_data, axis=0)

                # Append the channel data to the image data list
                image_data.append(channel_data)

            # Stack the image data along the channel axis to create a 4D numpy array with dimensions C x Z x Y x X
            image_data = np.stack(image_data, axis=1)

            # Adding a dimension for time to comply with required 'TZCYX' if the array is not 5D
            image_data = np.expand_dims(image_data, axis=0) if image_data.ndim == 4 else image_data

            #metadata.range_tuple, metadata.min_range, metadata.max_range = get_ranges(image_data)
            ranges = image_tools.get_ranges(image_data)
            self._metadata.update(**ranges)
            self.metadata["Info"] = self.info_string(image_data)
            
            yield self._metadata["image_name"], image_data

class OibImageReader(BaseImageReader):
    def __init__(self, oib_file_path, override_pixel_size_um=None):
        super().__init__(oib_file_path)
        self._oib_file = OifFile(oib_file_path)
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}.*")
        self._process_image_and_metadata()

    def close_oib_file(self):
        if self._oib_file:
            self._oib_file.close()
            self._oib_file = None
           
    def _process_image_and_metadata(self):

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

        settings_dict = dict(self._oib_file.mainfile)

        self.update_configurations = settings_dict

        image_data = self._oib_file.asarray()

        # Check and add dummy axes if needed
        for dim in range(5):
            if image_data.shape[dim] != axis_info["CZTYX"[dim]] and axis_info["CZTYX"[dim]] == 0:
                axis_info["CZTYX"[dim]] = 1
                image_data = np.expand_dims(image_data, axis=dim)

        # Transpose the image data if needed
        image_data = image_data.transpose(2, 1, 0, 3, 4)  # Rearrange the axes to TZCYX
        image_data = image_data[:, :, ::-1, :, :] # Reverse the channel order to go from highest to lowest wavelength

        tif_directory = f"{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name}.tif"

        # Initialize metadata attributes
        metadata_attrs = {
            #"image_name": f'{self._output_folder}\\{self._image_name}.tif',
            "image_name": tif_directory,
            "Info": None,
            "x_resolution": 1. / axis_info["X Conversion"],
            "y_resolution": 1. / axis_info["Y Conversion"],
            "slices": axis_info["Z"],
            "x_size": axis_info["X"],
            "y_size": axis_info["Y"],
            "frames": axis_info["T"],
            "time_dim": 1,
            "channels": axis_info["C"],
            "end": axis_info["End"],
            "begin": axis_info["Start"],
        }
        
        self._metadata.update(**metadata_attrs)
        
        ranges = image_tools.get_ranges(image_data)
        self.metadata.update(**ranges)

        #self.image_data = image_data

        self.metadata["Info"] = self.info_string(image_data)

        yield self.metadata["image_name"], image_data

    def store_originals(self):
        self._oib_file.close()
        super().store_originals()
        
class NdImageReader(BaseImageReader):
    def __init__(self, nd_image_path, override_pixel_size_um=None):
        super().__init__(nd_image_path, override_pixel_size_um=override_pixel_size_um)
        self._nd_image = None
        self._stage_count = 1
        self._parse_nd_file()
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}*")
        self._extract_metadata_from_associated_file()
        self._determine_image_file_tree()
        
    def _parse_nd_file(self):
        """Parses the ND file and extracts relevant information."""

        with open(self._image_path, "r") as f:
            lines = f.readlines()

        attributes = {}
        for line in lines:
            line = line.replace('"', '').strip()
            if line:
                key, *values = line.split(',')
                key = key.strip()
                if len(values) == 1:
                    value = values[0].strip()
                elif len(values) > 1:
                    value = '.'.join(values).strip()
            attributes[key] = value

        for key, value in attributes.items():
            if value.isdigit():
                attributes[key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                attributes[key] = float(value)
            else:
                attributes[key] = value

        self.configurations.update(attributes)

        # Extract and set dimensions
        dimensions = {
            "frames": ("DoTimelapse", "NTimePoints"),
            "channels": ("DoWave", "NWavelengths"),
            "slices": ("DoZSeries", "NZSteps"),  # Use "Z" for Z-stack size
            "stage": ("DoStage", "NStagePositions"),
            "end": ("DoZSeries", "ZStepSize"),
        }

        metadata_attributes = {}
        for dim_name, (flag_key, size_key) in dimensions.items():
            # Check for both flag and size key existence
            if flag_key in attributes and size_key in attributes:
                if attributes[flag_key].upper() == "TRUE":
                    size = attributes.get(size_key, 1)
                else:
                    size = 1
            else:
                # Assume dimension is absent if keys are missing
                size = 1
            metadata_attributes[dim_name] = size
        
        if "end" in attributes:
            if "z_size" in attributes:
                end_value = attributes["end"]
                z_size_value = attributes["z_size"]
                attributes["end"] = end_value * (z_size_value - 1)

        self._stage_count = metadata_attributes["stage"]
        self._metadata.update(**metadata_attributes)

    def _extract_metadata_from_associated_file(self):
        """Extracts metadata from the first associated TIF or STK file."""

        image_files = [
            f for f in self._associated_files if f.endswith((".tif", ".stk"))
        ]

        if image_files:
            with TiffFile(image_files[0]) as tif:
                ifd_tags = tif.pages[0].tags

                # Define relevant tags and extraction logic
                relevant_tags = (
                    256,  # ImageWidth
                    257,  # ImageLength 
                    258,  # BitsPerSample
                    277,  # SamplesPerPixel
                    259,  # Compression
                    262,  # PhotometricInterpretation
                    282,  # XResolution
                    283,  # YResolution
                    296,  # ResolutionUnit
                    305,  # Software
                    33628,  # DateTime, XCalibration, YCalibration
                )

                # Extract additional metadata with conditional logic
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
                            extracted_metadata["XResolution"] = "{:.1f}".format(float(value[0]))
                        elif tag == 283:
                            extracted_metadata["YResolution"] = "{:.1f}".format(float(value[0]))
                        elif tag == 296:
                            extracted_metadata["ResolutionUnit"] = "Centimeter" if value == 3 else None
                        elif tag == 305:
                            extracted_metadata["Software"] = str(value)
                        elif tag == 33628:
                            extracted_metadata["DateTime"] = value["CreateTime"].strftime("%Y:%m:%d %H:%M:%S")
                            if value["XCalibration"] == 0 and self._override_pixel_size_um is None:
                                extracted_metadata["XCalibration"] = 1. / (float(extracted_metadata["XResolution"]) / 10000)
                            elif self._override_pixel_size_um is not None:
                                extracted_metadata["XCalibration"] = 1. / (float(self._override_pixel_size_um))
                            else:
                                extracted_metadata["XCalibration"] = 1. / float(value["XCalibration"])
                            if value["YCalibration"] == 0 and self._override_pixel_size_um is None:
                                extracted_metadata["YCalibration"] = 1. / (float(extracted_metadata["YResolution"]) / 10000)
                            elif self._override_pixel_size_um is not None:
                                extracted_metadata["YCalibration"] = 1. / (float(self._override_pixel_size_um))
                            else:
                                extracted_metadata["YCalibration"] = 1. / float(value["YCalibration"])

                self.update_configurations = extracted_metadata

                metadata_payload = {
                        "x_size": extracted_metadata["ImageWidth"],
                        "y_size": extracted_metadata["ImageLength"],
                        "x_resolution": extracted_metadata["XCalibration"],
                        "y_resolution": extracted_metadata["YCalibration"],
                    }
                
                self._metadata.update(**metadata_payload)

    def _determine_image_file_tree(self):
        channel_count_metadata = self._metadata.metadata.channels
        time_count_metadata = self._metadata.metadata.frames
        stage_count_metadata = self._stage_count

        image_files = self._associated_files.copy()
        for file_path in image_files:
            if file_path.endswith('.nd'):
                nd_file_path, _ = os.path.splitext(file_path)
                image_files.remove(file_path)
        
        for i, file_path in enumerate(image_files):
            file_path = file_path
            image_files[i] = file_path.replace(nd_file_path, '')

        w_numbers = set()
        s_numbers = set()
        t_numbers = set()

        for file_name in image_files:
            w_match = re.search(r'_w(\d+)', file_name)
            if w_match:
                w_num = w_match.group(1)
                w_numbers.add(w_num)
            
            s_match = re.search(r'_s(\d+)', file_name)
            if s_match:
                s_num = s_match.group(1)
                s_numbers.add(s_num)
            
            t_match = re.search(r'_t(\d+)', file_name)
            if t_match:
                t_num = t_match.group(1)
                t_numbers.add(t_num)

        num_unique_w = max(len(w_numbers), 1)
        num_unique_s = max(len(s_numbers), 1)
        num_unique_t = max(len(t_numbers), 1)

        if num_unique_w != channel_count_metadata or num_unique_s != stage_count_metadata or num_unique_t != time_count_metadata:
            raise ValueError("Not all files for assembly of tif file are available.")
        else:
            pass

        ordered_image_files =[]
        for stage in range(stage_count_metadata):
            stage_list = []
            search_pattern = f'_s{stage+1}'
            stage_files = [file_name for file_name in image_files if search_pattern in file_name]
            if not stage_files:
                stage_files = image_files.copy()
            ordered_image_files.append(stage_list)
            for time in range(time_count_metadata):
                time_list = []
                search_pattern = f"_t{(time+1)}[_.]"

                time_files = [file_name for file_name in stage_files if re.search(search_pattern, file_name)]

                if not time_files:
                    time_files = stage_files.copy()
                ordered_image_files[stage].append(time_list)

                for channel in range(channel_count_metadata):
                    channel_list = []
                    search_pattern = f'_w{channel+1}'
                    channel_files = [file_name for file_name in time_files if search_pattern in file_name]
                    
                    if not channel_files:
                        channel_files = time_files.copy()
                    ordered_image_files[stage][time].append(channel_list)
                    ordered_image_files[stage][time][channel].extend(channel_files)
                    
        ordered_image_files = self._prefix_nested_list(ordered_image_files, os.path.splitext(self._image_path)[0])

        self._image_tree = ordered_image_files

    def _prefix_nested_list(self, nested_list, prefix):
        for i, item in enumerate(nested_list):
            if isinstance(item, list):
                self._prefix_nested_list(item, prefix)
            elif isinstance(item, str):
                nested_list[i] = prefix + item
        return nested_list
          
    def _process_image_and_metadata(self):
        images = self._image_tree
        filename = f'{self._image_name}'
        output_folder = f'{self.image_directory}{os.path.sep}TIF'
        stage_image_series = []
        for stage in range(len(images)):
            tif_filepath = f'{output_folder}{os.path.sep}{filename} Series {stage+1}.tif'
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
            stage_image_series.append(time_image_series)
            time_image_series = np.stack(time_image_series, axis=0)
            time_image_series = np.transpose(time_image_series, (0, 2, 1, 3, 4)) # Rearrange the axes to TZCYX
            time_image_series = time_image_series[:, :, ::-1, :, :] # Reverse the channel order to go from highest to lowest wavelength

            ranges = image_tools.get_ranges(time_image_series)
            self.metadata.update(**ranges)
            
            yield tif_filepath, time_image_series

class CziImageReader(BaseImageReader):
    def __init__(self, czi_file_path, override_pixel_size_um=None):
        super().__init__(czi_file_path)
        self._gather_associated_files(f"{self.image_directory}{os.path.sep}{self.filename}.*")
        self._czi_file = CziFile(czi_file_path)
        self._dimension_map = self._map_dimensions()
    
    def _map_dimensions(self):
        axes_sizes = self._czi_file.shape
        axes_ids = self._czi_file.axes

        if len(axes_sizes)!= len(axes_ids):
            raise ValueError("Number of axes in the CZI file does not match the number of dimensions in the image.")

        dimension_map = dict(zip(axes_ids, axes_sizes))
        return dimension_map

    def _process_image_and_metadata(self):

        tensor = self._czi_file.asarray()

        if "0" in self._dimension_map:
            position = list(self._dimension_map.keys()).index("0")
            tensor = np.squeeze(tensor, axis=position)

        if "H" in self._dimension_map:
            position = list(self._dimension_map.keys()).index("H")
            tensor = np.flip(tensor, axis=position)

            phase_tensors = []
            num_phases = self._dimension_map["H"]
            num_channels = self._dimension_map["C"]
            channels_per_phase = int(num_channels / num_phases)
            
            for phase_index in range(num_phases):
                
                phase_tensor_list = []

                for channel_offset in range(channels_per_phase):
                    channel_index = phase_index + channel_offset * num_phases
                    phase_tensor = tensor[phase_index, :, channel_index, :, :, :]
                    phase_tensor_list.append(phase_tensor)
                phase_tensor_stack = np.stack(phase_tensor_list, axis=1)
                phase_tensor_stack = np.transpose(phase_tensor_stack, (0, 2, 1, 3, 4))
                phase_tensors.append(phase_tensor_stack)
        else:
            phase_tensors = [tensor[:, :, :, :, :]]
            num_phases = 1

        for p, phase_tensor in enumerate(phase_tensors):
            self._configurations = czi_metadata.get_metadata_as_dict(self._image_path, 
                                                                     iteration=p,
                                                                     stepsize=num_phases)

            config = self._configurations
            
            mode = {0: 'Confocal', 1: 'AiryScan'}

            if num_phases == 2:
                image_name = f'{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name} {mode[p]}.tif'
            else:
                image_name = f'{self.image_directory}{os.path.sep}TIF{os.path.sep}{self._image_name} Phase {p + 1}.tif'


            self._metadata.update(
                image_name = image_name,
                #Info = self.info_string_generator(lif),
                x_resolution=1. /config["Scaling"].get('X', 0),
                y_resolution=1. /config["Scaling"].get('Y', 0),
                slices=config["Dimensions"].get('SizeZ', 0),
                x_size=config["Dimensions"].get('SizeX', 0),
                y_size=config["Dimensions"].get('SizeY', 0),
                frames=config["Dimensions"].get('SizeT', 0),
                time_dim=config["Scaling"].get('T', 1),
                channels=config["Dimensions"].get('SizeC', 0),
                end=float(config["Scaling"].get('Z', 0)*(config["Dimensions"].get('SizeZ', 0) - 1)),
                begin=0.0,
            )

            ranges = image_tools.get_ranges(phase_tensor)
            self._metadata.update(**ranges)
            self.metadata["Info"] = self.info_string(phase_tensor)

            yield self._metadata["image_name"], phase_tensor
        
class TifImageReader(BaseImageReader):

    def __init__(self, tif_path, override_pixel_size_um=None):
        super().__init__(tif_path)
        self._modulated_image_data = None
        self._image_data = next(self._process_image_and_metadata())[1]

    def _recylce_info_to_dict(self, info):
        lines = info.splitlines()
        lines = [line for line in lines if "[" in line]
        configuration_dict = {}
        for item in lines:
            key_value_pair = item.split(' = ')
            key = key_value_pair[0][1:-1]  # remove brackets from the key
            value = key_value_pair[1]
            configuration_dict[key] = value

        return configuration_dict

    def _process_image_and_metadata(self, array_override=None):
        """
        Loads a multi-dimensional TIFF file, parses dimensions, and adds singletons for 5D compliance.

        Returns:
            tuple: (data, metadata, xy_cal, config)
        """

        all_axes = 'TZCYX'

        with tifffile.TiffFile(self._image_path) as tif:
            if array_override is not None:
                tif_data = array_override
            else:
                tif_data = tifffile.imread(self._image_path)
                self._image_data = tif_data

            axes = tif.series[0].axes

            metadata = tif.imagej_metadata
            self._configurations = self._recylce_info_to_dict(metadata["Info"])
            self._metadata.update(**metadata)
            self._spacing = metadata["spacing"]

            x_res_numerator, x_res_denominator = tif.pages[0].tags['XResolution'].value
            self._xresolution = float(x_res_numerator) / float(x_res_denominator)

            y_res_numerator, y_res_denominator = tif.pages[0].tags['YResolution'].value
            self._yresolution = float(y_res_numerator) / float(y_res_denominator)

            missing_axes = []
            for i, dimension in enumerate(all_axes):
                if dimension not in axes:
                    missing_axes.append(i)

            for missing_index in missing_axes:
                tif_data = np.expand_dims(tif_data, axis=missing_index)

            self._slice_distance = metadata["spacing"]
            self._channel_count = tif_data.shape[2]
            self._z_size = tif_data.shape[1]
            self._time_count = tif_data.shape[0]

            end = float((self._z_size - 1) * self._slice_distance)

            self._metadata.update(
                image_name = f'{self._output_folder}\\{self._image_name}.tif',
                #Info = self.info_string_generator(lif),
                x_resolution=self._xresolution,
                y_resolution=self._yresolution,
                slices=tif_data.shape[1],
                x_size=tif_data.shape[4],
                y_size=tif_data.shape[3],
                frames=tif_data.shape[0],
                time_dim=1,
                channels=tif_data.shape[2],
                end=end,
                begin=0.0,
            )
            
            ranges = image_tools.get_ranges(tif_data)
            self.metadata.update(**ranges)
            
            yield self._metadata["image_name"], tif_data

    def save_to_tif(self, array=None, filepath=None, colormap=False) -> List[str]:
        """
        Save the image array to a TIFF file with metadata.

        Parameters:
        array (np.ndarray, optional): An optional array to override the internal image data.
        filepath (str, optional): An optional filepath to save the TIFF file to.

        Returns:
        List[str]: A list of file paths where the TIFF files were saved.
        """
        if array is not None and colormap is True:
            array = image_tools.apply_colormap(array)
        tif_paths = []
        for tif_filepath, image_data in self._process_image_and_metadata(array_override=array):
            if filepath is not None:
                tif_filepath = filepath
            self.metadata["image_name"] = tif_filepath
            self.metadata["Info"] = self.info_string(image_data)
            image_tools.save_tif(image_data, tif_filepath, self.metadata.imagej_compatible_metadata)
            tif_paths.append(tif_filepath)
        return tif_paths
       
    @property
    def filename(self):
        return self._image_path
    
class ImageReader:
    def __new__(cls, file_path, *args, **kwargs):
        file_path = str(file_path)
        extension = file_path.split('.')[-1].lower()
        if extension == 'lif':
            return LifImageReader(file_path, *args, **kwargs)
        elif extension == 'nd':
            return NdImageReader(file_path, *args, **kwargs)
        elif extension == 'oib':
            return OibImageReader(file_path, *args, **kwargs)
        elif extension == 'czi':
            return CziImageReader(file_path, *args, **kwargs)
        elif extension == 'tif':
            return TifImageReader(file_path, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")