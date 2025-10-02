import os
import sys
import inspect
from pathlib import Path


def get_resource_path(relative_path=None):
    """
    Get the absolute path to a resource or the base path if no relative path is provided.

    Parameters:
    -----------
    relative_path : Path or str, optional
        The relative path to the resource. This path should be relative to the base path
        determined by `sys._MEIPASS2` or the fallback path. The fallback path is the
        directory of the script or module that calls this function, not the `io.py` module.
        If no relative path is provided, the function returns the base path.

    Returns:
    --------
    Path
        The absolute path to the resource or the base path if no relative path is provided.

    Notes:
    ------
    - When the script is bundled with PyInstaller and executed as a single executable file
      using the `--onefile` and `--add-data` flags, PyInstaller sets the `sys._MEIPASS2`
      attribute to the temporary directory where the bundled files are extracted. This is
      the base path in this context.
    - If `sys._MEIPASS2` is not available (e.g., during development or when running the
      script directly), the fallback path is the directory of the script that calls this
      function. This ensures that the relative paths are resolved correctly regardless of
      the location of the `io.py` module.
    """
    try:
        base_path = Path(sys._MEIPASS2)
    except Exception:
        # Get the directory of the script that calls this function
        frame = inspect.currentframe().f_back
        base_path = Path(frame.f_globals['__file__']).parent

    if relative_path is None:
        return base_path
    else:
        return (base_path / relative_path).resolve()

def get_executable_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def write_status(update):
    status_file_path = os.path.join(get_executable_path(), "startup_status")
    with open(status_file_path, "w") as f:
        f.write(update + "\n")
        f.flush()

def create_directories(path, *subdirs, directory_depth=0):
    """
    Create one or multiple subdirectories within the given parent directory.

    Args:
        path (str): The path to the parent directory or file.
        *subdirs (str): Variable number of subdirectories to create.
        directory_depth (int, optional): Number of directory levels to move 
            up from the provided path before creating the subdirectories. 
            Default is 0.

    Returns:
        list: List of paths of the created subdirectories.

    Note:
        This function can also be used as a fail-safe mechanism within 
        functions that require specific directories to be already created. 
        If the specified subdirectories do not exist, this function will
        create them before returning.

    """
    if os.path.isfile(path):
        parent_dir = os.path.dirname(path)
    else:
        parent_dir = path
    
    for _ in range(directory_depth):
        parent_dir = os.path.dirname(parent_dir)

    created_subdirs = []
    
    # Create subdirectories
    for subdir in subdirs:
        subdir_path = os.path.join(parent_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        created_subdirs.append(subdir_path)

    return created_subdirs

def generate_output_path(input_path, subdirectory=None, directory_depth=0, prefix=None, 
                        suffix=None, output_extension=None, 
                        custom_directory=None):
    """
    Generate an output path based on the provided options.

    Args:
        input_path (str): The original path from which to generate the 
                                output path.
        subdirectory (str, optional):   Optional subdirectory to append 
                                to the path. Default is None.
        prefix (str, optional): Optional prefix to prepend to the filename with connecting underscore.
                                Default is None.
        suffix (str, optional): Optional suffix to append to the filename with connecting underscore. 
                                Default is None.
        output_extension (str, optional): Optional new file extension to replace 
                                the original extension. Default is None.
        custom_directory (str, optional): Optional custom directory where the 
                                output path should be created. If provided, this 
                                overrides the parent directory derived from the 
                                input_path. Default is None.

    Returns:
        str: The generated output path.
    """
    if custom_directory:
        base_dir = custom_directory
    else:
        #base_dir = os.path.dirname(input_path)
        #for _ in range(directory_depth):
        #    base_dir = os.path.dirname(input_path)

        if os.path.isfile(input_path):
            base_dir = os.path.dirname(input_path)
        else:
            base_dir = input_path
        
        for _ in range(directory_depth):
            base_dir = os.path.dirname(base_dir)

    filename, extension = os.path.splitext(os.path.basename(input_path))

    if subdirectory:
        base_dir = os.path.join(base_dir, subdirectory)

    if prefix:
        filename = f"{prefix}_{filename}"

    if suffix:
        filename = f"{filename}_{suffix}"

    if output_extension:
        extension = f".{output_extension}"

    output_path = os.path.join(base_dir, f"{filename}{extension}")

    return output_path

def get_filename(path, extension=False):

    basename, ext = os.path.splitext(os.path.basename(path))

    if extension:
        filename = f"{basename}{ext}"
    else:
        filename = basename
    
    return filename


class DirectoriesManager:
    """
    Manages directory paths, automatically detecting the instantiating script's location.
    
    - Supports PyInstaller (`sys._MEIPASS2`) for bundled execution.
    - Defaults to the directory of the script that instantiated this class.
    - Allows fetching absolute paths for resources within this directory.

    Parameters:
    -----------
    base_directory : str or Path, optional
        If provided, this path is used as the base directory. Otherwise, the path is
        automatically determined from `sys._MEIPASS2` (if running as an executable) or
        the script that instantiated this class.
    """
    
    def __init__(self, base_directory=None):
        instantiation_location = Path(inspect.currentframe().f_back.f_globals['__file__']).resolve()
        self._base_directory = self._get_base_directory(base_directory, instantiation_location)
        self._basename = self._get_basename(base_directory)
        self.series_count = 0
        self.cell_count = 0

    @property
    def base_directory(self):
        return self._base_directory
    
    @base_directory.setter
    def base_directory(self, new_base_directory):
        self._base_directory = self._get_base_directory(new_base_directory)
        self._basename = self._get_basename(new_base_directory)

    @property
    def series_number(self):
        return self.series_count
    
    @series_number.setter
    def series_number(self, new_series_number):
        self.series_count = new_series_number
    
    @property
    def cell_number(self):
        return self.cell_count
    
    @cell_number.setter
    def cell_number(self, new_cell_number):
        self.cell_count = new_cell_number

    @property
    def tifpath(self):
        return self._generate_output_path("tif")
    
    @property
    def csvpath(self):
        return self._generate_output_path("csv")
    
    def _get_base_directory(self, base_directory=None, instantiation_location=None):
        """Determine the base directory, considering PyInstaller or the instantiating script."""
        if base_directory is not None:
            base_path = Path(base_directory).resolve()
            if base_path.is_file():  # If a file is given, return the directory of that file
                return base_path.parent
            return base_path

        # Fallback: Use the script's location if no user-provided base directory
        return Path(instantiation_location).parent.resolve()

    def _get_basename(self, user_input=None):
        """
        Determines the base name based on the given `user_input` path.
        
        - If `user_input` is a file and exists, returns its name without extension.
        - If `user_input` is a directory, returns the directory name.
        - If `user_input` is None, falls back to `self.base_directory`.
        - If the path doesn't exist or is invalid, returns "default_name".

        Parameters:
        -----------
        user_input : str or Path, optional
            A file or directory path to determine the base name from.

        Returns:
        --------
        str
            - File name without extension if `user_input` is a valid file.
            - Directory name if `user_input` is a directory.
            - "default_name" if the path is invalid or doesn't exist.
        """
        base_path = Path(user_input).resolve() if user_input else self.base_directory

        if base_path.exists() and base_path.is_file():
            return base_path.stem
        elif base_path.is_dir():
            return base_path.name

        return "image"
           
    def _generate_output_path(self, output_type="tif"):
        """
        Generate the output path for the current series and cell count.
        
        Returns:
        --------
        Path
            The output path for the current series and cell count.
        """
        output_types = {
            "tif": self._base_directory / "tif" / f"{self._basename}_Series_{self.series_count}_Cell_{self.cell_count}.tif",
            "csv": self._base_directory / f"{self._basename}_Series_{self.series_count}_Cell_{self.cell_count}.csv"
        }

        return output_types.get(output_type, output_types["tif"])

    def get_resource_path(self, relative_path=None):
        """
        Get the absolute path to a resource relative to the base directory.
        If bundled with PyInstaller, use `sys._MEIPASS2` to resolve paths.
        
        Parameters:
        -----------
        relative_path : str or Path, optional
            A relative path within the base directory. If None, returns the base directory.

        Returns:
        --------
        Path
            Absolute path to the requested resource or the base directory.
        """
        try:
            # Check if PyInstaller has set this attribute (used in bundled mode)
            base_path = Path(sys._MEIPASS2)
        except Exception:
            # If not bundled, use the base directory from class initialization
            base_path = self.base_directory

        if relative_path is None:
            return base_path
        return (base_path / relative_path).resolve()

class ExperimentHandler:
    def __init__(self, base_directory=None):
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()

    def get_microscopy_files(self, directory=None):
        if directory is None:
            directory = self.base_directory

        all_files = list(Path(directory).iterdir())
        nd_basenames = {f.stem for f in all_files if f.is_file() and f.suffix.lower() == '.nd'}

        microscopy_files = []
        for f in all_files:
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in ['.lif', '.czi', '.nd', '.oib']:
                microscopy_files.append(f)
            elif ext == '.tif':
                if not any(f.name.startswith(nd) for nd in nd_basenames):
                    microscopy_files.append(f)

        return microscopy_files
    
    @property
    def each_file_is_one_experiment(self):
        """List of experiment dicts (one per file)."""
        experiments = []
        for file in self.get_microscopy_files():
            meta = self.get_experiment_metadata(file, channel_names=["F-Actin"], from_filename=1)
            meta["files"] = [file]
            experiments.append(meta)
        return experiments

    @property
    def each_folder_is_one_experiment(self):
        experiments = []
        for folder in [self.base_directory] + [p for p in self.base_directory.rglob('*') if p.is_dir()]:
            files = self.get_microscopy_files(folder)
            if not files:
                continue
            label = folder.name
            meta = self.get_experiment_metadata(
                files,
                channel_names=["F-Actin"],
                from_filename=1,
                label_override=label
            )
            meta["files"] = files
            experiments.append(meta)
        return experiments

    def get_experiment_metadata(self, files, channel_names=None, from_filename=None, label_override=None):
        if isinstance(files, (str, Path)):
            files = [Path(files)]

        main_file = files[0]
        label = label_override if label_override else main_file.stem

        # Default channel names
        if channel_names is None:
            channel_names = ["from_filename", "DAPI", "Hoechst"]

        # If from_filename index is provided, inject the placeholder
        if from_filename is not None:
            channel_names = channel_names[:]
            channel_names.insert(from_filename, "from_filename")

        # Check that 'from_filename' appears exactly once
        if channel_names.count("from_filename") != 1:
            raise ValueError("Exactly one channel must be filled from the filename label using 'from_filename'")

        channels = {}
        for i, name in enumerate(channel_names):
            channel_label = label if name == "from_filename" else name
            clean_channel_label = self.clean_channel_name(channel_label)
            channels[i] = {
                "name": clean_channel_label,
                "plot_path": self.base_directory / f"{label}_{channel_label}.png"
            }

        return {
            "label": label,
            "files": files,
            "pickle": self.base_directory / f"{label}.bppfx",
            "channels": channels
        }

    def clean_channel_name(self, name):
        """
        Replace special substrings in channel names for prettier display.
        Extend this method as needed.
        """
        if "delta" in name:
            name = name.replace("delta", r"$\Delta$")
        return name
   
    def read_file(self, filename):
        """Read a file and return its contents."""
        with open(self.base_directory / filename, 'r') as f:
            return f.read()

    def write_file(self, filename, content):
        """Write content to a file."""
        with open(self.base_directory / filename, 'w') as f:
            f.write(content)

if __name__ == "__main__":

    # dir = r"D:\Microscopy Testing\20250323 Revisiting Image_Tensors to Focus on direct Output of Arrays\Clathrin vs Drbrin Isotype Control.lif"

    # dm = DirectoriesManager()
    # #dm = MyClass()
    # print("Here we go")
    # print(dm.base_directory)
    # print(dm.get_resource_path("Freedom\FUCK"))
    dir = r"D:\Microscopy Testing\20240710 LIF-OIB-ND-CZI_test_files"
    dir = r"D:\Microscopy Testing\20250406 FULL PODOPROFILE RUN"
    files = ExperimentHandler(dir).each_file_is_one_experiment
    print(files)
    files = ExperimentHandler().each_file_is_one_experiment
    print("Files found:")
    for file in files:
        print(file)