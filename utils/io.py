import os

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