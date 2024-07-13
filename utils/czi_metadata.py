from utils import log_suppression

from czitools.metadata_tools.scaling import CziScaling
from czitools.metadata_tools.dimension import CziDimensions
from czitools.metadata_tools.boundingbox import CziBoundingBox
from czitools.metadata_tools.channel import CziChannelInfo
from czitools.metadata_tools.scaling import CziScaling
#from czitools.metadata_tools.sample import CziSampleInfo
from czitools.metadata_tools.objective import CziObjectives
from czitools.metadata_tools.microscope import CziMicroscope
from czitools.metadata_tools.detector import CziDetector


def get_metadata_as_dict(path, iteration=None, stepsize=None):
    """Get metadata from CZI as one dict."""

    # List of classes to instantiate
    metadata_classes = [
        CziChannelInfo,
        CziDimensions,
        CziScaling,
        #CziSampleInfo, # Unnecessary and too verbose
        CziObjectives,
        CziDetector,
        CziMicroscope,
        CziBoundingBox
    ]

    # Create an empty dictionary to store combined metadata
    combined_metadata = {}

    # Instantiate each class and gather its attributes
    for metadata_class in metadata_classes:
        # Get class name without "Czi" prefix
        class_name = metadata_class.__name__.replace('Czi', '')
        
        # Instantiate the class and get its attributes
        metadata_instance = metadata_class(path)
        metadata_dict = vars(metadata_instance)
        
        # Iterate over keys in metadata_dict and replace empty lists/dicts with None
        for key in metadata_dict:
            if isinstance(metadata_dict[key], list) and len(metadata_dict[key]) == 0:
                metadata_dict[key] = None
            elif isinstance(metadata_dict[key], dict) and len(metadata_dict[key]) == 0:
                metadata_dict[key] = None

        # If the class is CziChannelInfo, restructure parameters per channel
        if class_name == 'ChannelInfo':

            if iteration is not None and stepsize is not None:
                metadata_dict['names'] = metadata_dict['names'][iteration::stepsize]
                metadata_dict['dyes'] = metadata_dict['dyes'][iteration::stepsize]
                metadata_dict['colors'] = metadata_dict['colors'][iteration::stepsize]
                metadata_dict['clims'] = metadata_dict['clims'][iteration::stepsize]
                metadata_dict['gamma'] = metadata_dict['gamma'][iteration::stepsize]
                
            # Ensure all lists are of the same length
            num_channels = len(metadata_dict['names'])
            assert len(metadata_dict['dyes']) == num_channels
            assert len(metadata_dict['colors']) == num_channels
            assert len(metadata_dict['clims']) == num_channels
            assert len(metadata_dict['gamma']) == num_channels
            
            # Create lists for each parameter
            names = metadata_dict['names']
            dyes = metadata_dict['dyes']
            colors = metadata_dict['colors']
            clims = metadata_dict['clims']
            gamma = metadata_dict['gamma']

            # Create dictionaries for each channel
            channel_params = {}
            for i in range(num_channels):
                channel_params[f"Name{i+1}"] = names[i]
                channel_params[f"Dye{i+1}"] = dyes[i]
                channel_params[f"Color{i+1}"] = colors[i]
                channel_params[f"Clims{i+1}"] = clims[i]
                channel_params[f"Gamma{i+1}"] = gamma[i]
            
            # Update metadata_dict with the restructured channel parameters
            metadata_dict.update(channel_params)
            
            # Remove original lists from metadata_dict
            del metadata_dict['names']
            del metadata_dict['dyes']
            del metadata_dict['colors']
            del metadata_dict['clims']
            del metadata_dict['gamma']
        
        # Update combined_metadata with class_name as the key
        combined_metadata[class_name] = metadata_dict


    # Move 'czisource' to a top-level 'FileInfo' category and remove from other categories
    file_info = {'czisource': path}
    for key in combined_metadata.keys():
        if 'czisource' in combined_metadata[key]:
            del combined_metadata[key]['czisource']
    combined_metadata['FileInfo'] = file_info

    return combined_metadata

if __name__ == "__main__":

    path = r"D:\Microscopy Testing\LIF-OIB-ND-CZI_test_files\20240708_Clathrin+EEA1_-03.czi"

    metadata = get_metadata_as_dict(path)
    # Print each class name followed by its attributes on separate lines
    for class_name, attributes in metadata.items():
        print(class_name)
        for key, value in attributes.items():
            print(f"  {key}: {value}")