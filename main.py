import os
import shutil
import time
from image_tensors import ImageReader#, TifImageReader
import glob
from image_processor import ProjectAndBlendTif, DenoisingProcessor, ZProjectionProcessor


class ImageAnalyser:
    def __init__(self, image_path, override_pixel_size_um=None):
        self._image_path = image_path
        self._image_reader = ImageReader(image_path, override_pixel_size_um=override_pixel_size_um)
        self._image_reader.output_folder = r"._temp\TIF"
        self._temp_tiffs = self._image_reader.save_to_tif()
        
    def do_something_to_tensor(self):
        for tif in self._temp_tiffs:
            if not os.path.isfile(tif):
                continue
            print(f"Processing {tif}")
            tif = ProjectAndBlendTif(tif)
            tensor_shape = tif.image_data.shape
            print(f"Input tif shape; {tensor_shape}")

            result = tif._tensor_modulation(tif.image_data, separate_time_points=False, separate_channels=True, separate_z_layers=False,
                                            #selected_z=(2, 4, 6), selected_channels=(1,), selected_time=(1, 2, 3))
                                            z_range=(2, 7), channel_range=(0, 1), time_range=(1, 2))

    def save_to_tif(self) -> None:
        if self._temp_tiffs:

            for i, temp_filepath in enumerate(self._temp_tiffs):
                permanent_tif = temp_filepath.replace(r"._temp", "")
                os.makedirs(os.path.dirname(permanent_tif), exist_ok=True)
                os.rename(temp_filepath, permanent_tif)
                self._temp_tiffs[i] = permanent_tif


def main():

    print("""
    *******************************************************
    *                                                     *
    *             Welcome to BioPixel v0.3.14             *
    *                                                     *
    *    BioPixel is an advanced imaging software         *
    *    designed for biological research. This test      *
    *    release focuses on stress testing image          *
    *    conversion to TIF format, paving the way for     *
    *    future microscopy data analysis capabilities.    *
    *                                                     *
    *    Stay tuned for updates and enhancements as we    *
    *    continue to refine BioPixel to meet the needs    *
    *    of researchers and scientists worldwide.         *
    *                                                     *
    *******************************************************
    """)
    print("This TIF converter converts Lif, Oib, Metamorph(nd) and CZI files to TIF files")
    working_directory = input("Enter the directory containing microscopy images: ")

    temp_folder = fr"{working_directory}\._temp"

    os.makedirs(temp_folder, exist_ok=True)

    microscopy_images = glob.glob(f"{working_directory}\\*.*")

    # Filter only .nd, .lif, and .oib files
    microscopy_images = [f for f in microscopy_images if f.endswith(('.nd', '.lif', '.oib', '.czi'))]

    for image_path in microscopy_images:
        current = ImageAnalyser(image_path, override_pixel_size_um=0.11)
        current.save_to_tif()

    shutil.rmtree(temp_folder)

if __name__ == "__main__":
    main()





# """
# working_directory = r"D:\Microscopy Testing\New folder (2)"
# working_directory = r"D:\Microscopy Testing\New folder (2)\New folder"
# working_directory = r"D:\Microscopy Testing\CZI_test_files"

# temp_folder = fr"{working_directory}\._temp"

# os.makedirs(temp_folder, exist_ok=True)

# microscopy_images = glob.glob(f"{working_directory}\\*.*")

# # Filter only .nd, .lif, and .oib files
# microscopy_images = [f for f in microscopy_images if f.endswith(('.nd', '.lif', '.oib', '.czi'))]

# for image_path in microscopy_images:
#     current = ImageAnalyser(image_path, override_pixel_size_um=0.11)

#     #current._map_dimensions()
#     current.save_to_tif()
#     #current.do_something_to_tensor()
#     print("__________________________")

# shutil.rmtree(temp_folder)
# '''
#     # print(image_path)
#     # image = ImageReader(image_path, override_pixel_size_um=0.11)
#     # print(image)
#     # tifs = image.save_to_tif()
#     # print(tifs)
# # # Decorator to inject methods from selected MethodMixin classes into a class
# # def inject_methods(cls):
# #     print("Injecting methods into class", cls.__name__)
# #     mixins_to_inject = getattr(cls, '_mixins_to_inject', [])
# #     for mixin in mixins_to_inject:
# #         for name, method in mixin.__dict__.items():
# #             if callable(method) and not name.startswith("__") and name not in cls.__dict__:
# #                 print("Injecting method", name)
# #                 setattr(cls, name, method)
# #     return cls

# # # Class C inherits from A and injects methods from MethodMixin1 and MethodMixin2
# # #@inject_methods
# # class Cunt(TifImageReader):
# #     #_mixins_to_inject = [DenoisingProcessor, ZProjectionProcessor]

# #     def __init__(self):
# #         super().__init__()
    
# #     def inject_methods_later(self):
# #         #inject_methods(self)
# #         print("FUCKIT")


# # c = Cunt("D:\\Programming\\Test Data\\CombiLifOibNd\\TIF\\C-term431-499 EB3Red Series 1.tif")

# # print(dir(c))
# # print(type(c))
# # print(f"THIS is {c.filename}")
# # c.inject_methods_later()
# # print(dir(c))
# # #c.z_projektion()
# """