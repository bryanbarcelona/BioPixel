import os
import sys

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

#print("Loading Shutil module...", flush=True)
write_status("Loading Shutil module")
import shutil

#print("Loading Time module...", flush=True)
write_status("Loading Time module")
import time

#print("Loading Image Reader...", flush=True)
write_status("Loading Image Reader")
from image_tensors import ImageReader, TifImageReader

#print("Loading Glob module...", flush=True)
write_status("Loading Glob module")
import glob

#print("Loading Image Processors...", flush=True)
write_status("Loading Image Processors")
from image_processor import ProjectAndBlendTif

#print("Loading Image Tools...", flush=True)
write_status("Loading Image Tools")
from utils import image_tools
from utils.cropper import Cropper

#print("Loading Cell Detector...", flush=True)
write_status("Loading Cell Detector")
#from cell_detector import detect_cells
import cell_detector
#print("Loading Tiff File Handler...", flush=True)
write_status("Loading Tiff File Handler")
import tifffile

#print("Loading Tiff File Handler...", flush=True)
write_status("Loading Tiff File Handler")
import cv2
import numpy as np
import math

# You can add additional initialization or setup code here if needed

#print("BioPixel Main initialized successfully!")
write_status("END")

class BioPixelEntry:
    def __init__(self):
        self._working_directory = None
        self._images = None
        self._save_to_tif = False
        self._detect_cells = False
        self.current_instance = None
    
    def __getattr__(self, name):
        if self.current_instance and hasattr(self.current_instance, name):  # Modify this line
            return getattr(self.current_instance, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    @property
    def images(self):
        return self._images
    
    @images.setter
    def images(self, images_list):
        self._images = images_list
    
    def set_working_directory(self, working_directory):
        self._working_directory = working_directory
    
        images = glob.glob(f"{working_directory}\\*.*")

        images = [f for f in images if f.endswith(('.nd', '.lif', '.oib', '.czi'))]

        self._images = images

    def setup_processing(self):
        if self._working_directory is None:
            return

        working_directory = self._working_directory
        temp_folder = fr"{working_directory}\._temp"
        os.makedirs(temp_folder, exist_ok=True)

        return temp_folder
    
    def process_images(self, keep_tif=False, detect_cells=False):
        if not self._images:
            return
        
        temporary_directory = self.setup_processing()

        for image_path in self._images:
            self.current_instance = ImageAnalyser(image_path, override_pixel_size_um=0.11)
            if detect_cells:
                self.current_instance.segment_cells(keep_single_cells=True)
            if keep_tif:
                self.current_instance.save_to_tif()
            
            #self.current_instance.print_shape()

        shutil.rmtree(temporary_directory)
        self.current_instance = None 

class ImageAnalyser:
    def __init__(self, image_path, override_pixel_size_um=None):
        self._image_path = image_path
        self._image_reader = ImageReader(image_path, override_pixel_size_um=override_pixel_size_um)
        self._image_reader.output_folder = r"._temp\TIF"
        self._temp_tiffs = self._image_reader.save_to_tif()
        self.masks = None
        self.masks_path = None

    def __getattr__(self, name):
        if self._image_reader and hasattr(self._image_reader, name):
            return getattr(self._image_reader, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
    def detect_cells(self, keep_png=False, keep_masks=False, tif_file=None) -> None:
        if tif_file is not None:
            tif_files = [tif_file]
        else:
            tif_files = self._temp_tiffs
        
        for tif_path in tif_files:
            if not os.path.isfile(tif_path):
                continue
            
            tifname = os.path.basename(tif_path)

            if keep_masks:
                mask_output = f'{self.image_directory}{os.path.sep}MASKS'
            else:
                mask_output = f'{self.image_directory}{os.path.sep}._temp{os.path.sep}MASKS'

            tif = ProjectAndBlendTif(tif_path)
         
            max_projections = tif.max_project(provide_filename=True)

            for image_name, image_array in max_projections.items():
                os.makedirs(os.path.dirname(mask_output), exist_ok=True)
                diameter = int(tif._resolution_x * 30)
                self.masks = cell_detector.detect_cells(image_array, filename=tif_path, output_folder=mask_output, diameter=diameter)
                self.masks_path = mask_output
    
    def segment_cells(self, keep_single_cells=False):
        cropped_cell_files = []
        for tif_file in self._temp_tiffs:
            if not os.path.isfile(tif_file):
                continue

            if keep_single_cells:
                segmented_output = f'{self.image_directory}{os.path.sep}SINGLE CELLS'
            else:
                segmented_output = f'{self.image_directory}{os.path.sep}._temp{os.path.sep}SINGLE CELLS'

            os.makedirs(segmented_output, exist_ok=True)
            tifname = os.path.splitext(os.path.basename(tif_file))[0]

            self.detect_cells(keep_png=False, keep_masks=True, tif_file=tif_file)

            if self.masks is None:
                return

            image_array = tifffile.imread(tif_file)

            crop = Cropper(self.masks, image_array)

            for c, cell in enumerate(crop.cropped_cells):
                output_path = os.path.join(segmented_output, f"{tifname}_Cell_{c}.tif")
                TifPipe = TifImageReader(tif_file)
                TifPipe.save_to_tif(array=cell, filepath=output_path)
                cropped_cell_files.append(output_path)
            
        self._segmented_cell_tifs = cropped_cell_files



    # def print_shape(self):

    #     def point_in_circle(px, py, cx, cy, radius):
    #         # Calculate the distance between the point (px, py) and the center of the circle (cx, cy)
    #         distance = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    #         return distance < radius

    #     def get_signals(image_array, i=0):
    #         # Apply a binary threshold to get a binary image
    #         _, binary_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    #         binary_array.append(binary_image)

    #         # Find contours in the binary image
    #         contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         # Filter contours by area to remove small noise contours
    #         min_contour_area = 2
    #         max_contour_area = 100
    #         contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]

    #         #z_map = []
    #         for cnt in contours:
    #             area = cv2.contourArea(cnt)
    #             (x, y), radius = cv2.minEnclosingCircle(cnt)
    #             center = (int(x), int(y))
    #             radius = int(radius)
    #             signal = (i, center, radius, area, cnt)
    #             #z_map.append(signal)
    #             all_circles.append(signal)
            
    #         return all_circles

    #     for image in self._segmented_cell_tifs:
    #         no_ext = os.path.splitext(image)[0]
    #         output_path = f"{no_ext}_contours.tif"
    #         #tif = TifImageReader(image)
    #         tif = ProjectAndBlendTif(image)
    #         max_projection = tif.max_project()

    #         image_data = tif.image_data
    #         image_data = image_data[0, :, 0, :, :]

    #         contour_array = []
    #         binary_array = []
    #         all_circles = []

    #         print(type(max_projection), max_projection.shape)
    #         #signals = get_signals(max_projection)
    #         #all_circles.extend(signals)
    #         # Extract circles from each z-plane and store them in all_circles
    #         for i in range(image_data.shape[0]):
    #             two_dim_image = image_data[i, :, :]

    #             signals = get_signals(two_dim_image, i=i)

    #             all_circles.extend(signals)
    #             # # Apply a binary threshold to get a binary image
    #             # _, binary_image = cv2.threshold(two_dim_image, 127, 255, cv2.THRESH_BINARY)
    #             # binary_array.append(binary_image)

    #             # # Find contours in the binary image
    #             # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #             # # Filter contours by area to remove small noise contours
    #             # min_contour_area = 0
    #             # max_contour_area = 100
    #             # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]

    #             # #z_map = []
    #             # for cnt in contours:
    #             #     area = cv2.contourArea(cnt)
    #             #     (x, y), radius = cv2.minEnclosingCircle(cnt)
    #             #     center = (int(x), int(y))
    #             #     radius = int(radius)
    #             #     signal = (i, center, radius, area, cnt)
    #             #     #z_map.append(signal)
    #             #     all_circles.append(signal)

    #             #signal_map.append(z_map)
    #         print(f"Found {len(all_circles)} circles")
    #         # # Sort circles by size (radius)
    #         # all_circles.sort(key=lambda x: x[2], reverse=False)

    #         # # Filter out larger circles if a smaller one is inside
    #         # filtered_circles = []
    #         # for circle in all_circles:
    #         #     if not any(point_in_circle(circle[1][0], circle[1][1], c[1][0], c[1][1], c[2]) for c in filtered_circles):
    #         #         filtered_circles.append(circle)

    #         # # Assign circles to the most intense z-plane
    #         # intensity_map = {}
    #         # for circle in filtered_circles:
    #         #     z, center, radius, area, cnt = circle
    #         #     if (z, center) not in intensity_map or radius > intensity_map[(z, center)]:
    #         #         intensity_map[(z, center)] = radius

    #         # # Draw the circles only on the most intense z-plane
    #         # for i in range(image_data.shape[0]):
    #         #     two_dim_image = image_data[i, :, :]
    #         #     image_with_contours = cv2.cvtColor(two_dim_image, cv2.COLOR_GRAY2BGR)
    #         #     for (z, center), radius in intensity_map.items():
    #         #         if i == z:
    #         #             #color = tuple(np.random.randint(0, 255, size=3).tolist())
    #         #             color = (0, 255, 0)
    #         #             cv2.circle(image_with_contours, center, radius, color, 1)
    #         #     contour_array.append(image_with_contours)

    #         # # Save the result
    #         # np.stack(contour_array, axis=0)
    #         # tifffile.imwrite(output_path, contour_array)
    # # def print_shape(self):

    # #     def point_in_circle(px, py, cx, cy, radius):
    # #         # Calculate the distance between the point (px, py) and the center of the circle (cx, cy)
    # #         distance = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            
    # #         # Check if the distance is less than the radius of the circle
    # #         if distance < radius:
    # #             return True
    # #         else:
    # #             return False
            
    # #     for image in self._segmented_cell_tifs:

    # #         no_ext = os.path.splitext(image)[0]
    # #         output_path = f"{no_ext}_contours.tif"
    # #         tif = TifImageReader(image)
    # #         image_data = tif.image_data
    # #         image_data = image_data[0, :, 0, :, :]

    # #         contour_array = []
    # #         binary_array = []
    # #         signal_map = []
    # #         for i in range(image_data.shape[0]):

    # #             two_dim_image = image_data[i, :, :]

    # #             # Apply a binary threshold to get a binary image
    # #             _, binary_image = cv2.threshold(two_dim_image, 127, 255, cv2.THRESH_BINARY)

    # #             binary_array.append(binary_image)
    # #             # Find contours in the binary image
    # #             contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # #             # Filter contours by area to remove small noise contours
    # #             min_contour_area = 0  # Minimum area to consider a contour as valid
    # #             max_contour_area = 100
    # #             contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]
                
    # #             # # Draw contours on the original image
    # #             # # Create a copy of the original image to draw the contours on
    # #             image_with_contours = cv2.cvtColor(two_dim_image, cv2.COLOR_GRAY2BGR)
    # #             # cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # #             z_map = []
    # #             # Draw each contour with a different color
    # #             for j, cnt in enumerate(contours):
    # #                 #print(len(cnt))
    # #                 area = cv2.contourArea(cnt)
    # #                 (x,y),radius = cv2.minEnclosingCircle(cnt)
    # #                 center = (int(x),int(y))
    # #                 radius = int(radius)
    # #                 #color = tuple(np.random.randint(0, 255, size=3).tolist())  # Generate a random color
    # #                 #cv2.circle(image_with_contours,center,radius,color,2)
    # #                 signal = (j, center, radius, area, cnt)
    # #                 z_map.append(signal)

    # #                 #cv2.drawContours(image_with_contours, [cnt], -1, color, 2)

    # #             signal_map.append(z_map)
                
    # #             #contour_array.append(image_with_contours)
    # #             #print(type(image_with_contours))
    # #             # Display the result
    # #             # plt.figure(figsize=(10, 10))
    # #             # plt.imshow(image_with_contours)
    # #             # plt.title("Contours")
    # #             # plt.axis('off')
    # #             # plt.show()
    # #         i = 0

    # #         for z in signal_map:
    # #             for signal in z:
    # #                 i+=1

    # #         print(f"Initally {i} detected.")
    # #         # signal_id , centercord, radius, area, contour

    # #         new_id_list = []
    # #         for i, zlayer in enumerate(signal_map):
    # #             if i == len(signal_map) - 1:
    # #                 break

    # #             for j, signal in enumerate(zlayer):
    # #                 if i == 0:
    # #                     new_id_list.append(signal)
    # #                     continue

    # #                 for k, signal_ahead in enumerate(signal_map[i+1]):
    # #                     if point_in_circle(signal_map[i][j][1][0], signal_map[i][j][1][1], signal_map[i+1][k][1][0], signal_map[i+1][k][1][1], signal_map[i][j][2]):
    # #                         if signal_map[i][j][3] > signal_ahead[3]:
    # #                             new_id_list.append(signal_ahead)
    # #                         else:
    # #                             new_id_list.append(signal)

    # #         print(len(new_id_list))
    # #         new_id_dict = {}
    # #         for id in new_id_list:
    # #             color = tuple(np.random.randint(0, 255, size=3).tolist())
    # #             new_id_dict[color] = id

    # #         for i in range(image_data.shape[0]):
    # #             two_dim_image = image_data[i, :, :]
    # #             image_with_contours = cv2.cvtColor(two_dim_image, cv2.COLOR_GRAY2BGR)
    # #             for color, cnt in new_id_dict.items():
    # #                 cv2.circle(image_with_contours,cnt[1],cnt[2],color, 1)
    # #             contour_array.append(image_with_contours)

    # #         # for i in range(second_channel.shape[0]):
    # #         #     two_dim_image = second_channel[i, :, :]
    # #         #     image_with_contours = cv2.cvtColor(two_dim_image, cv2.COLOR_GRAY2BGR)
    # #         #     for j, cnt in enumerate(new_id_list):
    # #         #         color = tuple(np.random.randint(0, 255, size=3).tolist())  # Generate a random color
    # #         #         cv2.circle(image_with_contours,cnt[1],cnt[2],color,2)
    # #         #     contour_array.append(image_with_contours)

    # #         np.stack(contour_array, axis=0)
    # #         #np.stack(binary_array, axis=0)

    # #         #tif.save_to_tif(array=contour_array, filepath=output_path)
    # #         tifffile.imwrite(output_path, contour_array)

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
    *    Thank you for using BioPixel. We hope you find   *
    *    it useful.                                       *
    *                                                     *      
    *    (c) 2024 BioPixel. All rights reserved.          *
    *******************************************************
    """)
    print("This TIF converter converts Lif, Oib, Metamorph(nd) and CZI files to TIF files")
    working_directory = input("Enter the directory containing microscopy images: ")

    entry = BioPixelEntry()
    entry.set_working_directory(working_directory)

    entry.process_images(keep_tif=True, detect_cells=False)

    # temp_folder = fr"{working_directory}\._temp"

    # os.makedirs(temp_folder, exist_ok=True)

    # microscopy_images = glob.glob(f"{working_directory}\\*.*")

    # # Filter only .nd, .lif, and .oib files
    # microscopy_images = [f for f in microscopy_images if f.endswith(('.nd', '.lif', '.oib', '.czi'))]

    # for image_path in microscopy_images:
    #     current = ImageAnalyser(image_path, override_pixel_size_um=0.11)
    #     #current.detect_cells(keep_png=True, keep_masks=True)

    #     current.save_to_tif()

    # shutil.rmtree(temp_folder)

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