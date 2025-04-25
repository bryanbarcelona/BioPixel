import numpy as np
import cv2
import tifffile
import pickle
from typing import List, Dict, Tuple, Optional, Generator, Literal, Union, Any
from dataclasses import dataclass, field
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

@dataclass
class PLAData:
    """
    Class to hold the data for PLA analysis.
    """
    donors = 0
    control_donors = 0
    treatment_donors = 0
    views = 0
    control_views = 0
    treatment_views = 0
    cells = 0
    control_cells = 0
    treatment_cells = 0
    signals_count = 0
    control_signals = 0
    treatment_signals = 0
    podosome_associated_signals = 0
    control_podosome_associated_signals = 0
    treatment_podosome_associated_signals = 0
    non_podosome_associated_signals = 0
    control_non_podosome_associated_signals = 0
    treatment_non_podosome_associated_signals = 0
    signals: List[Dict[str, Any]] = field(default_factory=list)


class Visualization:
    def __init__(self, image_data):
        self._image_data = image_data
       
    def save_image(self, filename: str, podosomes: list = None, signals: list = None):
        """
        Save the image data to a file, optionally overlaying podosomes and/or signals.

        Parameters:
        filename (str): The name of the file to save the image.
        podosomes (list): Optional podosome data to draw.
        signals (list): Optional signal data to draw.
        """
        actin_channel = self._image_data[0, :, 1, :, :]
        pla_channel = self._image_data[0, :, 0, :, :]

        # Base grayscale views
        grayscale_4d_actin = np.expand_dims(actin_channel, axis=1)  # (Z, 1, Y, X)
        grayscale_4d_pla = np.expand_dims(pla_channel, axis=1)
        grayscale_5d_actin = np.stack([grayscale_4d_actin] * 3, axis=-1)  # (Z, 1, Y, X, 3)
        grayscale_5d_pla = np.stack([grayscale_4d_pla] * 3, axis=-1)

        # Layers to include in final stack
        layers = [grayscale_5d_actin, grayscale_5d_pla]

        # Optional podosome overlays
        if podosomes:
            data_without_centroid = self.draw_podosomes_on_3d_array(
                actin_channel,
                podosomes,
                draw_mode="contour",
                draw_centroid=False,
                draw_dilation=False,
            )
            data = self.draw_podosomes_on_3d_array(
                actin_channel,
                podosomes,
                draw_mode="contour",
                draw_centroid=True,
                draw_dilation=True,
            )
            layers.extend([data_without_centroid, data])

        # Add PLA grayscale
        layers.append(grayscale_5d_pla)

        # Optional signal dots
        if signals:
            data2 = self.draw_dots_on_mask(
                pla_channel,
                signals,
                radius=5,
                solid=True,
            )
            layers.append(data2)

        # Concatenate and save
        stacked_array = np.concatenate(layers, axis=1)
        tifffile.imwrite(filename, stacked_array, imagej=True, metadata={'axes': 'ZCYXS'})

    def draw_podosomes_on_3d_array(self, 
        rgb_3d: np.ndarray, 
        podosomes, 
        draw_mode: str = "contour", 
        alpha: float = 0.5,
        draw_centroid: bool = False,
        draw_dilation: bool = False,
    ) -> np.ndarray:
        
        """
        Draw podosomes on a 3D RGB array.

        Parameters:
        rgb_3d (np.ndarray): A 3D RGB array of shape (Z, Y, X, 3).
        podosomes (list): A list of Podosome objects.
        draw_mode (str): Either "contour" to draw contours or "fill" to fill regions with transparent color.
        alpha (float): Transparency level for the fill mode (0 = fully transparent, 1 = fully opaque).

        Returns:
        np.ndarray: The 3D RGB array with podosomes drawn.
        """
        if draw_mode not in ["contour", "fill"]:
            raise ValueError("draw_mode must be either 'contour' or 'fill'.")

        centroid_color = (255, 0, 255)  # Purple (BGR format for OpenCV)
        # Convert the greyscale 3D array to RGB
        rgb_3d = np.stack([rgb_3d] * 3, axis=-1)  # Shape: (Z, Y, X, 3)
        blank_image = np.zeros_like(rgb_3d)

        for podosome in podosomes.values():
            for z, data in podosome.slices.items():
                if draw_mode == "contour":
                    # Draw the contour
                    contour = data['contour']
                    contour_dilated = podosome.dilated_slices[z]['contour']
                    if contour is not None:
                        cv2.drawContours(rgb_3d[z], [contour], -1, podosome.color, thickness=1)
                        cv2.drawContours(blank_image[z], [contour], -1, podosome.color, thickness=1)
                    if contour_dilated is not None and draw_dilation:
                        #cv2.drawContours(rgb_3d[z], [contour_dilated], -1, podosome.color, thickness=1)
                        cv2.drawContours(blank_image[z], [contour_dilated], -1, podosome.color, thickness=1)
                elif draw_mode == "fill":
                    # Fill the region with a transparent color
                    pixels = data['pixels']
                    if pixels:
                        # Create a mask for the podosome region
                        mask = np.zeros_like(rgb_3d[z][:, :, 0], dtype=np.uint8)
                        for y, x in pixels:
                            mask[y, x] = 1
                        # Blend the podosome color with the underlying image
                        overlay = rgb_3d[z].copy()
                        overlay[mask == 1] = (
                            alpha * np.array(podosome.color) + (1 - alpha) * overlay[mask == 1]
                        )
                        rgb_3d[z] = overlay

            if draw_centroid and podosome.centroid is not None:
                cx, cy, cz = map(int, podosome.centroid)
                if 0 <= cz < rgb_3d.shape[0]:  # Ensure it's within the Z bounds
                    cv2.circle(rgb_3d[cz], (cx, cy), radius=2, color=centroid_color, thickness=-1)
                    cv2.circle(blank_image[cz], (cx, cy), radius=2, color=centroid_color, thickness=-1)

        rgb_3d = np.stack([rgb_3d, blank_image], axis=1)

        return rgb_3d

    def draw_dots_on_mask(self, mask: np.ndarray, signals: dict, radius=5, solid=True):
        """
        Draws colored dots or circles on a 3D binary mask at specified signal locations with a given radius.
        
        Parameters:
            mask (np.ndarray): A 3D uint8 binary mask with shape (Z, H, W) where 0 is background, 255 is mask.
            signals (dict): Dictionary containing keys:
                            - 'Center': tuple (x, y) for the dot's coordinates.
                            - 'z': Integer for the z-layer index.
                            - 'PodosomeAssociated': Boolean indicating if the signal is associated with the mask.
            radius (int): Radius of the dot to be drawn.
            solid (bool): If True, draws filled circles; if False, draws hollow circles.
        
        Returns:
            np.ndarray: A 3D mask with colored dots or circles overlayed on the correct layers.
        """
        # Ensure the mask has 3 channels for color visualization
        color_mask = np.stack([mask] * 3, axis=-1)  # Convert grayscale mask to RGB
        blank_image = np.zeros_like(color_mask)

        for signal in signals:
            x, y = signal.center
            z = signal.z
            dot_color = (0, 255, 0) if signal.podosome_associated else (255, 0, 0)  # Green if associated, red otherwise
            thickness = -1 if solid else 2  # -1 for filled, 2 for hollow
            
            # Ensure the coordinates are within bounds
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                cv2.circle(color_mask[z], (x, y), radius, dot_color, thickness)
                cv2.circle(blank_image[z], (x, y), radius, dot_color, thickness)  # Draw solid or hollow circle
        
        rgb_3d = np.stack([color_mask, blank_image], axis=1)
        
        return rgb_3d

class PLAPlotter:
    def __init__(self, analysis_data):
        self._pla_data = analysis_data

    def show_experimantal_numbers(self):

        print(f"Donors: {self._pla_data.donors}")
        print(f"Control Donors: {self._pla_data.control_donors}")
        print(f"Treatment Donors: {self._pla_data.treatment_donors}")

        print(f"Views: {self._pla_data.views}")
        print(f"Control Views: {self._pla_data.control_views}")
        print(f"Treatment Views: {self._pla_data.treatment_views}")

        print(f"Cells: {self._pla_data.cells}")
        print(f"Control Cells: {self._pla_data.control_cells}")
        print(f"Treatment Cells: {self._pla_data.treatment_cells}")

        print(f"Signals Count: {self._pla_data.signals_count}")
        print(f"Control Signals: {self._pla_data.control_signals}")
        print(f"Treatment Signals: {self._pla_data.treatment_signals}")

        print(f"Podosome Associated Signals: {self._pla_data.podosome_associated_signals}")
        print(f"Control Podosome Associated Signals: {self._pla_data.control_podosome_associated_signals}")
        print(f"Treatment Podosome Associated Signals: {self._pla_data.treatment_podosome_associated_signals}")

        print(f"Non Podosome Associated Signals: {self._pla_data.non_podosome_associated_signals}")
        print(f"Control Non Podosome Associated Signals: {self._pla_data.control_non_podosome_associated_signals}")
        print(f"Treatment Non Podosome Associated Signals: {self._pla_data.treatment_non_podosome_associated_signals}")
    
    def show_overall_pla_signals(self, save_location):
        # Define the data
        groups = ["Isotype Control", "PLA Experiment"]
        total_signals = [self._pla_data.control_signals, self._pla_data.treatment_signals]
        podosome_associated = [self._pla_data.control_podosome_associated_signals, self._pla_data.treatment_podosome_associated_signals]
        non_podosome_associated = [total - podo for total, podo in zip(total_signals, podosome_associated)]

        # Sample sizes
        n_control_donors = self._pla_data.control_donors
        n_treatment_donors = self._pla_data.treatment_donors
        n_control_cells = self._pla_data.control_cells
        n_treatment_cells = self._pla_data.treatment_cells

        # Create DataFrame
        df = pd.DataFrame({
            "Group": groups,
            "Podosome-Associated": podosome_associated,
            "Non-Podosome-Associated": non_podosome_associated
        })

        # Set seaborn style
        sns.set_style("whitegrid")

        # Create figure and axes objects with desired figure size
        fig, ax = plt.subplots(figsize=(12, 10))  # Set figure size here

        #Create stacked bar chart
        df.set_index("Group").plot(
            kind="bar", stacked=True, 
            color=["#104862", "#196B2490"],  # Thesis colors with transparency
            edgecolor="none",  # Remove borders
            width=0.6, ax=ax
        )

        # Add labels inside the bars
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:  # Avoid empty labels
                    ax.annotate(
                        f"{int(height)}",  # Convert to int for clean numbers
                        (bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                        ha="center", va="center", fontsize=16, fontweight="bold", color="white"
                    )

        # Customizing the plot
        ax.set_title("PLA Signal Breakdown", fontsize=18, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Total Signal Count", fontsize=16, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

        # Disable the grid
        plt.grid(False)

        # Add custom legend with sample sizes
        legend_labels = [
            f"Podosome-Associated Signals",

            f"Non-Podosome-Associated Signals"

        ]

        ax.legend(
            labels=legend_labels,
            title_fontsize=12,  
            fontsize=14,        
            loc="upper left",
            frameon=True
        )

        fig.text(
            0.14, 0.77,  # Adjust y-position to be below the legend
            f"n PLA: {n_treatment_donors} donors, {n_treatment_cells} cells \nn Control: {n_control_donors} donors, {n_control_cells} cells",
            fontsize=10, color="grey", ha='left', va='top',
            bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=0.5')
        )

        sns.despine()

        fig.savefig(save_location, dpi=300, bbox_inches="tight")
        # Show the final plot
        #plt.show()

    def plot_podosome_associated_signals(self, save_location, type: str = "hex"):
        # Step 1: Filter signals where "PodosomeAssociated" is True
        podosome_associated_signals = [
            signal for signal in self._pla_data.signals 
            if signal.get("PodosomeAssociated") is True and signal.get("Control") is False
        ]

        # Step 2: Extract values
        dist_to_podo = np.array([signal["DistToPodo"] for signal in podosome_associated_signals])
        relative_signal_height = np.array([signal["RelativeSignalHeight"] for signal in podosome_associated_signals])

        resolution_factor = 17.0
        dist_to_podo = dist_to_podo / resolution_factor

        # Step 3: Create DataFrame
        df = pd.DataFrame({
            "DistToPodo": dist_to_podo,
            "RelativeSignalHeight": relative_signal_height
        })

        if type == "hexbin":
            params = {
                "kind": "hex",
                "cmap": "Blues",
                "height": 8
            }
        elif type == "kde": 
            params = {
                "kind": "kde",
                "fill": True,
                "cmap": "Blues",
                "height": 8
            }
        elif type == "kde_smooth":
            params = {
                "kind": "kde",
                "fill": True,
                "cmap": "Blues",
                "levels": 100,
                "height": 8
            }

        # Step 4: Create hexbin plot
        sns.set_theme(style="white")  # Clean style
        hexbin_plot = sns.jointplot(
            data=df, x="DistToPodo", y="RelativeSignalHeight",
            **params,
        )

        # Step 5: Add histograms with std deviation and median
        ax_marg_x = hexbin_plot.ax_marg_x
        ax_marg_y = hexbin_plot.ax_marg_y

        # Plot histograms
        sns.histplot(dist_to_podo, ax=ax_marg_x, color="#104862", kde=False, alpha=0.6)
        sns.histplot(y=relative_signal_height, ax=ax_marg_y, color="#104862", kde=False, alpha=0.6)

        # Compute standard deviations
        std_x = np.std(dist_to_podo)
        std_y = np.std(relative_signal_height)

        # Compute medians
        median_x = np.median(dist_to_podo)
        median_y = np.median(relative_signal_height)

        # Add std deviation lines to histograms
        ax_marg_x.axvline(np.mean(dist_to_podo) - std_x, color="black", linestyle="dashed", linewidth=1.2)
        ax_marg_x.axvline(np.mean(dist_to_podo) + std_x, color="black", linestyle="dashed", linewidth=1.2)
        ax_marg_y.axhline(np.mean(relative_signal_height) - std_y, color="black", linestyle="dashed", linewidth=1.2)
        ax_marg_y.axhline(np.mean(relative_signal_height) + std_y, color="black", linestyle="dashed", linewidth=1.2)

        # Add median lines to histograms
        ax_marg_x.axvline(median_x, color="red", linestyle="solid", linewidth=1.5)
        ax_marg_y.axhline(median_y, color="red", linestyle="solid", linewidth=1.5)

        # Step 6: Remove ticks from histograms but keep in hexbin plot
        ax_marg_x.tick_params(axis="both", left=False, right=False, labelleft=False, bottom=False, labelbottom=False)
        ax_marg_y.tick_params(axis="both", left=False, right=False, labelleft=False, bottom=False, labelbottom=False)

        # Step 7: Customization
        hexbin_plot.figure.suptitle("Spatial Relationship of PLA Signals and Podosomes", fontsize=16)
        hexbin_plot.figure.subplots_adjust(top=0.95)  # Adjust title position

        hexbin_plot.ax_joint.set_xlabel("Distance to Podosome [μm]", fontsize=12)
        hexbin_plot.ax_joint.set_ylabel("Relative Signal Height", fontsize=12)

        hexbin_plot.ax_joint.grid(False)  # Remove grid

        hexbin_plot.figure.text(
            0.8, 0.14, f"n = {len(dist_to_podo)}", fontsize=12, ha='right', va='top',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        # DistToPodo Histogram: Show std and median
        ax_marg_x.text(
            0.05, 0.8, f"Std: {std_x:.2f}\nMedian: {median_x:.2f}", fontsize=8, ha='left', va='top',
            transform=ax_marg_x.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        # RelativeSignalHeight Histogram: Show std and median
        ax_marg_y.text(
            0.5, 0.98 , f"Std: {std_y:.2f}\nMedian: {median_y:.2f}", fontsize=8, ha='center', va='bottom',
            transform=ax_marg_y.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )
        
        # Add horizontal dashed line at y = 1.0 (less pronounced)
        hexbin_plot.ax_joint.axhline(y=1.0, color="grey", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.7)

        # Add text label near the line (less pronounced)
        hexbin_plot.ax_joint.text(
            x=hexbin_plot.ax_joint.get_xlim()[1] * 0.8,  
            y=1.02,  
            s="Podosome Apex",
            fontsize=10,   # Smaller font
            color="dimgray",  # Lighter text
            fontweight="medium",  # Avoid bold
            alpha=0.7  # Reduce opacity
        )

        # Show plot
        plt.show()

        hexbin_plot.savefig(save_location, dpi=300, bbox_inches='tight')


class Visualizer:
    @staticmethod
    def plot_podosome_associated_signals(
        result,
        save_path: Union[str] = None,
        kind: Literal["hexbin", "kde", "kde_smooth"] = "hexbin",
        show: bool = True,
        save: bool = True,
    ):
        """Plot podosome-associated PLA signals from Signal objects."""

        # Step 1: Filter signals
        signals = result.get_signals(podosome_associated=True, is_control=False)

        if not signals:
            print("No podosome-associated signals found.")
            return

        # Step 2: Extract values
        resolution_factor = 17.0
        dist = np.array([s.distance_to_podosome for s in signals]) / resolution_factor
        height = np.array([s.relative_signal_height for s in signals])

        df = pd.DataFrame({
            "DistToPodo": dist,
            "RelativeSignalHeight": height
        })

        # Step 3: Plot settings
        params = {
            "hexbin": {"kind": "hex", "cmap": "Blues", "height": 8},
            "kde": {"kind": "kde", "fill": True, "cmap": "Blues", "height": 8},
            "kde_smooth": {"kind": "kde", "fill": True, "cmap": "Blues", "levels": 100, "height": 8},
        }.get(kind, {})

        sns.set_theme(style="white")
        plot = sns.jointplot(data=df, x="DistToPodo", y="RelativeSignalHeight", **params)

        ax_x, ax_y = plot.ax_marg_x, plot.ax_marg_y
        sns.histplot(dist, ax=ax_x, color="#104862", kde=False, alpha=0.6)
        sns.histplot(y=height, ax=ax_y, color="#104862", kde=False, alpha=0.6)

        # Step 4: Stats & lines
        std_x, std_y = np.std(dist), np.std(height)
        med_x, med_y = np.median(dist), np.median(height)

        ax_x.axvline(dist.mean() - std_x, color="black", linestyle="dashed", linewidth=1.2)
        ax_x.axvline(dist.mean() + std_x, color="black", linestyle="dashed", linewidth=1.2)
        ax_y.axhline(height.mean() - std_y, color="black", linestyle="dashed", linewidth=1.2)
        ax_y.axhline(height.mean() + std_y, color="black", linestyle="dashed", linewidth=1.2)

        ax_x.axvline(med_x, color="red", linewidth=1.5)
        ax_y.axhline(med_y, color="red", linewidth=1.5)

        ax_x.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_y.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plot.figure.suptitle("Spatial Relationship of PLA Signals and Podosomes", fontsize=16)
        plot.figure.subplots_adjust(top=0.95)
        plot.ax_joint.set_xlabel("Distance to Podosome [μm]")
        plot.ax_joint.set_ylabel("Relative Signal Height")
        plot.ax_joint.grid(False)

        # Step 5: Annotation
        plot.figure.text(
            0.8, 0.14, f"n = {len(dist)}", fontsize=12, ha='right', va='top',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        ax_x.text(
            0.05, 0.8, f"Std: {std_x:.2f}\nMedian: {med_x:.2f}", transform=ax_x.transAxes,
            fontsize=8, ha='left', va='top',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        ax_y.text(
            0.5, 0.98, f"Std: {std_y:.2f}\nMedian: {med_y:.2f}", transform=ax_y.transAxes,
            fontsize=8, ha='center', va='bottom',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        plot.ax_joint.axhline(y=1.0, color="grey", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.7)
        plot.ax_joint.text(
            x=plot.ax_joint.get_xlim()[1] * 0.8,
            y=1.02,
            s="Podosome Apex",
            fontsize=10, color="dimgray", fontweight="medium", alpha=0.7
        )

        # Step 6: Output
        if show:
            plt.show()

        if save and save_path:
            plot.savefig(str(save_path), dpi=300, bbox_inches="tight")

    @staticmethod
    def plot_profiles(profiles, save_path=None, protein="F-Actin", cmap='afmhot', podosome_count=0, radial_count=0):
        
        mirrored_profiles = np.hstack([np.fliplr(profiles), profiles])

        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(mirrored_profiles, cmap=cmap, aspect='auto')

        plt.colorbar(cax, label='Intensity')

        plt.title(f'Radial Intensity Profiles of {protein} Signals')
        plt.xlabel('Relative Distance to Podosome Core (x/y)')
        plt.ylabel('Relative Distance to Podosome Core (z)')

        current_ylim = ax.get_ylim()

        yticks = np.linspace(current_ylim[0], current_ylim[1], 5)
        yticks_transformed = np.linspace(-2.25, 2.25, 5)

        ax.set_yticks(yticks)
        ytick_labels = [f'{tick:.2f}' for tick in yticks_transformed]

        ytick_labels[0] = ''
        ytick_labels[-1] = ''

        ax.set_yticklabels(ytick_labels)

        current_xlim = ax.get_xlim()

        xticks = np.linspace(current_xlim[0], current_xlim[1], 5)
        xticks_transformed = np.linspace(-2.5, 2.5, 5)

        ax.set_xticks(xticks)
        xtick_labels = [f'{tick:.2f}' for tick in xticks_transformed]

        xtick_labels[0] = ''
        xtick_labels[-1] = ''

        ax.set_xticklabels(xtick_labels)

        plt.gca().invert_yaxis()

        # Add text box with podosome and radial counts
        textstr = f'n Podosomes: {podosome_count}\nRadial Samples: {radial_count}'
        props = dict(boxstyle='round,pad=0.2', facecolor='lightgrey', alpha=0.4, edgecolor='none')
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        plt.close()

if __name__ == "__main__":
    
    pkl_path = r"D:\Microscopy Testing\20250323 FULL RUN\20250404_012559\results.pkl"
    pkl_path = r"D:\Microscopy Testing\20250404 TINY RUN\20250404_122831\results.pkl"
    with open(pkl_path, "rb") as f:
        pla_data = pickle.load(f)

    plotter = PLAPlotter(pla_data)
    path = r"C:\Users\bryan\Desktop\output_image.png"
    path2 = r"C:\Users\bryan\Desktop\output_image2.png"
    path3 = r"C:\Users\bryan\Desktop\output_image3.png"
    path4 = r"C:\Users\bryan\Desktop\output_image4.png"

    plotter.show_experimantal_numbers()
    plotter.show_overall_pla_signals(path)
    plotter.plot_podosome_associated_signals(path2, type="hexbin")
    plotter.plot_podosome_associated_signals(path3, type="kde_smooth")
    plotter.plot_podosome_associated_signals(path4, type="kde")

    # print(len(plotter.analysis_data))
    # all_signals = plotter._flatten_to_list()
    # print(len(all_signals))


# analysis_data[file_idx][scene_idx][cell_idx] = {
# "podosome_associated_count": pla_cell.podosome_associated_count,
# "non_podosome_associated_count": pla_cell.non_podosome_associated_count,
# "total_count": len(pla_cell.signals),
# "signals": pla_cell.signals,
# "podosome_associated_signals": pla_cell.podosome_associated_signals,
# "control": self._experimental_control,
# }
# import numpy as np
# import tifffile

# # Create a test image: (Z=5, Channels=3, Y=512, X=512, RGB=3)
# z, c, y, x = 5, 3, 512, 512
# data = np.random.randint(0, 256, (z, c, y, x, 3), dtype=np.uint8)  # Random RGB data

# # Save as ImageJ-compatible TIFF
# tifffile.imwrite(r"C:\Users\bryan\Desktop\multichannel_rgb_stack.tif", data, imagej=True, metadata={'axes': 'ZCYXS'})