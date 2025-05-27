import numpy as np
import pickle
from typing import Literal, Union
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_podosome_associated_signals(
        result,
        save_path: Union[str] = None,
        kind: Literal["hexbin", "kde", "kde_smooth"] = "hexbin",
        show: bool = True,
        save: bool = True,
    ):

        signals = result.get_signals(podosome_associated=True, is_control=False)

        if not signals:
            print("No podosome-associated signals found.")
            return

        resolution_factor = 17.0
        dist = np.array([s.distance_to_podosome for s in signals]) / resolution_factor
        height = np.array([s.relative_signal_height for s in signals])

        df = pd.DataFrame({
            "DistToPodo": dist,
            "RelativeSignalHeight": height
        })

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

        plot.figure.text(
            0.8, 0.14, f"n = {len(dist)}", fontsize=12, ha='right', va='top',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        ax_x.text(
            0.05, 0.8, f"Std: {std_x:.2f}\nMedian: {med_x:.2f}", transform=ax_x.transAxes,
            fontsize=11, ha='left', va='top',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        ax_y.text(
            0.6, 0.98, f"Std: {std_y:.2f}\nMedian: {med_y:.2f}", transform=ax_y.transAxes,
            fontsize=11, ha='center', va='bottom',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')
        )

        plot.ax_joint.axhline(y=1.0, color="dimgray", linestyle=(0, (3, 3)), linewidth=0.8, alpha=1.0)
        plot.ax_joint.text(
            x=plot.ax_joint.get_xlim()[1] * 0.8,
            y=1.02,
            s="Podosome Apex",
            fontsize=12, color="dimgray", fontweight="medium", alpha=1.0
        )

        if show:
            plt.show()

        if save and save_path:
            plot.savefig(str(save_path), dpi=300, bbox_inches="tight")

    @staticmethod
    def plot_profiles(profiles, save_path=None, protein="F-Actin", cmap='afmhot', podosome_count=0,
                      radial_count=0, tick_fontsize=16, colorbar_fontsize=16, statbox_fontsize=16):
            
            mirrored_profiles = np.hstack([np.fliplr(profiles), profiles])

            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(mirrored_profiles, cmap=cmap, aspect='auto')

            cbar = plt.colorbar(cax, label='')
            cbar.ax.tick_params(labelsize=colorbar_fontsize)

            plt.title(f'Radial Intensity Profiles of {protein} Signals')
            plt.xlabel('Relative Distance to Podosome Core (x/y)')
            plt.ylabel('Relative Distance to Podosome Core (z)')

            current_ylim = ax.get_ylim()

            yticks = np.linspace(current_ylim[0], current_ylim[1], 5)
            yticks_transformed = np.linspace(2.25, -2.25, 5)

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

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

            textstr = f'n Podosomes: {podosome_count}\nRadial Samples: {radial_count}'
            props = dict(boxstyle='round,pad=0.2', facecolor='lightgrey', alpha=0.6, edgecolor='none')
            ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=statbox_fontsize,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

            if save_path:
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
            plt.close()

if __name__ == "__main__":

    pass