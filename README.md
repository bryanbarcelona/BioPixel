<p align="center">
  <img src="images/images/BioPixelIcon.png" alt="BioPixel Logo" width="200"/>
</p>

<h1 align="center">BioPixel</h1>

<p align="center">
  🔬 Advanced cell image analysis pipeline powered by Cellpose and Python
</p>

---

## 📖 Overview

**BioPixel** is a growing Python-based pipeline designed for analyzing microscopy images. It currently supports:

1. **3D protein localization** relative to podosome architecture
   - Radial intensity profiling from podosome cores
   - Subdomain-specific targeting (core vs. cap vs. ring)

2. **Molecular interactions** via proximity ligation assay (PLA)
   - Single-molecule resolution spatial mapping
   - Statistical analysis of interaction hotspots

The tool is interactive and command-line driven, with support for key microscopy file types and a reproducible Conda-based environment.

---

### Why This Matters

Existing tools (ImageJ, Poji, proprietary software) require:
- Manual thresholding (subjective, slow)
- 2D analysis only
- Separate tools for localization vs. interaction analysis

**BioPixel provides:**
- ✅ Automated 3D quantification
- ✅ Unified platform (localization + PLA in one workflow)
- ✅ Statistical rigor across experimental conditions
- ✅ Reproducible analysis pipeline

---

## 🧬 Biological Discoveries Enabled

Using BioPixel on macrophage podosomes revealed:

**1. Drebrin localization architecture**
- Precise targeting to podosome caps (actin regulatory subdomain)
- Coiled-coil, helical, and C-terminal regions required
- ADFH and proline-rich domains dispensable

**2. Novel cytoskeletal crosstalk mechanism**
- Drebrin-EB3-clathrin complexes preferentially form at podosome peripheries
- Links actin remodeling (drebrin) to microtubule transport (+TIP protein EB3)
- Suggests podosomes coordinate actin and microtubule networks

**3. Functional implication**
- Podosomes may serve as integrated hubs for cytoskeletal coordination
- Spatial organization of molecular complexes guides matrix interactions

---

## 🚀 Features

- 🧠 **Cellpose-based segmentation** - robust cell and podosome detection
- 📊 **3D radial profiling** - quantify protein enrichment by distance from podosome core
- 🎯 **PLA spatial analysis** - map where protein interactions occur
- 📈 **Statistical comparison** - compare localization across conditions (control vs. treatment)
- 🔬 **Multi-format support** - reads Leica `.lif`, Zeiss `.czi`, Nikon `.nd`, Olympus `.oib`, and `.tif`
- ✅ Simple interactive CLI (no need to pre-learn command-line arguments)
- 🎯 Uses pre-trained or custom Cellpose models for segmentation
- 📊 Generates publication-ready plots (PNG outputs)
  - For **profiling**: radial intensity plots with error bars
  - For **spatial analysis**: color-coded spatial distribution visualizations
- 🧱 Modular codebase for future expansion

---

## 🛠️ Technical Implementation

### Challenge: Proprietary Microscopy Formats
Each manufacturer uses incompatible file formats:
- **Leica LIF**: Multi-series files with embedded metadata
- **Zeiss CZI**: XML-based with phase separation (confocal vs. AiryScan)
- **Nikon ND**: Manifest + hundreds of individual `.stk` files
- **Olympus OIB**: Archived multi-file format

**Solution:** Custom parsers with unified 5D array output (TZCYX), streaming architecture for memory efficiency (handles 50GB+ datasets).

### Analysis Pipeline
1. Load multi-dimensional microscopy data (any format)
2. Segment cells (Cellpose) and podosomes (custom detection)
3. Generate 3D radial intensity profiles per podosome
4. Map PLA signal spatial distribution
5. Statistical comparison across experimental groups

---

## 🧰 Prerequisites

> 💡 **GPU Support**
>
> If you want to use your **NVIDIA GPU** for faster image processing (especially with Cellpose), you must:
>
> - Have a compatible **NVIDIA GPU**
> - Install the correct **CUDA toolkit and cuDNN** libraries for your system
> - Install **PyTorch with GPU support**
>
> You can find the correct instructions and compatibility info here:
> 👉 [PyTorch Get Started (with CUDA)](https://pytorch.org/get-started/locally/)
>
> ⚠️ BioPixel does **not** automatically install GPU dependencies. If GPU isn't set up, it will fall back to CPU.


This tool is built for local use on Windows systems and assumes minimal programming experience. To use BioPixel, you’ll need:

- 🐍 **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** installed (preferred over Anaconda for lightweight environments)
- 🧬 **[Git for Windows](https://git-scm.com/download/win)**  Required to download (clone) the project from GitHub. After installing, you'll be able to use the `git` command in your terminal.
- 📂 Basic familiarity with your file system (copying/pasting paths)
- 🖼️ Microscopy images in one of the following formats:
  - `.lif`, `.czi`, `.tif`, `.nd`, `.oib` *(only `.lif` has been fully tested)*
- 💡 Optional: GPU for faster Cellpose segmentation

---

## 📦 Setup Instructions

> 💡 **Before you begin:**  
> Open the **Anaconda Prompt** (or your preferred terminal) and navigate to the folder where you want to store the BioPixel code. For example:
>
> ```bash
> cd D:\MyProjects
> ```

### 1. Clone the repository

```bash
git clone https://github.com/bryanbarcelona/BioPixel.git
cd BioPixel
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate biopixel-env
```

### 4. Run the tool

```bash
python biopixel.py
```

---

## 📂 Output

- `.png` files for plots and visualizations
- Organized result folders per input image
- Intermediate segmentation masks if enabled

| <img src="images\readme\pla_analysis.png" alt="Image 1" height="200"> | <img src="images\readme\profile.png" alt="Image 2" height="200"> |
|:-------------------------------------------------:|:-------------------------------------------------:|
| **PLA signal spatial distribution analysis.** Kernel density estimation (KDE) plot mapping the spatial relationship between protein-protein interaction signals and cellular structures. X-axis shows distance in µm, y-axis shows relative height positioning. Marginal distributions displayed along axes. Statistical metrics and sample sizes indicate analysis scale. Example output styled for presentation; spatial analysis and statistical computation performed in BioPixel.  | **Automated 3D radial intensity profiling.** Normalized intensity profiles showing protein localization patterns relative to podosome architecture across x, y, and z dimensions. Each profile aggregates data from multiple podosomes with statistical overlays. Example output styled for presentation; analysis pipeline and quantification performed in BioPixel.|

---

## 📁 File Format Support

BioPixel supports loading the following formats:
- `.lif` (Leica) ✅ *tested*
- `.czi` (Zeiss)
- `.tif` (TIFF stacks)
- `.nd` (Nikon)
- `.oib` (Olympus)

---

## 🧪 Test Dataset

*Coming soon* – sample data + results to demonstrate core functionality.

---

## 🛠️ Development Notes

- 🔒 Local-only operation (no cloud dependencies)
- ✅ Safe to use alongside a traditional Python `venv`
- ✅ Conda environment fully isolated from your base Python setup
- 🚀 Easily portable to other machines using `environment.yml`
- ⚠️ Still under active development — feedback and contributions welcome!

BioPixel was developed during my PhD research with the goal of solving specific limitations in existing podosome analysis tools. It represents a functional, production-used analysis platform rather than enterprise software. 

**Known areas for future improvement:**
- Refactor format readers for better separation of concerns
- Comprehensive test coverage
- GUI for non-technical users

The priority was **enabling biological discovery** over software architecture perfection - and it succeeded in generating novel insights into podosome function.

---

## 📌 Future Directions

- GUI front-end for less technical users
- Support for additional file formats and batch processing
- Integration with automated annotation tools

---

## 📚 Citation

If you use BioPixel in your research, please cite:

**Barcelona, B.** (2025). *BioPixel Reveals How Podosomes and Microtubule +TIPs Regulate Intracellular Transport.* Doctoral dissertation, University of Hamburg, Faculty of Mathematics, Computer Science and Natural Sciences, Department of Biology. Hamburg, Germany.

---

## 📬 Feedback & Contributions

If you encounter issues, feel free to [open an issue](https://github.com/bryanbarcelona/BioPixel/issues) or submit a pull request. This project welcomes contributions from researchers, developers, and tinkerers alike.

---

## 🤖 AI Usage Declaration

Well, well, well, look who’s reading the fine print! You, my astute friend, have just stumbled upon the one part of this README I didn’t want you to see — or did I?
Now, I’ll level with ya — this README didn’t write itself. Oh no, sir. We had a little… help from one of those newfangled word-slingin’ robot contraptions. You know the type: all circuits and no coffee breaks. But listen — every syllable was put through the wringer by yours truly. That’s right, I personally wrestled this text into shape.

---

© 2025 Bryan Barcelona. All rights reserved.
