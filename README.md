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

- 🧫 **Podosome profiling** — generates radial intensity plots with statistical overlays for each podosome-containing cell.
- 🧬 **PLA & podosome spatial analysis** — computes and visualizes spatial relationships between PLA signals and podosomes using Cellpose-based segmentation.

The tool is interactive and command-line driven, with support for key microscopy file types and a reproducible Conda-based environment.

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

## 🚀 Features

- ✅ Simple interactive CLI (no need to pre-learn command-line arguments)
- 🎯 Uses pre-trained or custom Cellpose models for segmentation
- 📊 Generates publication-ready plots (PNG outputs)
  - For **profiling**: radial intensity plots with error bars
  - For **spatial analysis**: color-coded spatial distribution visualizations
- 🧱 Modular codebase for future expansion

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

---

## 📌 Future Directions

- GUI front-end for less technical users
- Support for additional file formats and batch processing
- Integration with automated annotation tools

---

## 📬 Feedback & Contributions

If you encounter issues, feel free to [open an issue](https://github.com/bryanbarcelona/BioPixel/issues) or submit a pull request. This project welcomes contributions from researchers, developers, and tinkerers alike.

---

© 2025 Bryan Barcelona. All rights reserved.
