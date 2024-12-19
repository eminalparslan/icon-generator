# GenSVG

**GenSVG** is a project designed for generating SVG icons from text. It incorporates diffusion models to generate raster images that are then vectorized for efficient, diverse, and high-quality icon generation. The repository contains tools and scripts for SVG manipulation, training and inference workflows, and utility functions to streamline the development of scalable vector graphics.

---

## Features

- **Training and Inference**: Scripts for training LoRA (Low-Rank Adaptation) models and performing inference for icon generation.
- **Neural Networks**: Implementation of CNNs and ViTs for better vectorization and training performance.
- **Dataset Preparation**: Utilities for preparing rasterized and vectorized icon datasets.
- **SVG Manipulation**: Tools to convert, vectorize, and rasterize SVG paths.

---

## Repository Structure

```plaintext
icon-generator/
├── diffusers/             # Neural network tools for diffusion-based models.
├── diffvg/                # Tools for differentiable vector graphics.
├── inference/             # Inference outputs.
├── outlined_rasterized/   # Rasterized icon dataset.
├── ui-material-icons/     # HuggingFace LoRA model.
├── dataset.py             # Dataset testing script.
├── rasterize_path.py      # Rasterizes paths from SVG data.
├── rasterize_svg.py       # Converts SVG data into rasterized formats.
├── svg_to_paths.py        # Converts SVGs into path data for further processing from `svgpathtools`.
├── svg_utils.py           # Utilities for handling SVGs and path components.
├── remove_underscore.py   # Preprocessing script for dataset `metadata.csv`.
├── vectorizer.py          # Core vectorizer implementation.
├── requirements.txt       # Python dependencies for the project.
├── icon_lora_train.sh     # Shell script to train LoRA the model.
├── train_text_to_image_lora.py # LoRA training script from `diffusers`.
├── lora_inference.py      # Performs inference using our LoRA models.
```

---

## Installation

1. Clone the repository and its submodules:
   ```
   git clone --recurse-submodules https://github.com/eminalparslan/icon-generator.git
   cd icon-generator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary tools like Python 3.8+ and a compatible GPU (optional but recommended for training).

---

## Acknowledgments

We would like to acknowledge the contributions of:
- **Etai Sella**
- **Dr. Hadar Elor**

