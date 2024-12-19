# GenSVG

**GenSVG** is a project designed for generating SVG icons using machine learning techniques. It incorporates vectorization, rasterization, and training of models for efficient and high-quality icon generation. The repository contains tools and scripts for SVG manipulation, training and inference workflows, and utility functions to streamline the development of scalable vector graphics.

---

## Features

- **SVG Manipulation**: Tools to convert, vectorize, and rasterize SVG paths.
- **Training and Inference**: Scripts for training LoRA (Low-Rank Adaptation) models and performing inference for icon generation.
- **Dataset Preparation**: Utilities for preparing rasterized and vectorized icon datasets.
- **Neural Networks**: Implementation of CNNs and ViTs for better vectorization and training performance.
- **Submodule Integration**: Incorporates external libraries (`diffusers`, `diffvg`, `svgLearner`) for extended functionality.

---

## Repository Structure

```plaintext
icon-generator/
├── diffusers/             # Neural network tools for diffusion-based models.
├── diffvg/                # Tools for differentiable vector graphics.
├── inference/             # Scripts for training and inference with LoRA models.
├── outlined_rasterized/   # Experiments with ViTs and more data.
├── ui-material-icons/     # Material icon generation and model training.
├── dataset.py             # Prepares rasterized icons from datasets.
├── rasterize_path.py      # Rasterizes paths from SVG data.
├── rasterize_svg.py       # Converts SVG data into rasterized formats.
├── svg_to_paths.py        # Converts SVGs into path data for further processing.
├── svg_utils.py           # Utilities for handling SVG files.
├── remove_underscore.py   # Vectorization script improvements.
├── vectorizer.py          # Core vectorizer implementation.
├── requirements.txt       # Python dependencies for the project.
├── icon_lora_train.sh     # Shell script for training LoRA models.
├── train_text_to_image_lora.py # LoRA training for text-to-image applications.
├── lora_inference.py      # Performs inference using LoRA models.
├── .gitignore             # Ignore rules for Git tracking.
├── .gitmodules            # Configuration for Git submodules.
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

## Usage

### 1. Preparing the Dataset
Use `dataset.py` to generate rasterized icons:
```
python dataset.py --input_dir path/to/svg_files --output_dir path/to/output
```

### 2. Training
Train a LoRA model using the `icon_lora_train.sh` or `train_text_to_image_lora.py`:
```
bash icon_lora_train.sh
```

### 3. Inference
Run the `lora_inference.py` script to generate icons:
```
python lora_inference.py --model_path path/to/trained_model --input_text "example icon description"
```

---

## Contributing

Contributions are welcome! If you find a bug, have an idea for a new feature, or want to improve the documentation, feel free to open an issue or submit a pull request.

For any queries or support, please contact us at **support@gensvg.com**.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

We would like to acknowledge the contributions of:
- **Etai Sella**
- **Dr. Hadar Elor**

Their work has been instrumental in bringing **GenSVG** to life.

For any questions, please reach out via GitHub issues or email **team@gensvg.com**.

