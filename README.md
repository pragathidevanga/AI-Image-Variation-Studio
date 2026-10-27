# AI Image Variation & Reconstruction Studio

## Project Overview

The **AI Image Variation & Reconstruction Studio** is an interactive web application built with Python and Streamlit that demonstrates how a pretrained image-to-image AI model transforms an input image into a new visual variation.

Users can upload an image, enter a transformation prompt, adjust generation settings, compare the original and transformed images, and analyse pixel-level differences.

## Features

- Upload images in PNG, JPG, and JPEG formats.
- Generate transformed images using a pretrained Hugging Face Diffusers model.
- Customise the prompt and negative prompt.
- Adjust transformation strength, prompt guidance scale, inference steps, and random seed.
- View original and AI-transformed images side by side.
- Calculate Mean Squared Error (MSE).
- Visualise pixel-level differences using a difference map.
- Download the transformed image.

## Technologies Used

- Python
- Streamlit
- Hugging Face Diffusers
- Stable Diffusion image-to-image pipeline
- PyTorch
- Pillow
- NumPy
- Matplotlib

## Project Structure

```text
AI-Image-Variation-Studio/
├── app.py
├── image_backend.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
```

## Installation and Setup

### 1. Clone or download the project

Open the project folder in Visual Studio Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

The first run may take additional time because the pretrained model needs to download. Performance depends on the computer's hardware and available memory.

## How to Use

1. Launch the Streamlit application.
2. Upload an image.
3. Enter a prompt describing the desired transformation.
4. Adjust the transformation settings.
5. Click **Transform image**.
6. Compare the original and transformed images.
7. Review the MSE value and pixel difference map.
8. Download the generated image.

## Evaluation Metric: Mean Squared Error

Mean Squared Error measures the average squared difference between corresponding pixel values in the original and transformed images.

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(I_i-O_i)^2
\]

Here, \(I_i\) represents an original pixel value, \(O_i\) represents a transformed pixel value, and \(N\) is the number of pixel values compared.

A lower MSE indicates smaller pixel-level differences. However, MSE does not determine whether the generated image is more attractive, realistic, or semantically better.

## Model and Processing

The application uses a pretrained image-to-image diffusion pipeline from Hugging Face Diffusers. The input image and user prompt guide the transformation. Generation settings influence how strongly the output differs from the original.

The backend module handles image preprocessing, model loading, image generation, MSE calculation, and difference-map generation. The Streamlit frontend provides the interactive user interface.

## Limitations

- Image generation may be slow on CPU-only systems.
- Initial model download requires an internet connection.
- Generated images may contain visual artefacts or alter details in the input.
- MSE measures pixel differences rather than semantic image quality.

## Conclusion

This project demonstrates an interactive workflow for image-to-image AI transformation, parameter control, output comparison, and pixel-level difference analysis using Python and open-source machine learning tools.

## Author

**Pragathi BR**
