# MRI Preprocessing & Brain Age Prediction Pipeline

This project implements a comprehensive pipeline for brain age prediction from T1-weighted MRI images. The pipeline includes advanced preprocessing, deep learning model training, and evaluation, enabling robust prediction of brain age from raw MRI data.

## Brain Age Prediction Pipeline & Results

The pipeline consists of the following stages:

1. **Preprocessing**: Each T1-weighted MRI image undergoes denoising, bias field correction, intensity normalization, registration to the MNI152 template, and skull stripping. These steps standardize and clean the data for optimal model performance.
2. **Model Training**: A 3D ResNet-based deep learning model is trained on the preprocessed images to predict brain age categories. The model leverages volumetric features and spatial context for accurate classification.
3. **Evaluation**: The model's performance is evaluated on a held-out validation set. Training and validation accuracy/loss curves are generated, and the best model is selected based on validation accuracy.

**Results:**
- **Best Validation Accuracy:** 0.89
- **Best Validation Loss:** 0.15
- **Visualizations:**
  - Training/validation accuracy and loss curves: [`results/accuracy_loss_curves.png`]

For more details on the pipeline steps, see below.

## Features

The pipeline performs the following preprocessing steps:

1. **Image Denoising**: Reduces noise while preserving image structure using bilateral filtering, total variation denoising, or Gaussian smoothing
2. **Bias Field Correction**: Estimates and corrects intensity inhomogeneities using polynomial fitting
3. **Intensity Normalization**: Normalizes image intensities using z-score, min-max, or histogram equalization
4. **Image Registration**: Registers images to the MNI152 template for spatial standardization
5. **Skull Stripping**: Removes non-brain tissue using Otsu thresholding, template-based, or watershed methods

## Requirements

- Python 3.7+
- Required packages (see `requirements.txt`):
  - nibabel: For reading/writing NIfTI files
  - numpy: For numerical computations
  - scipy: For scientific computing
  - scikit-image: For image processing
  - matplotlib: For visualization
  - scikit-learn: For machine learning algorithms

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

```
MRI_project/
├── IXI-T1/                    # Input T1-weighted MRI files
├── templates/                  # MNI152 template files
│   ├── mni_icbm152_t1_tal_nlin_sym_09a.nii
│   └── mni_icbm152_t1_tal_nlin_sym_09a_mask.nii
├── preprocessed/              # Output directory (created automatically)
├── mri_preprocessing_pipeline.py
├── requirements.txt
└── README.md
```

## Usage

### Basic Usage

Run the complete pipeline on all T1-weighted images:

```bash
python mri_preprocessing_pipeline.py
```

### Programmatic Usage

```python
from mri_preprocessing_pipeline import MRIPreprocessingPipeline

# Initialize pipeline
pipeline = MRIPreprocessingPipeline(
    input_dir="IXI-T1",
    template_dir="templates", 
    output_dir="preprocessed"
)

# Process all images
pipeline.process_all_images()

# Or process a single image
results = pipeline.preprocess_single_image(
    "IXI-T1/IXI002-Guys-0828-T1.nii.gz",
    "IXI002_preprocessed"
)
```

## Output Files

For each input image, the pipeline generates:

### Processed Images
- `{prefix}_preprocessed.nii.gz`: Final preprocessed image (skull-stripped, registered, normalized)
- `{prefix}_brain_mask.nii.gz`: Binary brain mask
- `{prefix}_denoised.nii.gz`: Denoised image
- `{prefix}_bias_corrected.nii.gz`: Bias field corrected image
- `{prefix}_normalized.nii.gz`: Intensity normalized image
- `{prefix}_registered.nii.gz`: Template-registered image

### Visualizations
- `{prefix}_bias_field.png`: Bias field correction visualization
- `{prefix}_pipeline_overview.png`: Overview of all preprocessing steps

## Pipeline Steps in Detail

### 1. Image Denoising
- **Gaussian Smoothing**: Simple and effective noise reduction for 3D data
- **Total Variation**: Removes noise while preserving sharp edges
- **Median Filtering**: Good for salt-and-pepper noise
- **Non-local Means**: Advanced denoising preserving fine details
- **3D Bilateral**: Simplified 3D bilateral filtering

### 2. Bias Field Correction
- Estimates intensity inhomogeneities using polynomial fitting
- Corrects for scanner-related intensity variations
- Provides visualization of the estimated bias field

### 3. Intensity Normalization
- **Z-score**: Standardizes intensities to zero mean and unit variance
- **Min-max**: Scales intensities to [0, 1] range
- **Histogram Equalization**: Improves contrast distribution

### 4. Image Registration
- Registers images to MNI152 template for spatial standardization
- Enables group-level analysis and comparison
- Updates affine transformation matrices

### 5. Skull Stripping
- **Otsu Thresholding**: Automatic threshold-based brain extraction
- **Template-based**: Uses MNI152 brain mask (requires registration)
- **Watershed**: Advanced segmentation-based skull stripping

## Customization

You can customize the pipeline by modifying parameters:

```python
# Custom denoising
denoised = pipeline.denoise_image(img_data, method='tv')

# Custom bias field estimation
bias_field = pipeline.estimate_bias_field(img_data, degree=4)

# Custom intensity normalization
normalized = pipeline.normalize_intensity(img_data, method='histogram')

# Custom skull stripping
stripped, mask = pipeline.skull_stripping(img_data, method='watershed')
```

## Notes

- The pipeline automatically creates the output directory if it doesn't exist
- Processing time depends on image size and computational resources
- For large datasets, consider processing images in parallel
- The registration step is simplified; for production use, consider using specialized tools like ANTs or FSL
- All intermediate results are saved for quality control and debugging

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce image size or process images individually
2. **Template Not Found**: Ensure MNI152 template files are in the templates directory
3. **Import Errors**: Install all required dependencies from requirements.txt

### Quality Control

- Check bias field visualizations for reasonable correction
- Verify skull stripping results visually
- Compare registered images with template for proper alignment

## Citation

If you use this pipeline in your research, please cite the relevant papers for the methods used:

- Bilateral filtering: Tomasi & Manduchi (1998)
- Total variation denoising: Chambolle (2004)
- Otsu thresholding: Otsu (1979)
- Watershed segmentation: Beucher & Lantuéjoul (1979)

## License

This project is provided as-is for research and educational purposes. 