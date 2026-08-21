# GISTDA Wildfire U-Net Segmentation

A comprehensive deep learning pipeline for automated wildfire detection and burn area mapping using U-Net architecture and Sentinel-2 satellite imagery. This project was developed for the Geo-Informatics and Space Technology Development Agency (GISTDA) to support wildfire monitoring and management across Southeast Asia.

## 🔥 Overview

This repository provides a complete end-to-end solution for wildfire detection that processes Sentinel-2 satellite imagery through multiple stages: image preprocessing, cloud masking, deep learning-based burn area classification, and geospatial analysis. The system is specifically designed for the CLMVTH region (Cambodia, Laos, Myanmar, Vietnam, and Thailand).

## ✨ Key Features

- **Automated Sentinel-2 Processing**: Resamples and stacks spectral bands with vegetation indices
- **Cloud Masking**: Advanced cloud detection and masking using Scene Classification Layer (SCL)
- **U-Net Deep Learning**: State-of-the-art semantic segmentation for burn area detection
- **Multi-Country Support**: Administrative boundary intersection for CLMVTH countries
- **Geospatial Output**: Generates both raster masks and polygon shapefiles
- **Memory Efficient**: Chunked processing for large satellite images
- **Comprehensive Evaluation**: Detailed model performance metrics and visualizations

## 🏗️ Architecture

The system consists of five main modules:

1. **Image Preprocessing** (`classified_image_processing.py`)
   - Resamples Sentinel-2 bands to 10m resolution
   - Computes vegetation indices (NDVI, NDWI, SAVI, BAIS2)
   - Creates compressed, tiled GeoTIFF outputs

2. **Cloud Masking** (`classified_cloud_mask.py`)
   - Uses SCL data to identify and mask clouds
   - Memory-efficient chunked processing
   - Preserves geospatial metadata

3. **Model Training** (`unet_wildfire_training.py`, `unet_wildfire_no_shape_training.py`)
   - U-Net architecture with skip connections
   - Balanced dataset handling
   - Class-weighted loss function
   - Comprehensive evaluation metrics

4. **Prediction** (`unet_wildfire_predict.py`)
   - Tiled inference for large images
   - Overlapping window processing
   - Probability and binary mask outputs

5. **Polygon Generation** (`unet_polygon.py`)
   - Converts raster masks to vector polygons
   - Administrative boundary intersection
   - Multi-country attribute assignment

## 🧰 Before You Begin: System Requirements

Set these up **before** you clone the repo or run any script — several of the Python packages here (GDAL, rasterio, geopandas, PyTorch) depend on system-level libraries or drivers that pip alone won't fully resolve.

| Software | Why it's needed | Get it |
|---|---|---|
| **Git** | To clone this repository | [git-scm.com/downloads](https://git-scm.com/downloads) |
| **Python 3.10+** (project was built/tested on 3.13) | Runs every script in the pipeline | [python.org](https://www.python.org/downloads/) or via Conda below |
| **Miniconda / Anaconda** *(strongly recommended)* | Makes installing GDAL and its compiled dependencies far more reliable than pip alone | [docs.conda.io/miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| **GDAL (system library)** | `rasterio`, `geopandas`, and direct `from osgeo import gdal` calls in `classified_preprocessing.py` all need the compiled GDAL library, not just a Python wrapper | Easiest via `conda install -c conda-forge gdal`; otherwise `apt-get install gdal-bin libgdal-dev` (Ubuntu) or `brew install gdal` (macOS) |
| **NVIDIA GPU driver + CUDA** *(optional, recommended for training)* | Training/inference use PyTorch with CUDA (`torch==2.11.0+cu128` in the provided environment) for a major speed-up | Install the latest [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) supporting CUDA 12.8. No GPU? The code auto-falls-back to CPU, just slower. |
| **C++ Build Tools** *(Windows only, if pip-installing rather than using conda)* | Some geospatial packages compile native extensions on install | "Desktop development with C++" workload from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| **QGIS** *(optional)* | Not required to run the pipeline, but the scripts print QGIS styling hints and it's the easiest way to inspect output GeoTIFF masks/probability maps | [qgis.org/download](https://qgis.org/download/) |

**Quick check before cloning:**
```bash
git --version
python --version      # 3.10+
conda --version        # if using the conda route
nvidia-smi              # only relevant if you plan to use GPU acceleration
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GarterPoom/U-Net_Wildfire_Detection.git
   cd U-Net_Wildfire_Detection
   ```

2. **Create the environment**

   **Option A — Conda (recommended, especially on Windows/NVIDIA):** the repo ships a ready-made environment file.
   ```bash
   conda env create -f unet_fire_environment.yml
   conda activate unet_fire
   ```
   > Note: `unet_fire_environment.yml` was exported from a Windows + CUDA 12.8 setup. On macOS/Linux or CPU-only machines, use Option B instead.

   **Option B — Manual (pip / venv), e.g. macOS, Linux, or CPU-only:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate

   # Install system GDAL first (see table above), then:
   pip install torch torchvision   # add --index-url https://download.pytorch.org/whl/cu128 for a matching GPU build
   pip install rasterio geopandas shapely
   pip install scikit-image scikit-learn
   pip install matplotlib seaborn pandas
   pip install tqdm psutil GDAL=="$(gdal-config --version)"
   ```

3. **Prepare your data structure**
   ```
   project/
   ├── Classified_Image/          # Input Sentinel-2 JP2 files
   ├── Raster_Train/             # Training data (images + masks)
   ├── Wildfire_Polygon_Train/   # Training polygon labels
   └── CLMVTH_Administrative_Boundary/  # Country boundaries
   ```

4. **Run the complete workflow**
   ```bash
   python detection_module.py
   ```

### Individual Module Usage

**1. Preprocess Sentinel-2 imagery:**
```bash
python classified_image_processing.py
```

**2. Apply cloud masking:**
```bash
python classified_cloud_mask.py
```

**3. Train the U-Net model:**
```bash
python unet_wildfire_training.py
```

**4. Generate predictions:**
```bash
python unet_wildfire_predict.py --image_path Raster_Classified --output_dir Predicted_Mask
```

**5. Create polygon outputs:**
```bash
python unet_polygon.py
```

## 📊 Model Performance

The U-Net model achieves high accuracy in wildfire detection:

- **Pixel-wise Accuracy**: >90% on validation data
- **Tile-wise Accuracy**: >85% for burn area classification
- **Precision/Recall**: Balanced performance across burn and unburn classes
- **F1-Score**: Optimized for both pixel and tile-level evaluation

## 🗂️ Data Requirements

### Input Data
- **Sentinel-2 L2A Products**: JP2 format with all spectral bands
- **Scene Classification Layer (SCL)**: For cloud masking
- **Training Data**: GeoTIFF images with corresponding burn area masks
- **Administrative Boundaries**: Shapefiles for CLMVTH countries

### Output Data
- **Raster Masks**: Binary and probability maps of burn areas
- **Polygon Shapefiles**: Vectorized burn areas with administrative attributes
- **Model Artifacts**: Trained model weights and evaluation metrics

## 🔧 Configuration

### Key Parameters

**Image Processing:**
- Target resolution: 10m
- Tile size: 256×256 pixels
- Overlap: 32 pixels (for prediction)

**Model Training:**
- Epochs: 50-100
- Batch size: 4
- Learning rate: 1e-4
- Loss function: BCEWithLogitsLoss with class weighting

**Prediction:**
- Threshold: 0.5 (binary classification)
- Tile overlap: 32 pixels
- Memory optimization: Chunked processing

## 📁 Project Structure

```
gistda-wildfire-unet/
├── README.md                           # This file
├── detection_module.py                 # Main workflow orchestrator
├── classified_image_processing.py     # Sentinel-2 preprocessing
├── classified_cloud_mask.py           # Cloud masking module
├── unet_wildfire_training.py          # Model training with shapefiles
├── unet_wildfire_no_shape_training.py # Model training without shapefiles
├── unet_wildfire_predict.py           # Inference and prediction
├── unet_polygon.py                    # Polygon generation
├── Classified_Image/                  # Input Sentinel-2 data
├── Raster_Classified/                 # Processed imagery
├── SCL_Classified/                    # Scene classification data
├── Raster_Classified_Cloud_Mask/      # Cloud-masked imagery
├── Predicted_Mask/                    # Binary prediction masks
├── Predicted_Probability/             # Probability maps
├── unet_polygon/                      # Output polygon shapefiles
├── Export_Model/                      # Trained model weights
└── Model_Evaluation/                  # Performance metrics and plots
```

## 🌍 Geographic Coverage

This system is optimized for the CLMVTH region:
- **Cambodia** (KHM)
- **Laos** (LAO) 
- **Myanmar** (MMR)
- **Vietnam** (VNM)
- **Thailand** (THA)

Administrative boundary intersection provides detailed location information including province/district names in both local languages and English.

## 🔬 Technical Details

### U-Net Architecture
- **Encoder**: 4 downsampling blocks (64→128→256→512→1024 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Activation**: ReLU with Batch Normalization
- **Output**: Single channel binary segmentation

### Spectral Indices
- **NDVI**: Normalized Difference Vegetation Index
- **NDWI**: Normalized Difference Water Index  
- **SAVI**: Soil Adjusted Vegetation Index
- **BAIS2**: Burned Area Index for Sentinel-2

### Data Processing
- **Input**: Multi-band Sentinel-2 imagery (10+ bands)
- **Preprocessing**: Min-max normalization per band
- **Augmentation**: Tiled processing with overlap
- **Output**: GeoTIFF with preserved geospatial metadata

## 📈 Evaluation Metrics

The system provides comprehensive evaluation including:
- Classification reports (precision, recall, F1-score)
- Confusion matrices (pixel and tile-level)
- Training/validation loss and accuracy curves
- Model performance visualization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **GISTDA** (Geo-Informatics and Space Technology Development Agency)
- **Sentinel-2** data provided by ESA
- **PyTorch** deep learning framework
- **Rasterio** and **GDAL** for geospatial processing
- **Open source community** for various supporting libraries

## 📞 Contact

For questions, issues, or collaboration opportunities:
- **Email**: [siripoom.su@gmail.com]
- **Organization**: GISTDA

---

**Note**: This system is designed for research and operational wildfire monitoring. Always validate results with ground truth data and consider local environmental conditions when interpreting burn area maps.
