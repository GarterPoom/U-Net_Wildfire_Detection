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

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision
pip install rasterio geopandas
pip install scikit-image scikit-learn
pip install matplotlib seaborn pandas
pip install tqdm psutil

# GDAL (geospatial processing)
# On Windows: conda install gdal
# On Ubuntu: sudo apt-get install gdal-bin
# On macOS: brew install gdal
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/gistda-wildfire-unet.git
   cd gistda-wildfire-unet
   ```

2. **Prepare your data structure**
   ```
   project/
   ├── Classified_Image/          # Input Sentinel-2 JP2 files
   ├── Raster_Train/             # Training data (images + masks)
   ├── Wildfire_Polygon_Train/   # Training polygon labels
   └── CLMVTH_Administrative_Boundary/  # Country boundaries
   ```

3. **Run the complete workflow**
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
