import os
import logging
import typing
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from osgeo import gdal
import gc

class SentinelCloudMasker:
    """
    A memory-efficient class to handle cloud masking for Sentinel-2 imagery.
    """
    CLOUD_CLASSES = {
        3: "Cloud shadows",
        6: "Water",
        8: "Cloud medium probability",
        9: "Cloud high probability",
    }

    def __init__(self, scl_dir: str, band_dir: str, output_dir: str, 
                 chunk_size: int = 1024, log_level: int = logging.INFO):
        """
        Initialize the cloud masking processor.
        
        Args:
            scl_dir (str): Directory containing SCL files
            band_dir (str): Directory containing band files
            output_dir (str): Directory for output files
            chunk_size (int): Size of chunks to process at once
            log_level (int): Logging level
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size

        # Validate directories
        self._validate_directories(scl_dir, band_dir, output_dir)
        
        self.scl_dir = scl_dir
        self.band_dir = band_dir
        self.output_dir = output_dir

        # Prepare file mappings
        self.scl_files = self._get_file_mapping(scl_dir, '_SCL.tif')
        self.band_files = self._get_file_mapping(band_dir, '.tif')

    def _validate_directories(self, *dirs: str) -> None:
        """Validate directory existence and permissions."""
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory does not exist: {dir_path}")
            if not os.path.isdir(dir_path):
                raise ValueError(f"Not a directory: {dir_path}")
            if not os.access(dir_path, os.R_OK):
                raise ValueError(f"Directory not readable: {dir_path}")

    def _get_file_mapping(self, directory: str, suffix: str) -> dict:
        """Build file mapping dictionary."""
        return {
            os.path.basename(f).split(suffix)[0]: os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files if f.endswith(suffix)
        }

    def create_cloud_mask(self, scl_data: np.ndarray) -> np.ndarray:
        """Create cloud mask from SCL data."""
        return ~np.isin(scl_data, list(self.CLOUD_CLASSES.keys()))

    def _get_chunk_windows(self, width: int, height: int) -> typing.Iterator[Window]:
        """
        Generate chunk windows for processing.
        
        Args:
            width (int): Image width
            height (int): Image height
            
        Yields:
            Window: Rasterio window object for each chunk
        """
        for y in range(0, height, self.chunk_size):
            for x in range(0, width, self.chunk_size):
                window = Window(
                    x,
                    y,
                    min(self.chunk_size, width - x),
                    min(self.chunk_size, height - y)
                )
                yield window

    def process_chunk(self, scl_data: np.ndarray, band_data: np.ndarray) -> np.ndarray:
        """
        Process a single chunk of data.
        
        Args:
            scl_data (np.ndarray): SCL data chunk
            band_data (np.ndarray): Band data chunk
            
        Returns:
            np.ndarray: Processed chunk
        """
        cloud_mask = self.create_cloud_mask(scl_data)
        masked_chunk = np.where(cloud_mask, band_data, np.nan)
        return masked_chunk.astype(np.float32)

    def process_files(self) -> None:
        """Process all files in chunks to minimize memory usage."""
        os.makedirs(self.output_dir, exist_ok=True)
        processed_count = 0

        for identifier, scl_path in self.scl_files.items():
            if identifier not in self.band_files:
                self.logger.warning(f"No matching band file for {identifier}")
                continue

            try:
                self.logger.info(f"Processing: {identifier}")
                band_path = self.band_files[identifier]

                # Open both files to get metadata
                with rasterio.open(scl_path) as scl_ds, \
                     rasterio.open(band_path) as band_ds:
                    
                    # Prepare output metadata
                    out_meta = band_ds.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "dtype": "float32",
                        "nodata": np.nan
                    })

                    # Create temporary output file
                    temp_output_path = os.path.join(
                        self.output_dir, 
                        f"{identifier}_masked_uncompressed.tif"
                    )
                    
                    with rasterio.open(temp_output_path, "w", **out_meta) as dest:
                        # Process by chunks
                        for window in self._get_chunk_windows(band_ds.width, band_ds.height):
                            # Read chunks
                            scl_chunk = scl_ds.read(1, window=window)
                            band_chunks = band_ds.read(window=window)

                            # Process each band in the chunk
                            for band_idx in range(band_chunks.shape[0]):
                                processed_chunk = self.process_chunk(
                                    scl_chunk, 
                                    band_chunks[band_idx]
                                )
                                dest.write(processed_chunk, window=window, indexes=band_idx + 1)

                            # Force garbage collection after each chunk
                            del scl_chunk, band_chunks
                            gc.collect()

                # Compress output using GDAL
                output_path = os.path.join(self.output_dir, f"{identifier}_masked.tif")
                gdal.Translate(
                    destName=output_path,
                    srcDS=temp_output_path,
                    options=gdal.TranslateOptions(
                        format="GTiff",
                        creationOptions=[
                            "COMPRESS=LZW",
                            "PREDICTOR=3",
                            "TILED=YES",
                            "BLOCKXSIZE=256",
                            "BLOCKYSIZE=256",
                            "BIGTIFF=YES"
                        ]
                    )
                )

                # Clean up
                os.remove(temp_output_path)
                processed_count += 1
                self.logger.info(f"Completed processing: {identifier}")

            except Exception as e:
                self.logger.error(f"Error processing {identifier}: {e}")
            finally:
                # Force garbage collection after each file
                gc.collect()

        self.logger.info(f"Processed {processed_count} files")

def main():
    """Main entry point with configurable parameters."""
    try:
        scl_dir = 'SCL_Classified'
        band_dir = 'Raster_Classified'
        output_dir = "Raster_Classified_Cloud_Mask"

        # Initialize with smaller chunk size for lower memory usage
        cloud_masker = SentinelCloudMasker(
            scl_dir=scl_dir,
            band_dir=band_dir,
            output_dir=output_dir,
            chunk_size=512,  # Smaller chunks for lower memory usage
            log_level=logging.INFO
        )
        cloud_masker.process_files()

    except Exception as e:
        logging.error(f"Unexpected error in main process: {e}")

if __name__ == "__main__":
    main()