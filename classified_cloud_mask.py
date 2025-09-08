# Import the os module for operating system interactions, such as file and directory handling
import os
# Import the logging module for logging messages and debugging
import logging
# Import the typing module for type hinting support
import typing
# Import numpy for numerical operations and array handling
import numpy as np
# Import rasterio for reading and writing geospatial raster data
import rasterio
# Import Affine from rasterio.transform for handling geospatial transformations
from rasterio.transform import Affine
# Import CRS from rasterio.crs for coordinate reference system handling
from rasterio.crs import CRS
# Import Window from rasterio.windows for defining subsets of raster data
from rasterio.windows import Window
# Import gdal from osgeo for advanced geospatial data processing
from osgeo import gdal
# Import gc for manual garbage collection to manage memory
import gc

# Define the SentinelCloudMasker class for handling cloud masking in Sentinel-2 imagery
class SentinelCloudMasker:
    """
    A memory-efficient class to handle cloud masking for Sentinel-2 imagery.
    Processes Sentinel-2 imagery to mask clouds, shadows, and other unwanted features.
    """
    # Define a dictionary mapping SCL (Scene Classification Layer) class values to their descriptions
    CLOUD_CLASSES = {
        3: "Cloud shadows",  # Represents cloud shadows in the SCL data
        6: "Water",         # Represents water bodies in the SCL data
        8: "Cloud medium probability",  # Represents areas with medium probability of clouds
        9: "Cloud high probability",    # Represents areas with high probability of clouds
    }

    # Initialize the SentinelCloudMasker with directory paths, chunk size, and logging level
    def __init__(self, scl_dir: str, band_dir: str, output_dir: str, 
                 chunk_size: int = 1024, log_level: int = logging.INFO):
        """
        Initialize the cloud masking processor.
        
        Args:
            scl_dir (str): Directory containing SCL (Scene Classification Layer) files
            band_dir (str): Directory containing band files (e.g., spectral bands)
            output_dir (str): Directory where output masked files will be saved
            chunk_size (int): Size of chunks to process at once for memory efficiency
            log_level (int): Logging level for controlling verbosity (e.g., logging.INFO)
        """
        # Configure the logging system with the specified log level and format
        logging.basicConfig(
            level=log_level,  # Set the logging level (e.g., INFO, DEBUG)
            format='%(asctime)s - %(levelname)s: %(message)s'  # Define log message format
        )
        # Create a logger instance for this class
        self.logger = logging.getLogger(__name__)
        # Store the chunk size for processing data in smaller blocks
        self.chunk_size = chunk_size

        # Validate the existence, type, and permissions of input and output directories
        self._validate_directories(scl_dir, band_dir, output_dir)
        
        # Store the SCL directory path
        self.scl_dir = scl_dir
        # Store the band directory path
        self.band_dir = band_dir
        # Store the output directory path
        self.output_dir = output_dir

        # Create a mapping of SCL files based on their filenames
        self.scl_files = self._get_file_mapping(scl_dir, '_SCL.tif')
        # Create a mapping of band files based on their filenames
        self.band_files = self._get_file_mapping(band_dir, '.tif')

    # Validate that the provided directories exist, are directories, and are readable
    def _validate_directories(self, *dirs: str) -> None:
        """
        Validate directory existence and permissions.
        
        Args:
            *dirs (str): Variable number of directory paths to validate
        
        Raises:
            ValueError: If a directory does not exist, is not a directory, or is not readable
        """
        # Iterate over each directory path provided
        for dir_path in dirs:
            # Check if the directory exists
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory does not exist: {dir_path}")
            # Check if the path is a directory
            if not os.path.isdir(dir_path):
                raise ValueError(f"Not a directory: {dir_path}")
            # Check if the directory is readable
            if not os.access(dir_path, os.R_OK):
                raise ValueError(f"Directory not readable: {dir_path}")

    # Build a dictionary mapping file identifiers to their full paths
    def _get_file_mapping(self, directory: str, suffix: str) -> dict:
        """
        Build file mapping dictionary.
        
        Args:
            directory (str): Directory to search for files
            suffix (str): File suffix to filter files (e.g., '.tif')
            
        Returns:
            dict: Dictionary mapping file identifiers to their full paths
        """
        # Walk through the directory and collect files with the specified suffix
        return {
            os.path.basename(f).split(suffix)[0]: os.path.join(root, f)  # Map filename (without suffix) to full path
            for root, _, files in os.walk(directory)  # Iterate through directory structure
            for f in files if f.endswith(suffix)  # Filter files by suffix
        }

    # Create a binary cloud mask from SCL data
    def create_cloud_mask(self, scl_data: np.ndarray) -> np.ndarray:
        """
        Create cloud mask from SCL data.
        
        Args:
            scl_data (np.ndarray): Scene Classification Layer data array
            
        Returns:
            np.ndarray: Binary mask where True indicates valid pixels and False indicates clouds
        """
        # Create a mask where pixels not in CLOUD_CLASSES are marked as valid (True)
        return ~np.isin(scl_data, list(self.CLOUD_CLASSES.keys()))

    # Generate windows for chunked processing of raster data
    def _get_chunk_windows(self, width: int, height: int) -> typing.Iterator[Window]:
        """
        Generate chunk windows for processing.
        
        Args:
            width (int): Width of the image in pixels
            height (int): Height of the image in pixels
            
        Yields:
            Window: Rasterio window object defining a chunk of the image
        """
        # Iterate over the image in chunks based on chunk_size
        for y in range(0, height, self.chunk_size):
            for x in range(0, width, self.chunk_size):
                # Define a window with coordinates and size, ensuring it fits within image bounds
                window = Window(
                    x,  # Starting x-coordinate
                    y,  # Starting y-coordinate
                    min(self.chunk_size, width - x),  # Width of the chunk
                    min(self.chunk_size, height - y)  # Height of the chunk
                )
                yield window  # Yield the window object

    # Process a single chunk of SCL and band data
    def process_chunk(self, scl_data: np.ndarray, band_data: np.ndarray) -> np.ndarray:
        """
        Process a single chunk of data.
        
        Args:
            scl_data (np.ndarray): SCL data chunk for cloud classification
            band_data (np.ndarray): Band data chunk to be masked
            
        Returns:
            np.ndarray: Processed chunk with clouds masked as NaN
        """
        # Create a cloud mask for the chunk
        cloud_mask = self.create_cloud_mask(scl_data)
        # Apply the mask to the band data, setting cloud pixels to NaN
        masked_chunk = np.where(cloud_mask, band_data, np.nan)
        # Convert the masked chunk to float32 for consistency
        return masked_chunk.astype(np.float32)

    # Process all files in the input directories in chunks
    def process_files(self) -> None:
        """
        Process all files in chunks to minimize memory usage.
        Matches SCL and band files, applies cloud masking, and saves results.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize a counter for processed files
        processed_count = 0

        # Iterate over SCL files
        for identifier, scl_path in self.scl_files.items():
            # Check if a matching band file exists
            if identifier not in self.band_files:
                self.logger.warning(f"No matching band file for {identifier}")
                continue

            try:
                # Log the start of processing for the current file
                self.logger.info(f"Processing: {identifier}")
                # Get the path to the corresponding band file
                band_path = self.band_files[identifier]

                # Open SCL and band files to access their data and metadata
                with rasterio.open(scl_path) as scl_ds, \
                     rasterio.open(band_path) as band_ds:
                    
                    # Copy the band file's metadata for the output file
                    out_meta = band_ds.meta.copy()
                    # Update metadata with output format and data type
                    out_meta.update({
                        "driver": "GTiff",  # Use GeoTIFF format
                        "dtype": "float32",  # Set data type to float32
                        "nodata": np.nan  # Set no-data value to NaN
                    })

                    # Define the path for the temporary uncompressed output file
                    temp_output_path = os.path.join(
                        self.output_dir, 
                        f"{identifier}_masked_uncompressed.tif"
                    )
                    
                    # Create and write to the output file
                    with rasterio.open(temp_output_path, "w", **out_meta) as dest:
                        # Process the image in chunks
                        for window in self._get_chunk_windows(band_ds.width, band_ds.height):
                            # Read the SCL chunk
                            scl_chunk = scl_ds.read(1, window=window)
                            # Read the band chunk (all bands)
                            band_chunks = band_ds.read(window=window)

                            # Process each band in the chunk
                            for band_idx in range(band_chunks.shape[0]):
                                # Apply cloud masking to the current band
                                processed_chunk = self.process_chunk(
                                    scl_chunk, 
                                    band_chunks[band_idx]
                                )
                                # Write the processed chunk to the output file
                                dest.write(processed_chunk, window=window, indexes=band_idx + 1)

                            # Delete chunk data to free memory
                            del scl_chunk, band_chunks
                            # Force garbage collection to manage memory
                            gc.collect()

                # Define the final compressed output file path
                output_path = os.path.join(self.output_dir, f"{identifier}_masked.tif")
                # Compress the temporary file using GDAL with LZW compression
                gdal.Translate(
                    destName=output_path,  # Output file path
                    srcDS=temp_output_path,  # Source temporary file
                    options=gdal.TranslateOptions(
                        format="GTiff",  # Output format
                        creationOptions=[
                            "COMPRESS=LZW",  # Use LZW compression
                            "PREDICTOR=3",  # Optimize for floating-point data
                            "TILED=YES",  # Enable tiling for efficient access
                            "BLOCKXSIZE=256",  # Set tile width
                            "BLOCKYSIZE=256",  # Set tile height
                            "BIGTIFF=YES"  # Support large files
                        ]
                    )
                )

                # Remove the temporary uncompressed file
                os.remove(temp_output_path)
                # Increment the processed file counter
                processed_count += 1
                # Log completion of processing for the current file
                self.logger.info(f"Completed processing: {identifier}")

            except Exception as e:
                # Log any errors that occur during processing
                self.logger.error(f"Error processing {identifier}: {e}")
            finally:
                # Force garbage collection after each file to manage memory
                gc.collect()

        # Log the total number of files processed
        self.logger.info(f"Processed {processed_count} files")

# Define the main function as the entry point for the script
def main():
    """
    Main entry point with configurable parameters.
    Sets up directories and initializes the cloud masking process.
    """
    try:
        # Define the directory for SCL files
        scl_dir = 'SCL_Classified'
        # Define the directory for band files
        band_dir = 'Raster_Classified'
        # Define the output directory for masked files
        output_dir = "Raster_Classified_Cloud_Mask"

        # Initialize the SentinelCloudMasker with specified parameters
        cloud_masker = SentinelCloudMasker(
            scl_dir=scl_dir,  # Directory for SCL files
            band_dir=band_dir,  # Directory for band files
            output_dir=output_dir,  # Directory for output files
            chunk_size=512,  # Use smaller chunk size for memory efficiency
            log_level=logging.INFO  # Set logging level to INFO
        )
        # Process all files using the cloud masker
        cloud_masker.process_files()

    except Exception as e:
        # Log any unexpected errors in the main process
        logging.error(f"Unexpected error in main process: {e}")

# Check if the script is run directly and execute the main function
if __name__ == "__main__":
    main()