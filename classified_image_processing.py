import os # For file and directory operations
import sys # For system-specific parameters and functions
import numpy as np # For numerical operations
from osgeo import gdal # For geospatial data processing
from pathlib import Path # For path manipulations
import time # For sleep function
import logging # For logging

def setup_logging(): 
    """
    Set up logging to help diagnose issues.
    """
    logging.basicConfig( # Configure logging
        level=logging.INFO, # Set logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s', # Log message format
        handlers=[ # Handlers for logging
            logging.FileHandler('sentinel_processing.log'), # Log to a file
            logging.StreamHandler(sys.stdout) # Also log to standard output
        ]
    )
    return logging.getLogger(__name__)

def resample_image(input_path, output_path, target_resolution=10):
    """
    Resamples a single image to a target resolution using GDAL and saves it as a compressed GeoTIFF file.
    """
    try:
        print(f"Resampling image: {input_path} to {output_path} at {target_resolution}m resolution.") # Log the resampling action
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the input dataset
        src_ds = gdal.Open(input_path)
        if not src_ds:
            raise ValueError(f"Could not open {input_path}")
            
        # Get the input resolution
        gt = src_ds.GetGeoTransform()
        input_res = gt[1]  # pixel width
        
        # Calculate new dimensions
        src_xsize = src_ds.RasterXSize # original width
        src_ysize = src_ds.RasterYSize # original height
        dst_xsize = int(src_xsize * (input_res / target_resolution)) # new width
        dst_ysize = int(src_ysize * (input_res / target_resolution)) # new height

        # Create translation options
        translate_options = gdal.TranslateOptions( # Options for gdal.Translate
            format='GTiff', # Output format
            width=dst_xsize, # New width
            height=dst_ysize, # New height
            resampleAlg=gdal.GRA_NearestNeighbour, # Resampling algorithm
            creationOptions=[ # Creation options for output file
                'COMPRESS=LZW', # Use LZW compression
                'PREDICTOR=2', # Use horizontal differencing
                'TILED=YES', # Create tiled GeoTIFF
                'BLOCKXSIZE=256', # Block size in X direction
                'BLOCKYSIZE=256', # Block size in Y direction
                'BIGTIFF=YES' # Allow BigTIFF files
            ]
        )
        
        # Perform resampling with compression
        gdal.Translate( # Translate the dataset
            destName=output_path, # Output file path
            srcDS=src_ds, # Source dataset
            options=translate_options # Options for translation
        )
        
        # Close the dataset
        src_ds = None
        
        print(f"Resampling completed with compression: {output_path}") # Log completion
        return True # Indicate success
    
    except Exception as e: # Catch any exceptions
        print(f"Error resampling image {input_path}: {str(e)}") # Log the error
        return False # Indicate failure
 
def safe_remove(file_path, max_attempts=5, delay=1):
    """
    Safely remove a file with multiple attempts and delay between attempts.
    """
    file_path = Path(file_path) # Convert to Path object
    if not file_path.exists(): # Check if file exists
        return True # If not, consider it removed
    
    for attempt in range(max_attempts): # Try multiple times
        try:
            file_path.unlink() # Attempt to remove the file
            return True # If successful, return True
        
        except PermissionError: # Catch permission errors
            if attempt < max_attempts - 1: # If not the last attempt
                time.sleep(delay) # Wait before retrying
            continue # Retry

        except Exception as e: # Catch other exceptions
            print(f"Error removing file {file_path}: {str(e)}") # Log the error
            return False # Indicate failure
        
    return False # If all attempts fail, return False

def build_pyramids_nearest(raster_path, overview_levels=[2, 4, 8, 16, 32], resample_alg='NEAREST'):
    """
    Build raster pyramids (overviews) using Nearest Neighbor resampling.
    
    Args:
        raster_path (str): Path to the GeoTIFF file
        overview_levels (list): List of overview levels to build
        resample_alg (str): Resampling algorithm ('NEAREST', 'AVERAGE', etc.)
    """
    try:
        print(f"Building pyramids for: {raster_path}") # Log the pyramid building action
        
        dataset = gdal.Open(raster_path, gdal.GA_Update) # Open dataset in update mode, in update mode raster pyramid will not generate OVR file but embed in the original file
        if not dataset: # Check if dataset opened successfully
            raise ValueError(f"Could not open raster for pyramid building: {raster_path}") # Raise error if cannot open
        
        # Build overviews
        dataset.BuildOverviews(resample_alg, overview_levels) # Build the overviews
        dataset = None  # Close dataset 
        
        print(f"Successfully built pyramids: {overview_levels} using {resample_alg}") # Log success
        return True # Indicate success

    except Exception as e: # Catch any exceptions
        print(f"Error building pyramids for {raster_path}: {str(e)}") # Log the error
        return False # Indicate failure

def process_bands(input_folder, output_folder, scl_output_folder=None):
    """
    Processes Sentinel-2 band files:
      - Resamples and compresses to 10m
      - Stacks spectral bands into a single raster
      - Computes vegetation/water indices and appends them as extra bands
      - Exports SCL (Scene Classification Layer) separately if requested
      - Builds pyramids for fast visualization
    """
    temp_folder = None # Initialize temp_folder variable

    try:
        print(f"Processing bands in folder: {input_folder}") # Log the processing action
        
        input_folder = Path(input_folder) # Convert to Path object
        if not input_folder.exists(): # Check if input folder exists
            raise ValueError(f"Input folder does not exist: {input_folder}") # Raise error if not
        
        output_folder = Path(output_folder) # Convert to Path object
        if not output_folder.exists(): # Check if output folder exists
            raise ValueError(f"Output folder does not exist: {output_folder}") # Raise error if not
        
        output_folder.mkdir(parents=True, exist_ok=True) # Ensure output folder exists

        # Handle SCL output folder
        if scl_output_folder: # If SCL output folder is provided
            scl_output_folder = Path(scl_output_folder) # Convert to Path object
            scl_output_folder.mkdir(parents=True, exist_ok=True) # Ensure SCL output folder exists
        
        jp2_files = list(input_folder.glob('*.jp2')) # List all JP2 files in the input folder
        if not jp2_files: # If no JP2 files found
            print("No JP2 files found in the input folder.") # Log the absence of files
            return # Exit the function

        temp_folder = output_folder / 'temp' # Create a temporary folder inside the output folder
        temp_folder.mkdir(parents=True, exist_ok=True) # Ensure temp folder exists

        resampled_files = {} # Dictionary to hold resampled file paths
        scl_file = None # Variable to hold SCL file path

        # Band mapping
        band_map = {
            'B02': 'B02', 'B03': 'B03', 'B04': 'B04', 'B05': 'B05',
            'B06': 'B06', 'B07': 'B07', 'B08': 'B08', 'B8A': 'B8A',
            'B11': 'B11', 'B12': 'B12'
        }

        for jp2_file in jp2_files: # Process each JP2 file
            output_path = temp_folder / f"{jp2_file.stem}_resampled.tif" # Define output path for resampled file
            
            if resample_image(str(jp2_file), str(output_path)): # Resample the image
                # Check for SCL file
                if 'SCL' in jp2_file.name: # If the file is SCL
                    scl_file = str(output_path) # Store the SCL file path
                    continue # Skip further processing for SCL
                
                for band_key in band_map: # Check each band key
                    if band_key in jp2_file.name: # If the band key is in the file name
                        resampled_files[band_key] = str(output_path) # Store the resampled file path
                        break # Exit the loop once matched

        # Process regular bands in standard order
        ordered_bands = ['B02', 'B03', 'B04', 'B05', 'B06',
                         'B07', 'B08', 'B8A', 'B11', 'B12']
        final_resampled_files = [resampled_files[b] for b in ordered_bands if b in resampled_files] # Ensure correct order

        if not final_resampled_files: # If no valid bands were processed
            raise ValueError("No valid band files were processed") # Raise error

        # Filename generation
        sample_filename = jp2_files[0].name # Use the first JP2 file as a sample
        parts = sample_filename.split('_') # Split the filename by underscores
        tile_date_timestamp = f"{parts[0]}_{parts[1]}" # Extract tile and date information
        output_filename = f"{tile_date_timestamp}.tif" # Define output filename
        output_path = output_folder / output_filename # Define full output path

        print(f"Creating compressed output file: {output_path}") # Log the output file creation

        # ✅ Load spectral bands into memory
        band_data = {} # Dictionary to hold band data
        for band_name, path in resampled_files.items(): # Load each resampled band
            ds = gdal.Open(path) # Open the dataset
            band_data[band_name] = ds.ReadAsArray().astype(np.float32) # Read as array and convert to float32
            ds = None # Close the dataset

        # ✅ Compute indices
        eps = 1e-6  # avoid div-by-zero

        # BAIS2 calculation (Burned Area Index for Sentinel-2)
        term1 = 1 - np.sqrt((band_data['B06'] * band_data['B07'] * band_data['B8A']) / (band_data['B04'] + eps))
        term2 = ((band_data['B12'] - band_data['B8A']) /
                 np.sqrt(band_data['B12'] + band_data['B8A'] + eps)) + 1
        bais2 = term1 * term2

        # NDVI calculation (Normalized Difference Vegetation Index)
        ndvi = (band_data['B08'] - band_data['B04']) / (band_data['B08'] + band_data['B04'] + eps)

        # NDWI calculation (Normalized Difference Water Index)
        ndwi = (band_data['B03'] - band_data['B08']) / (band_data['B03'] + band_data['B08'] + eps)

        # SAVI calculation (Soil Adjusted Vegetation Index)
        L = 1.0
        savi = ((band_data['B08'] - band_data['B04']) /
                (band_data['B08'] + band_data['B04'] + L)) * (1 + L)

        indices = {     # Dictionary to hold computed indices
            "BAIS2": bais2, # Burned Area Index for Sentinel-2
            "NDVI": ndvi, # Normalized Difference Vegetation Index
            "NDWI": ndwi, # Normalized Difference Water Index
            "SAVI": savi # Soil Adjusted Vegetation Index
        }

        # ✅ Create final raster with spectral bands + indices
        ref_ds = gdal.Open(final_resampled_files[0]) # Open reference dataset for geotransform and projection
        driver = gdal.GetDriverByName("GTiff") # Get GTiff driver
        total_bands = len(final_resampled_files) + len(indices) # Total number of bands (spectral + indices)

        out_ds = driver.Create( # Create output dataset
            str(output_path), # Output file path
            ref_ds.RasterXSize, # X size
            ref_ds.RasterYSize, # Y size
            total_bands, # Total number of bands
            gdal.GDT_Float32, # Data type
            options=[ # Creation options
                'COMPRESS=LZW', # Use LZW compression
                'PREDICTOR=2', # Use horizontal differencing
                'TILED=YES', # Create tiled GeoTIFF
                'BLOCKXSIZE=256', # Block size in X direction
                'BLOCKYSIZE=256', # Block size in Y direction
                'BIGTIFF=YES' # Allow BigTIFF files
            ]
        )

        out_ds.SetProjection(ref_ds.GetProjection()) # Set projection
        out_ds.SetGeoTransform(ref_ds.GetGeoTransform()) # Set geotransform

        # Write spectral bands
        for idx, band_path in enumerate(final_resampled_files, start=1): # Write each spectral band
            ds_band = gdal.Open(band_path) # Open the band dataset
            arr = ds_band.ReadAsArray().astype(np.float32) # Read as array and convert to float32
            out_band = out_ds.GetRasterBand(idx) # Get the output band
            out_band.WriteArray(arr) # Write the array to the output band
            out_band.SetDescription(ordered_bands[idx-1]) # Set band description
            ds_band = None # Close the band dataset

        # Write indices
        for i, (name, arr) in enumerate(indices.items(), start=len(final_resampled_files)+1): # Write each index
            out_band = out_ds.GetRasterBand(i) # Get the output band
            out_band.WriteArray(arr.astype(np.float32)) # Write the array to the output band
            out_band.SetDescription(name) # Set band description

        out_ds.FlushCache() # Flush cache to ensure all data is written
        out_ds = None # Close the output dataset
        ref_ds = None # Close the reference dataset

        # ✅ Build pyramids
        build_pyramids_nearest(str(output_path))

        # ✅ Export SCL separately if requested
        if scl_file and scl_output_folder: # If SCL file and output folder are provided
            scl_output_filename = f"{tile_date_timestamp}_SCL.tif" # Define SCL output filename
            scl_output_path = scl_output_folder / scl_output_filename # Define full SCL output path
            
            translate_options_scl = gdal.TranslateOptions( # Options for translating SCL file
                format='GTiff', # Output format
                creationOptions=[ # Creation options for output file
                    'COMPRESS=LZW', # Use LZW compression
                    'PREDICTOR=2', # Use horizontal differencing
                    'TILED=YES', # Create tiled GeoTIFF
                    'BLOCKXSIZE=256', # Block size in X direction
                    'BLOCKYSIZE=256' # Block size in Y direction
                ]
            )
            
            gdal.Translate( # Translate the SCL file
                destName=str(scl_output_path), # Output file path
                srcDS=scl_file, # Source dataset
                options=translate_options_scl # Options for translation
            )
            
            print(f"Exported SCL file: {scl_output_path}") # Log SCL export
        
        # ✅ Cleanup
        print("Cleaning up temporary files.") # Log cleanup action
        for path in list(resampled_files.values()): # Remove all resampled files
            safe_remove(path) # Safely remove the file

        if scl_file: # If SCL file was processed
            safe_remove(scl_file) # Safely remove the SCL file

        if temp_folder and temp_folder.exists(): # If temp folder exists
            try: # Attempt to remove the temp folder
                temp_folder.rmdir() # Remove the temp folder
            except Exception as e: # Catch any exceptions
                print(f"Warning: Could not remove temp folder: {str(e)}") # Log the warning

    except Exception as e: # Catch any exceptions during processing
        print(f"Error processing bands: {str(e)}") # Log the error
        raise # Re-raise the exception for higher-level handling
    
    finally: # Always execute cleanup
        # Final cleanup attempt
        if temp_folder and temp_folder.exists(): # If temp folder exists
            try: # Attempt to remove any remaining files and the folder
                for file in temp_folder.glob('*'): # Iterate over files in temp folder
                    safe_remove(file) # Safely remove each file
                temp_folder.rmdir() # Remove the temp folder
            except Exception as e: # Catch any exceptions
                print(f"Warning: Failed final cleanup: {str(e)}") # Log the warning

def find_and_process_folders(root_folder, output_folder):
    """
    Searches for and processes folders containing .jp2 files.
    """
    try:
        root_folder = Path(root_folder) # Convert to Path object
        output_folder = Path(output_folder) # Convert to Path object
        
        print(f"Searching for folders in: {root_folder}") # Log the search action
        
        for dirpath in root_folder.rglob('*'): # Recursively search for directories
            if dirpath.is_dir() and any(f.suffix == '.jp2' for f in dirpath.iterdir()): # Check if directory contains JP2 files
                relative_path = dirpath.relative_to(root_folder) # Get relative path
                current_output_folder = output_folder / relative_path # Define current output folder
                print(f"Found JP2 files in: {dirpath}. Processing...") # Log the processing action
                process_bands(dirpath, current_output_folder) # Process the bands in the directory
        
        print("All folders processed.") # Log completion    
        
    except Exception as e: # Catch any exceptions
        print(f"Error processing folders: {str(e)}") # Log the error
        sys.exit(1) # Exit with error
 
def main():
    # Configure logging
    logger = setup_logging()

    try:
        # Enable GDAL exceptions
        gdal.UseExceptions()
        
        # Get current working directory
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")

        # Check input folders
        root_folder = current_dir / 'Classified_Image'
        output_folder = current_dir / 'Raster_Classified'
        scl_output_folder = current_dir / 'SCL_Classified'

        # Check if input folder exists
        if not root_folder.exists():
            logger.error(f"Input folder does not exist: {root_folder}")
            logger.info("Please create a 'Classified_Image' folder and place Sentinel-2 JP2 files inside.")
            sys.exit(1)

        # Create output folders if they don't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        scl_output_folder.mkdir(parents=True, exist_ok=True)

        # Find and process JP2 files
        jp2_files = list(root_folder.rglob('*.jp2'))
        
        if not jp2_files:
            logger.error(f"No JP2 files found in {root_folder}")
            logger.info("Ensure Sentinel-2 JP2 files are present in the 'Classified_Image' folder.")
            sys.exit(1)

        logger.info(f"Found {len(jp2_files)} JP2 files to process") # Log the number of files found

        def find_and_process_folders(root_folder, output_folder, scl_output_folder): # Nested function to find and process folders
            try:
                root_folder = Path(root_folder) # Convert to Path object
                output_folder = Path(output_folder) # Convert to Path object
                scl_output_folder = Path(scl_output_folder) # Convert to Path object
                
                logger.info(f"Searching for folders in: {root_folder}") # Log the search action
                
                processed_folders = 0 # Counter for processed folders
                for dirpath in root_folder.rglob('*'): # Recursively search for directories
                    if dirpath.is_dir() and any(f.suffix == '.jp2' for f in dirpath.iterdir()): # Check if directory contains JP2 files
                        relative_path = dirpath.relative_to(root_folder) # Get relative path
                        current_output_folder = output_folder / relative_path # Define current output folder
                        current_scl_output_folder = scl_output_folder / relative_path # Define current SCL output folder
                        logger.info(f"Found JP2 files in: {dirpath}. Processing...") # Log the processing action
                        process_bands(dirpath, current_output_folder, current_scl_output_folder) # Process the bands in the directory
                        processed_folders += 1 # Increment processed folder count
                
                if processed_folders == 0: # If no folders were processed
                    logger.warning("No folders with JP2 files were processed.") # Log a warning
                else: # If folders were processed
                    logger.info(f"Processed {processed_folders} folders.") # Log the number of processed folders
                
            except Exception as e: # Catch any exceptions
                logger.error(f"Error processing folders: {str(e)}") # Log the error
                sys.exit(1) # Exit with error
        
        # Run processing
        find_and_process_folders(root_folder, output_folder, scl_output_folder)
        
        logger.info("Processing complete.") # Log completion

    except Exception as e: # Catch any unexpected exceptions
        logger.error(f"Unexpected error: {str(e)}") # Log the unexpected error
        sys.exit(1) # Exit with error

if __name__ == "__main__":
    main() # Run the main function if this script is executed directly