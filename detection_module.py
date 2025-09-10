from classified_image_processing import main as resampling # Import Main module from classified_image_processing
from classified_cloud_mask import main as cloud_mask # Import Main module from classified_cloud_mask
from unet_wildfire_predict import main as wildfire # Import Main module from wildfire_classified
from unet_polygon import main as polygon # Import Main module from wildfire_polygon

def main():
    """
    Main function to execute the complete detection workflow.

    This function orchestrates the entire process by calling the following
    modules in sequence:

    1. resampling: Resamples Sentinel-2 images to a target resolution.
    2. cloud_mask: Applies cloud masking to the resampled images.
    3. wildfire: Classifies the processed images to detect wildfire areas.
    4. polygon: Creates polygon shapefiles from the classified wildfire areas.
    """

    resampling()
    cloud_mask()
    wildfire()
    polygon()

    print("Detection workflow completed successfully.")

if __name__ == "__main__":
    main()