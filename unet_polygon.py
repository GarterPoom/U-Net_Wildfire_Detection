import os
import re
import shutil  # To move files to the new folder
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import rasterio as rio
import numpy as np
from IPython.display import display
import pandas as pd
from typing import Optional

def find_all_files_in_root(root_folder, file_extension='.tif'):
    """
    Finds all files in a root folder with a given file extension.

    Parameters
    ----------
    root_folder : str
        The root folder to search in.
    file_extension : str, optional
        The file extension to search for. Defaults to '.tif'.

    Returns
    -------
    list
        A list of file paths (full paths) of all files found with the given extension.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(file_extension):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def create_output_folder_for_file(input_file_path, output_base_folder):
    """
    Creates an output folder for a given input file path within a specified base folder.

    Parameters
    ----------
    input_file_path : str
        The file path of the input file for which the output folder is to be created.
    output_base_folder : str
        The base folder where the output folder will be created.

    Returns
    -------
    str
        The path to the created output folder.
    """

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_folder = os.path.join(output_base_folder, base_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def extract_fire_date_from_filename(filename):
    """
    Extracts the fire date from a given filename in the format 'YYYYMMDDT<rest of filename>'
    and returns it as a string in the format 'YYYY-MM-DD'. If no fire date is found, returns None.

    Parameters
    ----------
    filename : str
        The filename to extract the fire date from.

    Returns
    -------
    str
        The fire date as a string in the format 'YYYY-MM-DD', or None if no fire date is found.
    """
    match = re.search(r'_(\d{8})T', filename)
    if match:
        date_str = match.group(1)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return formatted_date
    return None

def create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder, admin_shp_paths):
    """
    Creates a polygon shapefile from a given GeoTIFF file containing classified burnt areas.
    Performs intersection with multiple countries' administrative boundaries.
    
    Args:
        tif_file_path (str): Path to the input GeoTIFF file
        output_folder (str): Path to the output folder
        admin_shp_paths (dict): Dictionary of country names and their admin boundary shapefile paths
    """
    base_name = os.path.splitext(os.path.basename(tif_file_path))[0]
    output_shapefile_name = f"{base_name}_Burn_classified.shp"
    output_shapefile_path = os.path.join(output_folder, output_shapefile_name)

    # Load GeoTIFF file
    with rio.open(tif_file_path) as src:
        burn_classified = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Create shapes from raster
    mask = burn_classified == 1
    burnt_shapes = shapes(burn_classified, mask=mask, transform=transform)
    
    # Convert to geometries
    geoms = [shape(geom) for geom, value in burnt_shapes if value == 1]
    
    if not geoms:
        print(f"Warning: No burnt areas (value=1) found in {tif_file_path}")
        return None

    # Create GeoDataFrame with the found geometries
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    
    # Calculate area in original projection
    gdf['AREA'] = gdf.geometry.area
    gdf = gdf[gdf['AREA'] >= 500]  # Keep only areas >= 500 sqm
    
    # Convert to WGS84 for lat/long calculations
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    
    # Create a new GeoDataFrame for attribute data (in WGS84)
    gdf_attr = gpd.GeoDataFrame(geometry=gdf_wgs84.geometry, crs='EPSG:4326') # Change CRS to WGS1984
    gdf_attr = pd.DataFrame()
    gdf_attr['LATITUDE'] = gdf_wgs84.geometry.centroid.y
    gdf_attr['LONGITUDE'] = gdf_wgs84.geometry.centroid.x
    gdf_attr['FIRE_DATE'] = extract_fire_date_from_filename(os.path.basename(tif_file_path))
    gdf_attr['AREA'] = gdf['AREA'].values  # Maintain AREA in original units
    
    # Initialize empty GeoDataFrame for results
    intersected_results = []
    
    # Process each country's admin boundaries
    for country, admin_path in admin_shp_paths.items():
        try:
            admin_gdf = gpd.read_file(admin_path).to_crs(gdf.crs)
            if 'COUNTRY' not in admin_gdf.columns:
                admin_gdf['COUNTRY'] = country
            
            # Make sure to keep the AP_EN and PV_EN columns from admin_gdf
            country_intersection = gpd.overlay(gdf, admin_gdf, how='intersection')
            
            if not country_intersection.empty:
                intersected_results.append(country_intersection)
                print(f"Found intersections with {country}")
        except Exception as e:
            print(f"Error processing {country}: {str(e)}")
            continue
    
    if not intersected_results:
        print("Warning: No intersections found with any country. Using original burnt areas.")
        return None
    
    final_gdf = pd.concat(intersected_results, ignore_index=True)

    # Instead of using the values from gdf_attr directly, create a constant value for all rows
    fire_date = extract_fire_date_from_filename(os.path.basename(tif_file_path))
    
    # Assign attribute columns
    final_gdf['FIRE_DATE'] = fire_date  # Assign the same date to all rows

    # Calculate centroids from the final geometries
    final_gdf['LATITUDE'] = final_gdf.geometry.centroid.to_crs(epsg=4326).y
    final_gdf['LONGITUDE'] = final_gdf.geometry.centroid.to_crs(epsg=4326).x

    # Keep other columns as they are
    final_gdf['AREA'] = final_gdf.get('AREA', pd.NA) # Get Area in Square Meter Unit
    final_gdf['AREA_RAI'] = final_gdf['AREA'] / 1600 # Calculate Area in Rai which is Thai Area Measure Unit, Which is 1 Rai = 1600 sqm.
    final_gdf['TB_TN'] = final_gdf.get('TB_TN', pd.NA)  # Get TB_TN from admin boundaries
    final_gdf['AP_TN'] = final_gdf.get('AP_TN', pd.NA)  # Get AP_TN from admin boundaries
    final_gdf['PV_TN'] = final_gdf.get('PV_TN', pd.NA)  # Get PV_TN from admin boundaries
    final_gdf['TB_EN'] = final_gdf.get('TB_EN', pd.NA)  # Get TB_EN from admin boundaries
    final_gdf['AP_EN'] = final_gdf.get('AP_EN', pd.NA)  # Get AP_EN from admin boundaries
    final_gdf['PV_EN'] = final_gdf.get('PV_EN', pd.NA)  # Get PV_EN from admin boundaries
    final_gdf['COUNTRY'] = final_gdf.get('COUNTRY', pd.NA)
    final_gdf['ISO3'] = final_gdf.get('ISO3', pd.NA)
    
    # Reorder columns so geometry is last
    final_columns = ['FIRE_DATE', 'LATITUDE', 'LONGITUDE', 'AREA', 'AREA_RAI', 'TB_TN', 'TB_EN', 'AP_TN', 'AP_EN', 'PV_TN', 'PV_EN', 'COUNTRY', 'ISO3', 'geometry']

    final_gdf = final_gdf[final_columns]

    # Save to file
    final_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile', encoding='TIS-620') # Use TIS-620 encoding for Support Thai Characters.
    print(f"Polygon shapefile '{output_shapefile_path}' has been created.")

    # Display Final GeoDataframe
    display(final_gdf)
    
    return final_gdf

def main():
    """
    Main function to process GeoTIFF files and create polygon shapefiles for classified burnt areas.
    Handles multiple countries' administrative boundaries.
    """
    root_folder = r'Predicted_Mask'
    output_base_folder = r'unet_polygon'
    
    # Dictionary of country names and their admin boundary shapefile paths
    admin_shp_paths = {
        'Thailand': r'CLMVTH_Administrative_Boundary\Thailand\Thailand - Subnational Administrative Boundaries.shp',
        'Laos': r'CLMVTH_Administrative_Boundary\Laos\lao_admbnda_adm2_ngd_20191112.shp',
        'Cambodia': r'CLMVTH_Administrative_Boundary\Cambodia\khm_admbnda_adm3_gov_20181004.shp',
        'Myanmar': r'CLMVTH_Administrative_Boundary\Myanmar\Myanmar_Administrative_Boundary.shp',
        'Vietnam': r'CLMVTH_Administrative_Boundary\Vietnam\vnm_admbnda_adm2_gov_20201027.shp'
        # Add more countries as needed
    }

    tif_files = find_all_files_in_root(root_folder)
    for tif_file_path in tif_files:
        # Process each GeoTIFF file
        print(f"\nProcessing file: {tif_file_path}")

        # Create output folder
        output_folder = create_output_folder_for_file(tif_file_path, output_base_folder)

        # Create polygon shapefile
        final_gdf = create_polygon_shapefile_from_burnt_areas(tif_file_path, output_folder, admin_shp_paths)
        
if __name__ == "__main__":
    main()