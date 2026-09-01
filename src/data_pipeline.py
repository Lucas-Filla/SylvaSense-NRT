import ee
import os
import requests
import zipfile
import io
import rasterio
from dotenv import load_dotenv

#define the 6 bands required by the U-Net
U_NET_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'NDVI']

#Crete square area of interest, or aoi around a specific point
def get_aoi(long, lat, buffer_km):
    point = ee.Geometry.Point([long, lat])
    # Multiply by 1000 to turn to meters
    return point.buffer(buffer_km * 1000).bounds()

# Calculate the NDVI (B8 = NIR, B4 = red from Sentinel-2)
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def get_labels(aoi):
    #calling the hansen dataset from the ee catalog
    hansen = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    #loss band is pre calculated array: 1 = deforested 0 = background
    label_mask = hansen.select('loss').clip(aoi)
    return label_mask.rename('TARGET')

# Fetches the sentinel data
def fetch(aoi, start, end):
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                #   .map(water_mask) #Retired this function because it wasn't working as well as I'd like, and I'd rather have the AI learn not to count the giant red splotches of water
                  .map(add_ndvi))
    
    # Returns most recent median to reduce overall noise
    return collection.median().clip(aoi)

#Without this function water would be a dark red
# def water_mask(image):
    # #Calculate MNDWI (modified normalized difference water index) using B3 (green) and B11 (SWIR)
    # ndwi = image.normalizedDifference(['B3', 'B11']).rename('NDWI')
    # #Binary mask of land vs water
    # land_only = ndwi.lt(0.0)
    # return image.updateMask(land_only)

def prepare_image(image, aoi):
    """Filters the composite image down to the 6 bands required by the U-Net"""
    return image.select(U_NET_BANDS).clip(aoi)

def download_patch(image, aoi):
    """Requests a raw GeoTIFF from EE, downloads it to memory, and converts it to clean Numpy Array for PyTorch"""
    print(f"Requesting patch from Earth Engine Servers...")

    download_url = image.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': aoi,
        'format': 'GEO_TIFF'
    })

    response = requests.get(download_url)
    #check if zip file
    if response.status_code != 200 or not response.content.startswith(b'PK'):
        print(f"Error from GEE server (Status {response.status_code}):")
        print(response.text[:500]) #first 500 char of error message
        raise RuntimeError("EE failed to generate valid GeoTIFF download.")
        
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        tif_filename = z.namelist()[0]
        z.extract(tif_filename, ".")

    with rasterio.open(tif_filename) as src:
        array3d = src.read()

    os.remove(tif_filename)

    print(f"Successfully extracted tensor shape: {array3d.shape}")
    return array3d

if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("EE_PROJECT_ID")
    ee.Initialize(project=project_id)
    
    aoi = get_aoi(-63.90, -8.76, 2.5)
    raw_image = fetch(aoi, '2025-01-01', '2026-05-12')
    formatted_image = prepare_image(raw_image, aoi)

    print("\nData Pipeline Initialized")
    print(f"Bands Selected: {formatted_image.bandNames().getInfo()}")

    satellite_array = download_patch(formatted_image, aoi)
