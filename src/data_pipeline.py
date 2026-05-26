import ee
import os
from dotenv import load_dotenv

#Crete square area of interest, or aoi around a specific point
def get_aoi(long, lat, buffer_km):
    point = ee.Geometry.Point([long, lat])
    # Multiply by 1000 to turn to meters
    return point.buffer(buffer_km * 1000).bounds()

# Calculate the NDVI (B8 = NIR, B4 = red from Sentinel-2)
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Fetches the sentinel data
def fetch(aoi, start, end):
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .map(water_mask)
                  .map(add_ndvi))
    
    # Returns most recent median to reduce overall noise
    return collection.median().clip(aoi)

#Without this function water would be a dark red
def water_mask(image):
    #Calculate MNDWI (modified normalized difference water index) using B3 (green) and B11 (SWIR)
    ndwi = image.normalizedDifference(['B3', 'B11']).rename('NDWI')
    #Binary mask of land vs water
    land_only = ndwi.lt(0.0)
    return image.updateMask(land_only)

if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("EE_PROJECT_ID")
    ee.Initialize(project=project_id)
    
    aoi = get_aoi(-63.90, -8.76, 10)
    image = fetch(aoi, '2025-01-01', '2026-05-12')
    print("\nData Pipeline initialized with AOI of 20km x 20km")
    print(f"Bands Available: {image.bandNames().getInfo()}")
