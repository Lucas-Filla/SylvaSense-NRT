import ee
import os
from dotenv import load_dotenv
from src.data_pipeline import get_aoi, fetch, get_labels

def export_training_patch():
    load_dotenv()
    ee.Initialize(project=os.getenv("EE_PROJECT_ID"))

    print("Initializing...")

    #Porto velho cords (where the hansen dataset is)
    aoi = get_aoi(-63.9, -8.76, 10)
    #Sentinel-2 data from 2023 to match hansen data from 2023
    features = fetch(aoi, '2023-01-01', '2023-12-31')
    labels = get_labels(aoi)

    selected_features = features.select(['B2', 'B3', 'B4', 'B8', 'B11', 'NDVI'])
    training_tensor = selected_features.addBands(labels)

    print("Building GeoTIFF...")
    task = ee.batch.Export.image.toDrive( # type: ignore
        image=training_tensor,
        description='SylvaSense_Training_PortoVelho',
        folder='SylvaSense_Exports',
        scale=10,
        region=aoi,
        fileFormat='GeoTIFF',
        maxPixels=int(1e10)
    )

    task.start()
    print('Task sent to GEE')
    print('GeoTIFF will appear in google drive soon')

if __name__ == "__main__":
    export_training_patch() 
