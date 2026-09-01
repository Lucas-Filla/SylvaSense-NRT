import os
import numpy as np
import ee
from dotenv import load_dotenv
from data_pipeline import get_aoi, fetch, prepare_image, get_labels, download_patch

if __name__ == "__main__":
    load_dotenv
    project_id = os.getenv("EE_PROJECT_ID")
    ee.Initialize(project=project_id)