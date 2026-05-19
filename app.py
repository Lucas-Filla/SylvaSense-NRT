import os
import ee
import streamlit as st
import geemap
import geemap.foliumap as geemap_folium
from src.data_pipeline import fetch, get_aoi
from dotenv import load_dotenv

load_dotenv()

st.title("SylvaSense NRT: Amazon Rainforest Monitoring")
st.write("Near Real-time deforestation tracker using Sentinel-2 and Deep Learning")

st.sidebar.header("Target Settings")
longitude = st.sidebar.number_input("Longitude", value=-63.90, format="%.2f")
latitude = st.sidebar.number_input("Latitude", value=-8.76, format="%.2f")
buffer = st.sidebar.slider("Buffer", 1, 20, 10)

if st.button("Fetch Latest Scan"):
    with st.spinner("Fetching..."):
        project_id = os.getenv("EE_PROJECT_ID")
        ee.Initialize(project=project_id)

        aoi = get_aoi(longitude, latitude, buffer)
        image = fetch(aoi, '2025-01-01', '2026-05-12')

        m = geemap_folium.Map()
        m.centerObject(aoi, 12)
        #Add the true color view
        m.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, "Natural Forest (RGB)")
        #Add the NDVI view
        m.addLayer(image, {'bands': ['NDVI'], 'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}, "Forest Health (NDVI)")
        #Embed into streamlit
        m.to_streamlit(height=600)
