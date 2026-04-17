# SylvaSense-NRT
### A Near-Real-Time Deforestation Tracker via Multi-Spectral U-Net Architecture

## Overview

Sylva = Latin for *wood, **forest**, or woodland*

Sense = We're sensing that ***sylva***

NRT = Near-real-time

SylvaSense NRT is designed to be an automated monitoring system to detect forest loss and potentially illegal logging activities. Using ***Sentinel-2*** satellite imagery, ***U-Net*** deep learning architecture, and Google Earth Engine (GEE), the system can accurately identify environmental degradation patterns in near-real-time. 

SylvaSense is different from existing deforestation trackers because it utilizes a Multi-Spectral U-Net to leverage non-visible light frequencies (NIR/SWIR) and spacial context which aren't used in traditional RGB-based thresholding. This drastically increases the accuracy of the tool and enables the detection of non-linear degradation patterns that are often missed by standard tools. 

## Motivation

Sustainability is a very important practice to me, and illegal logging / deforestation goes directly against it. This project serves as a chance for me to directly support sustainability while enhancing my programming knowledge and skills. 

## Research Question
> "To what degree can a Multi-Spectral U-Net architecture improve the Mean Intersection over Union (mIoU) of near-real-time deforestation tracking compared to traditional RGB-based change detection methods?"

## Tech Stack
- **Data:** Google Earth Engine API (Sentinel-2 Multi-Spectral)
- **AI/ML:** PyTorch / U-Net
- **Geospatial:** Geemap, Rasterio, NumPy
- **Dashboard:** Streamlit

## Roadmap
- [ ] GEE Environment Setup & Data Pipeline
- [ ] Multi-Spectral Feature Engineering (NDVI/NIR)
- [ ] U-Net Model Architecture & Training
- [ ] Change Detection Logic Implementation
- [ ] Live Dashboard Deployment

