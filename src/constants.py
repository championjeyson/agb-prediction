"""
Constants used across the AGB prediction pipeline.

This module defines no-data values, Sentinel-2 band selections, and reference bands.
All constants are grouped logically with explanations for clarity.

Example
-------
from src.constants import NODATAVALS

print(NODATAVALS['DEM'])  # -9999
"""

# ---------------------------
# No-data values
# ---------------------------
# Default values representing missing or invalid data for each data source.
NODATAVALS = {
    'S2_bands': 0,      # Sentinel-2 reflectance bands
    'ALOS_bands': 0,    # ALOS PALSAR bands
    'DEM': -9999,       # Digital Elevation Model
    'LC': 255           # Land cover raster
}

# ---------------------------
# Sentinel-2 bands by resolution
# ---------------------------
# Organized according to the spatial resolution of each band.
S2_BANDS_PER_RES = {
    '10m': ['B02', 'B03', 'B04', 'B08'],                     # Blue, Green, Red, NIR
    '20m': ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],  # Vegetation, SWIR, Scene classification
    '60m': ['B01', 'B09']                                     # Coastal aerosol, Water vapor
}

# ---------------------------
# Sentinel-2 reference band
# ---------------------------
S2_REF_RES = '10m'   # Reference resolution used for resampling or alignment
S2_REF_BAND = 'B02'  # Reference band for geometric alignment or normalization