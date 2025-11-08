# No-data values associated to each source
NODATAVALS = {
    'S2_bands': 0, 
    'ALOS_bands': 0, 
    'DEM': -9999,
    'LC': 255
    }

# Sentinel-2 L2A bands that we want to use, organized by resolution
S2_BANDS_PER_RES = {
    '10m' : ['B02', 'B03', 'B04', 'B08'],
    '20m' : ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],
    '60m' : ['B01', 'B09']
    }

S2_REF_RES = '10m'
S2_REF_BAND = 'B02'