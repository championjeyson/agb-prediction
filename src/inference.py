import xarray as xr
import torch

from src.constants import *
from src.imagery_processing import (
    process_s2_data,
    process_alos_data,
    process_dem_data,
    process_land_cover_data,
    normalize_bands,
    )
from src.utils.config import *
from src.models import (
    load_model_config, 
    load_model_data_statistics,
    load_inference_model,
    predict_AGB,
    )

# Load common config
CFG = get_config("default.yaml")
CHUNKSIZE = get_chunk_size(CFG)

def run_inference():
    # Load model config and data statistics
    model_cfg = load_model_config()
    model_norm_values = load_model_data_statistics()

    # Initialize data list
    data = []

    # Process and prepare data
    #--------------------------

    # ----- S2 and coordinates (12 + 4 encoded bands) -----
    # Process
    s2_processed_bands, s2_mask, xds_s2_ref_band, ref_crs, xds_encoded_coordinates = process_s2_data()
    # Stack and reorder bands
    s2_band_order = model_cfg['bands']
    s2_processed_bands = xr.concat([s2_processed_bands[band] for band in s2_band_order], dim="band")
    s2_processed_bands = s2_processed_bands.assign_coords(band=s2_band_order)
    # Reorder dimensions
    s2_processed_bands = s2_processed_bands.transpose("y", "x", "band")
    # Normalize bands according to the statistics used in the model
    s2_processed_bands = normalize_bands(s2_processed_bands, model_norm_values['S2_bands'], s2_band_order, model_cfg['norm_strat'], NODATAVALS['S2_bands'])
    # Add data to data list
    data.extend([s2_processed_bands])
    data.extend([xds_encoded_coordinates])

    # ----- ALOS (2 bands) -----
    # Process
    xds_alos_gamma = process_alos_data(xds_s2_ref_band)
    # Normalize bands
    alos_order = ['HH', 'HV']
    xds_alos_gamma = normalize_bands(xds_alos_gamma, model_norm_values['ALOS_bands'], alos_order, model_cfg['norm_strat'], NODATAVALS['ALOS_bands'])
    # Add data to list
    data.extend([xds_alos_gamma])

    # ----- DEM (1 band) -----
    # Process
    xds_dem = process_dem_data(xds_s2_ref_band)
    # Normalize
    xds_dem = normalize_bands(xds_dem, model_norm_values['DEM'], None, model_cfg['norm_strat'], NODATAVALS['DEM']) #
    # Add data to list
    data.extend([xds_dem])

    # ----- LC (3 encoded bands)
    # Process
    xds_lc = process_land_cover_data(xds_s2_ref_band)
    # Add data to list
    data.extend([xds_lc])

    # Combine data into a unique cube
    data = xr.concat(data, dim="band").chunk({'y': CHUNKSIZE, 'x': CHUNKSIZE, 'band': -1})

    # Predict AGB
    #--------------
    
    # Set device for running model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model architecture and pretrained weights
    inference_model = load_inference_model(CFG['model']['arch'], CFG['model']['model_id'], model_cfg, device)
    # Predict
    output_path = Path(CFG['paths']['output_dir']) / 'agb_prediction.tif'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    predict_AGB(data, inference_model, device, output_path, xds_s2_ref_band.rio.crs, xds_s2_ref_band.rio.transform(), mask=s2_mask)

    return 0