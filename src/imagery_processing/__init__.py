"""
Image processing functions for the AGB prediction pipeline.

This module handles loading, aligning, and preprocessing remote sensing data, including:
- Sentinel-2 multispectral bands
- ALOS PALSAR2 radar data
- Digital Elevation Model (DEM)
- Land cover data

Functions mostly return xarray.DataArray objects with Dask chunking for efficient lazy computation,
and encode additional features such as geographic coordinates or land cover transformations.
"""

from pathlib import Path
import rioxarray
from rasterio.enums import Resampling
import rasterio as rio
from rasterio.vrt import WarpedVRT
from pyproj import Transformer
from typing import Optional

from src.utils.config import *
from src.constants import *
from src.imagery_processing.helper import *

# Load common config
CFG = get_config("default.yaml")
CHUNKSIZE = get_chunk_size(CFG)

# ---------------------------
# Sentinel-2 processing
# ---------------------------
def process_s2_data() -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.DataArray, any, xr.DataArray]:
    """
    Load, reproject, and preprocess Sentinel-2 bands.

    Returns
    -------
    processed_bands : dict
        Dictionary of preprocessed Sentinel-2 bands (excluding SCL).
    mask : xr.DataArray
        Boolean mask identifying clouds and shadows using the SCL band.
    xds_ref : xr.DataArray
        Reference raster (used for alignment).
    crs : CRS
        Coordinate Reference System of the reference raster.
    encoded_da : xr.DataArray
        Latitude/longitude encoded as cosine/sine features for ML input.
    """
    # Load reference band
    ref_band_path = get_s2_band_path(S2_REF_BAND)
    xds_ref = rioxarray.open_rasterio(
        ref_band_path,
        masked=True,
        chunks=(1, CHUNKSIZE, CHUNKSIZE)
    ).squeeze()

    # Loop over bands
    processed_bands = {}
    for res, bands in S2_BANDS_PER_RES.items() :
        for band in bands:
            band_path = get_s2_band_path(band)
            if res != S2_REF_RES:
                if band == 'SCL':
                    resampling_method = Resampling.nearest
                else:
                    resampling_method = Resampling.bilinear
                with rio.open(band_path) as src:
                    with WarpedVRT(
                        src,
                        crs=xds_ref.rio.crs,
                        transform=xds_ref.rio.transform(),
                        height=xds_ref.rio.height,
                        width=xds_ref.rio.width,
                        resampling=resampling_method,
                    ) as vrt:
                        xds_reprojected = rioxarray.open_rasterio(
                            vrt,
                            chunks=(1, CHUNKSIZE, CHUNKSIZE),
                            masked=True
                        ).squeeze()
                xds_reprojected = fill_na(xds_reprojected)
                # Store result
                processed_bands[band] = xds_reprojected
            else:
                xds = rioxarray.open_rasterio(
                    band_path,
                    masked=True,
                    chunks=(1, CHUNKSIZE, CHUNKSIZE)
                ).squeeze()
                xds = fill_na(xds)
                processed_bands[band] = xds

    # Extract classification band for masking
    scl_band = processed_bands.pop('SCL')
    mask = (scl_band == NODATAVALS['S2_bands']) | (scl_band == 6) | (scl_band == 11)

    # Add offset
    print('Applying offset on S2 data')
    for band in processed_bands:
        processed_bands[band] += 0.1

    # Extract, convert and encode coordinates
    crs = xds_ref.rio.crs
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lat_cos, lat_sin, lon_cos, lon_sin = encode_coordinates(xds_ref, transformer)

    coords = {"y": xds_ref.y, "x": xds_ref.x, "spatial_ref": xds_ref.spatial_ref}

    lat_cos_da = xr.DataArray(lat_cos, dims=("y","x"), coords=coords, name="lat_cos")
    lat_sin_da = xr.DataArray(lat_sin, dims=("y","x"), coords=coords, name="lat_sin")
    lon_cos_da = xr.DataArray(lon_cos, dims=("y","x"), coords=coords, name="lon_cos")
    lon_sin_da = xr.DataArray(lon_sin, dims=("y","x"), coords=coords, name="lon_sin")

    encoded_da = xr.concat(
        [lat_cos_da, lat_sin_da, lon_cos_da, lon_sin_da],
        dim="band",
    ).assign_coords(band=["lat_cos", "lat_sin", "lon_cos", "lon_sin"]).transpose("y", "x", "band").chunk({'y': CHUNKSIZE, 'x': CHUNKSIZE, 'band': -1})
    encoded_da.name = None

    return processed_bands, mask, xds_ref, crs, encoded_da

# ---------------------------
# ALOS processing
# ---------------------------
def process_alos_data(xds_ref: xr.DataArray) -> xr.DataArray:
    """
    Load, align, fill, and gamma-transform ALOS PALSAR2 data.

    Parameters
    ----------
    xds_ref : xr.DataArray
        Reference raster for alignment.

    Returns
    -------
    xr.DataArray
        Preprocessed ALOS raster with gamma-naught transformation applied.
    """
    alos_image_path = get_alos_path()
    xds_alos = load_and_align_raster_to_reference(alos_image_path, xds_ref, resampling=Resampling.bilinear)
    xds_alos = xds_alos.assign_coords(band=list(xds_alos.attrs['long_name']))
    xds_alos = fill_na(xds_alos)
    xds_alos_gamma = xds_alos.where(
        xds_alos != NODATAVALS['ALOS_bands'], -9999.0  # replace nodata with -9999
    )
    # Apply gamma naught transformation lazily
    xds_alos_gamma = 10 * np.log10(xds_alos_gamma ** 2) - 83.0
    return xds_alos_gamma

# ---------------------------
# DEM processing
# ---------------------------
def process_dem_data(xds_ref: xr.DataArray) -> xr.DataArray:
    """Load, align, and fill missing values for Digital Elevation Model (DEM) data.

    Parameters
    ----------
    xds_ref : xr.DataArray
        Reference raster for alignment.

    Returns
    -------
    xr.DataArray
        Preprocessed DEM raster.
    """
    dem_image_path = get_dem_path()
    xds_dem = load_and_align_raster_to_reference(dem_image_path, xds_ref, resampling=Resampling.bilinear)
    xds_dem = fill_na(xds_dem)
    return xds_dem

# ---------------------------
# Land cover processing
# ---------------------------
def process_land_cover_data(xds_ref: xr.DataArray) -> xr.DataArray:
    """
    Load, align, fill, and encode land cover raster as cosine/sine/probability features.

    Parameters
    ----------
    xds_ref : xr.DataArray
        Reference raster for alignment.

    Returns
    -------
    xr.DataArray
        Encoded land cover raster with dimensions (y, x, band) for ML input.
    """
    land_cover_image_path = get_land_cover_path()
    xds_lc = load_and_align_raster_to_reference(land_cover_image_path, xds_ref, resampling=Resampling.nearest)
    xds_lc = fill_na(xds_lc)
    def encode_lc_numpy(lc_data):
        lc_map = lc_data[:, :, 0]
        lc_cos = np.where(lc_map == NODATAVALS['LC'], 0, (np.cos(2 * np.pi * lc_map / 100) + 1) / 2)
        lc_sin = np.where(lc_map == NODATAVALS['LC'], 0, (np.sin(2 * np.pi * lc_map / 100) + 1) / 2)
        lc_prob = lc_data[:, :, 1]
        lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)
        return np.stack([lc_cos, lc_sin, lc_prob], axis=-1)  # shape (..., 3)
    xds_lc_encoded = xr.apply_ufunc(
        encode_lc_numpy,
        xds_lc,
        input_core_dims=[['band']],                # input has a 'band' dimension
        output_core_dims=[['encoded_band']],       # output has a new dimension
        output_sizes={'encoded_band': 3},          # must specify its length!
        dask='parallelized',                       # use dask-aware parallelism
        output_dtypes=[xds_lc.dtype],              # same dtype as input
    )
    xds_lc_encoded = xds_lc_encoded.assign_coords(encoded_band=["lc_cos", "lc_sin", "lc_prob"])
    # Rename to 'band'
    xds_lc_encoded = xds_lc_encoded.rename({'encoded_band': 'band'})
    return xds_lc_encoded


# ---------------------------
# Band normalization
# ---------------------------
def normalize_bands(
    bands_da: xr.DataArray,
    norm_values: dict,
    band_order: Optional[list],
    norm_strat: str,
    nodata_value=None
) -> xr.DataArray:
    """
    Normalize all bands in an xarray.DataArray lazily, either in order or using all bands.

    Parameters
    ----------
    bands_da : xr.DataArray
        Input bands dataset.
    norm_values : dict
        Normalization parameters for each band.
    band_order : list or None
        If given, normalize bands in this order.
    norm_strat : str
        Normalization strategy: "mean_std", "pct", "min_max".
    nodata_value : optional
        Value treated as no-data (set to zero after normalization).

    Returns
    -------
    xr.DataArray
        Normalized dataset with same dimensions and Dask chunking.
    """
    if band_order is None:
        da_band_normed = normalize_data_xr(bands_da, norm_values, norm_strat, nodata_value)
        return da_band_normed 
    else:
        normed_bands = []
        for band in band_order:
            da_band = bands_da.sel(band=band)
            band_norm = norm_values[band]
            da_band_normed = normalize_data_xr(da_band, band_norm, norm_strat, nodata_value)
            normed_bands.append(da_band_normed)
        return xr.concat(normed_bands, dim=pd.Index(band_order, name="band")).transpose("y", "x", "band")
