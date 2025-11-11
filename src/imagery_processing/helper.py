"""
Helper functions for raster imagery processing in the AGB prediction pipeline.

Includes:
- File path retrieval for input images (Sentinel-2, ALOS, DEM, Land Cover)
- Filling missing values (NaNs) chunk-wise
- Encoding geographic coordinates
- Normalizing raster bands
- Aligning rasters to reference datasets

All functions are designed to handle large raster datasets efficiently,
leveraging Dask and xarray where appropriate.
"""

import numpy as np
from scipy import ndimage
import xarray as xr
import dask.array as da
import pandas as pd
from rasterio.enums import Resampling
import rasterio as rio
import rioxarray
from rasterio.vrt import WarpedVRT
from src.utils.config import *

# Load common config
CFG = get_config("default.yaml")
CHUNKSIZE = get_chunk_size(CFG)

# ---------------------------
# Input image paths
# ---------------------------

def get_input_image_dir_path() -> Path:
    """
    Retrieve the input image directory from the configuration.

    Returns
    -------
    Path
        Path to the folder containing input images.

    Raises
    ------
    ValueError
        If the input directory is not set in the config.
    FileNotFoundError
        If the folder does not exist on disk.
    """
    input_image_dir_path = CFG.get('paths', {}).get('input_image_dir', None)
    if input_image_dir_path is None:
        raise ValueError('Path to input image folder not set in the configuration file.')
    input_image_dir_path = Path(input_image_dir_path)
    if not input_image_dir_path.exists():
        raise FileNotFoundError(
            f'Path to input image folder does not exist: {input_image_dir_path.resolve().absolute()}'
        )
    return input_image_dir_path

def get_s2_band_path(band: str) -> Path:
    """Return the file path for a Sentinel-2 band."""
    input_image_dir_path = get_input_image_dir_path()
    path = input_image_dir_path / f'S2_{band}.tif'
    if not path.exists():
        raise FileNotFoundError(f'Path to input image does not exist: {path.resolve().absolute()}')
    return path

def get_alos_path() -> Path:
    """Return the file path for the ALOS PALSAR2 raster."""
    input_image_dir_path = get_input_image_dir_path()
    path = input_image_dir_path / f'SAR_PALSAR2.tif'
    if not path.exists():
        raise FileNotFoundError(f'Path to input image does not exist: {path.resolve().absolute()}')
    return path

def get_dem_path() -> Path:
    """Return the file path for the AW3D30 DEM raster."""
    input_image_dir_path = get_input_image_dir_path()
    path = input_image_dir_path / f'AW3D30.tif'
    if not path.exists():
        raise FileNotFoundError(f'Path to input image does not exist: {path.resolve().absolute()}')
    return path

def get_land_cover_path() -> Path:
    """Return the file path for the CGLS_LC100 land cover raster."""
    input_image_dir_path = get_input_image_dir_path()
    path = input_image_dir_path / f'CGLS_LC100.tif'
    if not path.exists():
        raise FileNotFoundError(f'Path to input image does not exist: {path.resolve().absolute()}')
    return path

# ---------------------------
# Fill missing values
# ---------------------------
def fill_na_chunkwise(block: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a 2D or 3D numpy array using nearest neighbor interpolation.
    Works within a single chunk only.

    Parameters
    ----------
    block : np.ndarray
        Input array (2D or 3D with last axis as band).

    Returns
    -------
    np.ndarray
        Array with NaNs filled using nearest neighbor values.
    """
    # if 3D (y, x, band), fill each band separately
    if block.ndim == 3:
        filled = np.empty_like(block)
        for b in range(block.shape[2]):
            mask = np.isnan(block[:, :, b])
            if mask.any():
                indices = ndimage.distance_transform_edt(
                    mask,
                    return_distances=False,
                    return_indices=True
                )
                filled[:, :, b] = block[:, :, b][indices[0], indices[1]]
            else:
                filled[:, :, b] = block[:, :, b]
        return filled
    else:  # 2D case
        mask = np.isnan(block)
        if mask.any():
            indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
            return block[tuple(indices)]
        else:
            return block
        
def fill_na(xds: xr.DataArray, overlap: int = 25) -> xr.DataArray:
    """
    Fill NaNs in an xarray.DataArray lazily using Dask's map_overlap.

    Parameters
    ----------
    xds : xr.DataArray
        Input raster dataset.
    overlap : int, optional
        Number of pixels to overlap between chunks. Default is 25.

    Returns
    -------
    xr.DataArray
        Raster with NaNs filled.
    """
    # Determine chunk sizes along y and x
    chunk_y, chunk_x = xds.chunks[0][0], xds.chunks[1][0]  # assume uniform chunking

    # Validate overlap
    if overlap > chunk_y or overlap > chunk_x:
        raise ValueError(
            f"Requested overlap ({overlap}) is larger than the chunk size "
            f"(y: {chunk_y}, x: {chunk_x}). Reduce overlap or rechunk the array."
        )
    
    xds_filled = xr.DataArray(
        xds.data.map_overlap(
                fill_na_chunkwise,
                depth={0: overlap, 1: overlap},  # y -> dim 0, x -> dim 1
                boundary='reflect',
                trim=True
            ),
            dims=xds.dims,
            coords=xds.coords,
            attrs=xds.attrs
        )
    return xds_filled

# ---------------------------
# Coordinate encoding
# ---------------------------
def encode_coordinates(xds: xr.DataArray, transformer) -> tuple[da.Array, da.Array, da.Array, da.Array]:
    """
    Encode raster coordinates as cosine and sine values for use in ML models.

    Parameters
    ----------
    xds : xr.DataArray
        Reference raster for dimensions and CRS.
    transformer : callable
        Function to transform coordinates to lon/lat.

    Returns
    -------
    tuple of dask.array
        Cosine and sine encoded latitude and longitude grids.
    """
    width, height = xds.rio.width, xds.rio.height

    top_left_corner = [xds.x[0], xds.y[0]]
    top_right_corner = [xds.x[-1], xds.y[0]]
    bottom_left_corner = [xds.x[0], xds.y[-1]]

    top_left_corner_lonlat = transformer.transform(*top_left_corner)
    top_right_corner_lonlat = transformer.transform(*top_right_corner)
    bottom_left_corner_lonlat = transformer.transform(*bottom_left_corner)

    lon_vec_deg = da.linspace(top_left_corner_lonlat[0],
                            top_right_corner_lonlat[0],
                            num=width, chunks=CHUNKSIZE)
    lat_vec_deg = da.linspace(top_left_corner_lonlat[1],
                            bottom_left_corner_lonlat[1],
                            num=height, chunks=CHUNKSIZE)

    lon_grid_deg, lat_grid_deg = da.meshgrid(lon_vec_deg, lat_vec_deg)

    lat_grid_cos_encoded = (np.cos(np.pi/90 * lat_grid_deg) + 1) / 2
    lat_grid_sin_encoded = (np.sin(np.pi/90 * lat_grid_deg) + 1) / 2
    lon_grid_cos_encoded = (np.cos(np.radians(lon_grid_deg)) + 1) / 2
    lon_grid_sin_encoded = (np.sin(np.radians(lon_grid_deg)) + 1) / 2

    return lat_grid_cos_encoded, lat_grid_sin_encoded, lon_grid_cos_encoded, lon_grid_sin_encoded

# ---------------------------
# Normalization
# ---------------------------
def normalize_data_xr(da: xr.DataArray, norm_values: dict, norm_strat: str, nodata_value=None) -> xr.DataArray:
    """
    Normalize an xarray.DataArray lazily (supports Dask).

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    norm_values : dict
        Dictionary containing normalization parameters (mean/std, percentiles, min/max).
    norm_strat : str
        Normalization strategy: "mean_std", "pct", or "min_max".
    nodata_value : optional
        Value to treat as no-data (set to zero after normalization).

    Returns
    -------
    xr.DataArray
        Normalized raster with all bands chunked together.
    """
    if norm_strat == "mean_std":
        mean, std = norm_values["mean"], norm_values["std"]
        normed = (da - mean) / std

    elif norm_strat == "pct":
        p1, p99 = norm_values["p1"], norm_values["p99"]
        normed = (da - p1) / (p99 - p1)
        normed = normed.clip(0, 1)

    elif norm_strat == "min_max":
        min_val, max_val = norm_values["min"], norm_values["max"]
        normed = (da - min_val) / (max_val - min_val)

    else:
        raise ValueError(f"Unknown normalization strategy: {norm_strat}")

    if nodata_value is not None:
        normed = xr.where(da == nodata_value, 0, normed)

    if 'band' in normed.dims:
        return normed.chunk({'y': CHUNKSIZE, 'x': CHUNKSIZE, 'band': -1})
    else:
        return normed.chunk({'y': CHUNKSIZE, 'x': CHUNKSIZE})

# ---------------------------
# Raster alignment
# ---------------------------
def load_and_align_raster_to_reference(path_to_raster: str, xds_ref: xr.DataArray, resampling=Resampling.nearest) -> xr.DataArray:
    """
    Load a raster and reproject/resample it to match a reference xarray.DataArray.

    Parameters
    ----------
    path_to_raster : str
        Path to the input raster.
    xds_ref : xr.DataArray
        Reference dataset defining CRS, transform, width, and height.
    resampling : rasterio.enums.Resampling, optional
        Resampling method. Default is nearest neighbor.

    Returns
    -------
    xr.DataArray
        Raster aligned to reference dataset with Dask chunking applied.
    """
    with rio.open(path_to_raster) as src:
        with WarpedVRT(
            src,
            crs=xds_ref.rio.crs,
            transform=xds_ref.rio.transform(),
            width=xds_ref.rio.width,
            height=xds_ref.rio.height,
            resampling=resampling,
        ) as vrt:
            raster_reproj = rioxarray.open_rasterio(
                vrt,
                masked=True,
                chunks={"y": CHUNKSIZE, "x": CHUNKSIZE, "band": 1}
            )
    return raster_reproj.transpose("y", "x", "band").chunk({'y': CHUNKSIZE, 'x': CHUNKSIZE, 'band': -1}).squeeze()