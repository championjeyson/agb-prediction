import pytest
import numpy as np
import xarray as xr
import dask.array as da
from pathlib import Path
from types import SimpleNamespace
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

import src.imagery_processing.helper as helper

# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def tmp_input_dir(tmp_path):
    d = tmp_path / "input_images"
    d.mkdir()
    (d / "S2_B04.tif").touch()
    (d / "SAR_PALSAR2.tif").touch()
    (d / "AW3D30.tif").touch()
    (d / "CGLS_LC100.tif").touch()
    return d


@pytest.fixture(autouse=True)
def patch_config(monkeypatch, tmp_input_dir):
    """Patch get_config to always return our temporary folder."""
    monkeypatch.setattr(helper, "CFG", {"paths": {"input_image_dir": str(tmp_input_dir)}})
    yield


# ----------------------------------------------------------
# Path retrieval
# ----------------------------------------------------------

def test_get_input_image_dir_path_ok(tmp_input_dir):
    path = helper.get_input_image_dir_path()
    assert path.exists()
    assert "input_images" in str(path)


def test_get_input_image_dir_path_missing(monkeypatch):
    monkeypatch.setattr(helper, "CFG", {"paths": {"input_image_dir": None}})
    with pytest.raises(ValueError):
        helper.get_input_image_dir_path()


def test_get_s2_band_path_found(tmp_input_dir):
    path = helper.get_s2_band_path("B04")
    assert path.exists()
    assert path.name == "S2_B04.tif"


def test_get_s2_band_path_not_found(tmp_input_dir):
    with pytest.raises(FileNotFoundError):
        helper.get_s2_band_path("B08")


# ----------------------------------------------------------
# NaN filling
# ----------------------------------------------------------

def test_fill_na_chunkwise_2d():
    arr = np.array([[1, np.nan], [3, 4]])
    out = helper.fill_na_chunkwise(arr)
    assert not np.isnan(out).any()
    assert out.shape == arr.shape


def test_fill_na_chunkwise_3d():
    arr = np.dstack([np.array([[1, np.nan], [2, 3]]), np.array([[np.nan, 2], [3, 4]])])
    print(arr, arr.shape)
    out = helper.fill_na_chunkwise(arr)
    assert out.shape == arr.shape
    assert not np.isnan(out).any()


def test_fill_na_xarray_success():
    """Test fill_na with a small array and valid overlap."""
    data = np.array([[np.nan, 2], [3, 4]], dtype=float)
    xds = xr.DataArray(data, dims=("y", "x"))
    xds = xds.chunk({"y": 2, "x": 2})  # set chunking

    # Use overlap smaller than chunk size
    result = helper.fill_na(xds, overlap=1)
    filled = result.compute()
    assert not np.isnan(filled).any()
    assert filled.shape == data.shape

def test_fill_na_xarray_overlap_too_large():
    """Test fill_na raises ValueError if overlap > chunk size."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    xds = xr.DataArray(data, dims=("y", "x"))
    xds = xds.chunk({"y": 2, "x": 2})

    with pytest.raises(ValueError, match="Requested overlap.*larger than the chunk size"):
        _ = helper.fill_na(xds, overlap=3)  # intentionally too large


# ----------------------------------------------------------
# Coordinate encoding
# ----------------------------------------------------------

import numpy as np
import dask.array as da
from types import SimpleNamespace
from src.imagery_processing import helper

def test_encode_coordinates_sanity():
    # Dummy xarray-like object
    class Dummy:
        def __init__(self):
            self.rio = SimpleNamespace(width=4, height=3)
            self.x = np.linspace(0, 3, 4)
            self.y = np.linspace(0, 2, 3)

    xds = Dummy()
    # Simple transformer: just add constants
    transformer = SimpleNamespace(transform=lambda x, y: (x + 10, y + 20))

    lat_cos, lat_sin, lon_cos, lon_sin = helper.encode_coordinates(xds, transformer)

    # 1. Check types
    for arr in [lat_cos, lat_sin, lon_cos, lon_sin]:
        assert isinstance(arr, da.Array)

    # 2. Check shapes
    assert lat_cos.shape == (3, 4)
    assert lat_sin.shape == (3, 4)
    assert lon_cos.shape == (3, 4)
    assert lon_sin.shape == (3, 4)

    # 3. Check value ranges
    for arr in [lat_cos, lat_sin, lon_cos, lon_sin]:
        arr_computed = arr.compute()
        assert np.all(arr_computed >= 0) and np.all(arr_computed <= 1)

    # 4. Check first corner value matches expected simple calculation
    expected_lat_cos = (np.cos(np.pi/90 * (20)) + 1)/2
    expected_lat_sin = (np.sin(np.pi/90 * (20)) + 1)/2
    expected_lon_cos = (np.cos(np.radians(10)) + 1)/2
    expected_lon_sin = (np.sin(np.radians(10)) + 1)/2

    assert np.isclose(lat_cos[0,0].compute(), expected_lat_cos)
    assert np.isclose(lat_sin[0,0].compute(), expected_lat_sin)
    assert np.isclose(lon_cos[0,0].compute(), expected_lon_cos)
    assert np.isclose(lon_sin[0,0].compute(), expected_lon_sin)


# ----------------------------------------------------------
# Normalization
# ----------------------------------------------------------

@pytest.mark.parametrize("strategy", ["mean_std", "pct", "min_max"])
def test_normalize_data_xr(strategy):
    arr = xr.DataArray(np.array([[1, 2], [3, 4]]), dims=["y", "x"])
    norm_values = {"mean": 2, "std": 1, "p1": 1, "p99": 4, "min": 1, "max": 4}
    result = helper.normalize_data_xr(arr, norm_values, strategy)

    # Check type and shape
    assert isinstance(result, xr.DataArray)
    assert result.shape == arr.shape

    # For pct and min_max, check values are in [0, 1]
    if strategy in ["pct", "min_max"]:
        vals = result.data.compute() if hasattr(result.data, "compute") else result.data
        assert np.all(vals >= 0) and np.all(vals <= 1)


def test_normalize_data_xr_invalid():
    arr = xr.DataArray(np.ones((2, 2)), dims=["y", "x"])
    with pytest.raises(ValueError):
        helper.normalize_data_xr(arr, {}, "unknown")


# ----------------------------------------------------------
# Raster alignment
# ----------------------------------------------------------

def test_load_and_align_raster_to_reference_real():
    """Test alignment of a small in-memory raster to a reference."""

    # Create a tiny reference xarray
    ref_data = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims=["y", "x"]
    )
    ref_data.rio.write_crs("EPSG:4326", inplace=True)
    ref_data.rio.write_transform(rasterio.transform.from_origin(0, 2, 1, 1), inplace=True)

    # Create a small in-memory raster (3x3) to align
    data = np.arange(9, dtype=np.float32).reshape((3, 3))
    transform = rasterio.transform.from_origin(0, 3, 1, 1)  # slightly different origin

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=3,
            width=3,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dataset:
            dataset.write(data, 1)
        
        # The path to MemoryFile can be used via .name for rasterio.open
        with memfile.open() as dataset:
            # Align to reference
            out = helper.load_and_align_raster_to_reference(dataset.name, ref_data, resampling=Resampling.nearest)

    # Checks
    assert isinstance(out, xr.DataArray)
    assert out.shape == ref_data.shape  # aligned shape
    assert out.rio.crs == ref_data.rio.crs
    # Values check: nearest resampling should map closest cells
    assert np.all(np.isfinite(out.values))  # no NaNs