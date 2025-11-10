"""
Model initialization and inference utilities for AGB prediction.

This module provides:
- Functions to load model configurations and data statistics.
- Helpers to instantiate and restore PyTorch models for inference.
- A patch-based prediction workflow using raster I/O and progress tracking.

All configurations are read from the YAML file specified in the repository's configs directory.
"""

import pickle
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm

from src.utils.config import *
from src.models.helper import *

# Load common config
CFG = get_config("default.yaml")

def load_model_config():
    """
    Load the model configuration (.pkl) file.

    Returns
    -------
    dict
        Dictionary containing model hyperparameters and settings loaded from the pickled configuration file.
    """
    model_pkl_config_path = Path(CFG['paths']['model_weight_dir']) / CFG['model']['arch'] / (CFG['model']['model_id'] + '_cfg.pkl')
    with open(model_pkl_config_path, 'rb') as f:
        cfg = pickle.load(f)
    return cfg

def load_model_data_statistics():
    """
    Load precomputed data statistics (.pkl) used for normalization or scaling.

    Returns
    -------
    dict
        Dictionary of input data statistics (e.g., means, standard deviations).
    """
    input_pkl_data_stat_path = Path(CFG['paths']['input_data_stats'])
    with open(input_pkl_data_stat_path, 'rb') as f:
        data_stats = pickle.load(f)
    return data_stats


def load_inference_model(arch, model_id, model_cfg, device):
    """
    Load precomputed data statistics (.pkl) used for normalization or scaling.

    Returns
    -------
    dict
        Dictionary of input data statistics (e.g., means, standard deviations).
    """

    net = Net(
        model_name=model_cfg['arch'],
        in_features=model_cfg['in_features'],
        num_outputs=model_cfg['num_outputs'],
        channel_dims=model_cfg['channel_dims'],
        max_pool=model_cfg['max_pool'],
        downsample=None,
        leaky_relu=model_cfg['leaky_relu'],
        patch_size=model_cfg['patch_size']
    )

    model = Model(
        net,
        lr=model_cfg['lr'],
        step_size=model_cfg['step_size'],
        gamma=model_cfg['gamma'],
        patch_size=model_cfg['patch_size'],
        downsample=model_cfg['downsample'],
        loss_fn=model_cfg['loss_fn']
    )

    ckpt_path = get_pretrained_weights_dir_path() / arch / f"{model_id}_best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")
    state_dict = torch.load(ckpt_path, map_location=device)['state_dict']

    # # Fix potential key mismatches
    # if arch == 'nico':
    #     state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.model.eval()

    return model.model

def predict_AGB(input_cube, inf_model, device, output_path, crs, transform,
                mask=None, return_array=False):
    """
    Run patch-based AGBD prediction and export results as a GeoTIFF.

    Parameters
    ----------
    input_cube : np.ndarray or dask.array.core.Array
        Input raster cube with shape (height, width, channels).
    inf_model : torch.nn.Module
        Trained inference model (in eval mode).
    device : torch.device
        Torch device for computation.
    output_path : str or Path
        Destination path for the output GeoTIFF file.
    crs : rasterio.crs.CRS or str
        Coordinate reference system of the output.
    transform : affine.Affine
        GeoTransform defining pixel coordinates.
    mask : np.ndarray, optional
        Boolean mask (True for no-data areas). Masked predictions will be set to NaN.
    return_array : bool, optional
        If True, also return the in-memory prediction array.

    Returns
    -------
    np.ndarray or None
        2D array of predicted AGBD values if `return_array=True`, otherwise None.

    Notes
    -----
    - The prediction is performed patch by patch to avoid memory overflow.
    - Patches overlap by a configurable size (`CFG['prediction']['overlap_size']`).
    - Output GeoTIFF is written incrementally using rasterio’s windowed writes.
    - A progress bar via `tqdm` tracks inference progress.
    """

    # Extract tile size (ty, tx), patch size (py, px), and overlap size (oy, ox)
    ty, tx = input_cube.shape[:2]
    py, px = CFG['prediction']['patch_size']
    oy, ox = CFG['prediction']['overlap_size']
    # Compute step sizes
    sy, sx = py - oy, px - ox
    # Calculate number of patches needed in each dimension
    # For x dimension, we need px + (nx - 1) * sx >= tx to cover the full width, so nx = ceil((tx - px) / sx + 1)
    # For y dimension, we need py + (ny - 1) * sy >= ty to cover the full height, so ny = ceil((ty - py) / sy + 1)
    ny = int(np.ceil((ty - py) / sy + 1))
    nx = int(np.ceil((tx - px) / sx + 1))
    total_patches = nx * ny
    # Initialize output array with NaNs
    if return_array:
        predictions = np.full((ty, tx), np.nan, dtype=np.float32)
    # Build rasterio profile
    profile = {
        "driver": "GTiff",
        "height": ty,
        "width": tx,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512
    }
    with rio.open(output_path, "w", **profile) as dst:
        patch_idx = 0
        # Loop over patches 
        with tqdm(total=total_patches, desc="Predicting patches", unit="patch") as pbar:
            for iy in range(ny):
                # Calculate patch start index along y-dimension
                if iy == ny - 1:
                    py_start = ty - py  # last patch aligns with the bottom edge, leading to a larger overlap
                else:
                    py_start = iy * sy
                # Calculate the half-overlap area to not take into account
                hoy_top = int(oy / 2)
                hoy_bottom = int(oy / 2)
                if iy == 0:
                    hoy_top = 0
                elif iy == ny - 2:
                    last_y_overlap =  (ny-2)*sy+py - (ty-py) + 1 # End of second to last to start of last patch + 1: (ny-2)*sy+py - (ty-py)
                    hoy_bottom = last_y_overlap // 2
                elif iy == ny - 1:
                    last_y_overlap = (ny-2)*sy+py - (ty-py) + 1
                    hoy_top = last_y_overlap // 2
                    hoy_bottom = 0
                for ix in range(nx):
                    # Calculate patch start index along x-dimension
                    if ix == nx - 1:
                        px_start = tx - px  # last patch aligns with the right edge, leading to a larger overlap
                    else:
                        px_start = ix * sx
                    # Calculate the half-overlap area to not take into account
                    hox_left = int(ox / 2)
                    hox_right = int(ox / 2)
                    if ix == 0:
                        hox_left = 0
                    elif ix == nx - 2:
                        last_x_overlap =  (nx-2)*sx+px - (tx-px) + 1 # End of second to last to start of last patch + 1: (nx-2)*sx+px - (tx-px)
                        hox_right = last_x_overlap // 2
                    elif ix == nx - 1:
                        last_x_overlap = (nx-2)*sx+px - (tx-px) + 1
                        hox_left = last_x_overlap // 2
                        hox_right = 0
                    # Extract patch from input cube
                    patch = input_cube[py_start:py_start + py, px_start:px_start + px, :]
                    # Predict on the patch
                    patch_pred = predict_patch(inf_model, patch, device)
                    # Apply mask
                    if mask is not None:
                        # Extract the corresponding mask patch
                        mask_patch = mask[py_start:py_start + py, px_start:px_start + px]
                        # Apply mask to the predicted patch
                        patch_pred = np.where(mask_patch, np.nan, patch_pred)
                    # Update predictions array
                    if return_array:
                        predictions[py_start+hoy_top:py_start+py-hoy_bottom, px_start+hox_left:px_start+px-hox_right] = patch_pred[hoy_top:py-hoy_bottom, hox_left:px-hox_right]
                    # Write to file
                    dst.write(
                        patch_pred[hoy_top:py-hoy_bottom, hox_left:px-hox_right],
                        1,
                        window=Window(
                            px_start + hox_left,
                            py_start + hoy_top,
                            (px - hox_left - hox_right),
                            (py - hoy_top - hoy_bottom)
                        )
                    )
                    patch_idx += 1
                    pbar.update(1)
    if return_array:
        return predictions
    else:
        return None