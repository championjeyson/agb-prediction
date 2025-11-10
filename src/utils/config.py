"""
Configuration utilities for the AGB prediction pipeline.

Provides:
- Config singleton class to load YAML once
- Convenience access function
- Helper to extract chunk size
"""

from pathlib import Path
import yaml
from typing import Optional, Any, Dict

class Config:
    """
    Singleton-like loader to read a YAML configuration file once.

    This class ensures that the configuration is loaded a single time per Python session
    and can be accessed across modules without repeatedly reading the YAML file.

    Usage
    -----
    cfg = Config.load("defaults.yaml")
    """

    _cfg: Optional[Dict[str, Any]] = None

    @classmethod
    def load(cls, filename="default.yaml", force_reload=False):
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        filename : str, optional
            Name of the YAML configuration file located in the `configs/` directory.
            Defaults to "default.yaml".
        force_reload: bool, optional
            Only load once per session so this argument allows to force that and reload anyway.

        Returns
        -------
        dict
            The loaded configuration as a dictionary.

        Notes
        -----
        Only loads the file once. Subsequent calls return the cached configuration.
        """
        if cls._cfg is None or force_reload:
            config_path = (
                Path(__file__).resolve().parent.parent.parent / "configs" / filename
            )
            with open(config_path) as f:
                cls._cfg = yaml.safe_load(f)
        return cls._cfg

def get_config(filename="default.yaml"):
    """
    Convenience function to access the global configuration.

    Parameters
    ----------
    filename : str, optional
        Name of the YAML configuration file. Defaults to "default.yaml".

    Returns
    -------
    dict
        Loaded configuration dictionary.
    """
    return Config.load(filename)

def get_chunk_size(cfg):
    """
    Retrieve the chunk size from the configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (from `Config.load` or `get_config`).

    Returns
    -------
    int
        Chunk size specified in `cfg['tiling']['chunk_size']`.

    Raises
    ------
    ValueError
        If `chunk_size` is missing in the configuration.
    """
    try:
        return cfg['tiling']['chunk_size']
    except KeyError:
        raise KeyError("chunk_size missing in the configuration file.")