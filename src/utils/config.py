from pathlib import Path
import yaml

class Config:
    """
    Singleton-like loader to read a YAML config once and share across modules.
    """
    _cfg = None

    @classmethod
    def load(cls, filename="default.yaml"):
        if cls._cfg is None:
            # Resolve path from repo root
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / filename
            with open(config_path) as f:
                cls._cfg = yaml.safe_load(f)
        return cls._cfg

# Helper function (optional)
def get_config(filename="default.yaml"):
    return Config.load(filename)

def get_chunk_size(cfg):
    try:
        return cfg['tiling']['chunk_size']
    except KeyError:
        raise ValueError("chunk_size missing in the configuration file.")