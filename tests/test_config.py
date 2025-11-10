# tests/test_config.py
import yaml
from pathlib import Path
import pytest
from src.utils.config import Config, get_config, get_chunk_size

@pytest.fixture
def dummy_config_file():
    """
    Create a temporary YAML config file inside the repo's configs/ folder.
    Deleted automatically after the test.
    """
    repo_root = Path(__file__).resolve().parent.parent
    configs_dir = repo_root / "configs"
    configs_dir.mkdir(exist_ok=True)

    cfg_path = configs_dir / "test_for_pytest.yaml"
    cfg_data = {"tiling": {"chunk_size": 199}} # A value that will probably not be use otherwise

    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_data, f)

    yield cfg_path

    if cfg_path.exists():
        cfg_path.unlink()


def test_config_load_and_cache(dummy_config_file):
    cfg = Config.load(dummy_config_file.name, force_reload=True)
    assert "tiling" in cfg
    assert cfg["tiling"]["chunk_size"] == 199 
    # Cached instance test
    cfg2 = get_config(dummy_config_file.name)
    assert cfg is cfg2

def test_get_chunk_size_success():
    cfg = {"tiling": {"chunk_size": 256}}
    assert get_chunk_size(cfg) == 256

def test_get_chunk_size_missing():
    with pytest.raises(KeyError):
        get_chunk_size({})
