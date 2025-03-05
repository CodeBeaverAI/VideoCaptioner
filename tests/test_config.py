import os
import logging
from pathlib import Path
import pytest

from app import config

def test_constants():
    """Test configuration constants."""
    assert config.VERSION == "v1.3.2"
    assert config.YEAR == 2025
    assert config.APP_NAME == "VideoCaptioner"
    assert config.AUTHOR == "Weifeng"
    assert config.HELP_URL == "https://github.com/WEIFENG2333/VideoCaptioner"
    assert config.GITHUB_REPO_URL == "https://github.com/WEIFENG2333/VideoCaptioner"
    assert config.RELEASE_URL == "https://github.com/WEIFENG2333/VideoCaptioner/releases/latest"
    assert config.FEEDBACK_URL == "https://github.com/WEIFENG2333/VideoCaptioner/issues"

def test_paths():
    """Test that all defined paths are instances of pathlib.Path."""
    paths = [
        config.ROOT_PATH,
        config.RESOURCE_PATH,
        config.APPDATA_PATH,
        config.WORK_PATH,
        config.BIN_PATH,
        config.ASSETS_PATH,
        config.SUBTITLE_STYLE_PATH,
        config.LOG_PATH,
        config.SETTINGS_PATH,
        config.CACHE_PATH,
        config.MODEL_PATH,
        config.FASER_WHISPER_PATH,
    ]
    for p in paths:
        assert isinstance(p, Path)

def test_environment_variables():
    """Test that environment variables were set as expected.
    The PATH variable should have FASER_WHISPER_PATH at index 0 and BIN_PATH at index 1.
    PYTHON_VLC_MODULE_PATH should equal BIN_PATH/vlc.
    """
    current_path = os.environ.get("PATH", "")
    faser_whisper_str = str(config.FASER_WHISPER_PATH)
    bin_path_str = str(config.BIN_PATH)
    splits = current_path.split(os.pathsep)
    assert splits[0] == faser_whisper_str
    assert splits[1] == bin_path_str

    expected_vlc_path = str(config.BIN_PATH / "vlc")
    assert os.environ.get("PYTHON_VLC_MODULE_PATH") == expected_vlc_path

def test_directories_creation():
    """Test that the CACHE, LOG, WORK and MODEL directories exist as created by the config."""
    for p in [config.CACHE_PATH, config.LOG_PATH, config.WORK_PATH, config.MODEL_PATH]:
        assert p.exists()
        assert p.is_dir()
def test_logging_configuration():
    """Test logging configuration parameters."""
    import logging
    assert config.LOG_LEVEL == logging.INFO
    expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert config.LOG_FORMAT == expected_format

def test_relative_path_structure():
    """Test that relative path structure is set by using config.ROOT_PATH as base."""
    # Verify that RESOURCE_PATH, APPDATA_PATH, WORK_PATH, etc. are computed correctly
    assert config.RESOURCE_PATH == config.ROOT_PATH.parent / "resource"
    assert config.APPDATA_PATH == config.ROOT_PATH.parent / "AppData"
    assert config.WORK_PATH == config.ROOT_PATH.parent / "work-dir"
    assert config.BIN_PATH == config.RESOURCE_PATH / "bin"
    assert config.ASSETS_PATH == config.RESOURCE_PATH / "assets"
    assert config.SUBTITLE_STYLE_PATH == config.RESOURCE_PATH / "subtitle_style"
def test_reload_config_creates_directories():
    """Test that reloading the config module recreates missing directories without error."""
    import importlib
    import shutil
    from app import config as config_module

    # Remove CACHE_PATH if it exists to simulate a missing directory scenario.
    if config_module.CACHE_PATH.exists():
        shutil.rmtree(config_module.CACHE_PATH)

    # Reload the module; the missing directory should be recreated.
    importlib.reload(config_module)
    assert config_module.CACHE_PATH.exists() and config_module.CACHE_PATH.is_dir()

def test_parent_directories_relationship():
    """Test that the configuration directories have the correct parent-child relationships."""
    from app import config as config_module
    # RESOURCE_PATH and APPDATA_PATH should be subdirectories of ROOT_PATH.parent.
    assert config_module.RESOURCE_PATH.parent == config_module.ROOT_PATH.parent
    assert config_module.APPDATA_PATH.parent == config_module.ROOT_PATH.parent
    # Verify that subdirectories of RESOURCE_PATH are computed correctly.
    assert config_module.BIN_PATH.parent == config_module.RESOURCE_PATH
    assert config_module.ASSETS_PATH.parent == config_module.RESOURCE_PATH
    assert config_module.SUBTITLE_STYLE_PATH.parent == config_module.RESOURCE_PATH
def test_multiple_reload_increments_env_path():
    """Test that reloading the config module multiple times increments environment PATH with duplicates accordingly."""
    import os
    from importlib import reload
    from app import config as config_module
    original_split = os.environ["PATH"].split(os.pathsep)
    reload(config_module)
    split_after_one = os.environ["PATH"].split(os.pathsep)
    # On each reload, two paths (FASER_WHISPER_PATH and BIN_PATH) are prepended.
    assert len(split_after_one) == len(original_split) + 2
    assert split_after_one[0] == str(config_module.FASER_WHISPER_PATH)
    assert split_after_one[1] == str(config_module.BIN_PATH)
    reload(config_module)
    split_after_two = os.environ["PATH"].split(os.pathsep)
    assert len(split_after_two) == len(split_after_one) + 2
    assert split_after_two[0] == str(config_module.FASER_WHISPER_PATH)
    assert split_after_two[1] == str(config_module.BIN_PATH)

def test_settings_path_not_dir():
    """Test that the settings path is not created as a directory, since it represents a file path candidate."""
    from app import config
    # If the settings file exists, it should not be a directory.
    if config.SETTINGS_PATH.exists():
        assert not config.SETTINGS_PATH.is_dir()