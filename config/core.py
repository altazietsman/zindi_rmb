from pathlib import Path
import calendar
from strictyaml import YAML, load

# Project Directories
PACKAGE_ROOT = Path().cwd().resolve()
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


def find_config_file():
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None):
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None):
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    return parsed_config


config = create_and_validate_config()
