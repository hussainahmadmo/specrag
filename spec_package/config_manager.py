# config_manager.py
import yaml

class ConfigManager:
    _config = None

    @staticmethod
    def load_config(config_path="../configs/central.yaml"):
        """Load the configuration file."""
        if ConfigManager._config is None:
            with open(config_path, "r") as f:
                ConfigManager._config = yaml.safe_load(f)
        return ConfigManager._config

    @staticmethod
    def get_config():
        """Get the loaded configuration."""
        if ConfigManager._config is None:
            raise ValueError("Configuration has not been loaded. Call 'load_config' first.")
        return ConfigManager._config