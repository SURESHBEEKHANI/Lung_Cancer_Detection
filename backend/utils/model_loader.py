import os
import yaml
from ultralytics import YOLO

class ModelLoader:
    def __init__(self, config_path: str = None):
        # Default config path
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))  # backend/
            config_path = os.path.join(base_dir, "config", "config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Check if 'models' key exists
        self.models = self.config.get("models")
        if self.models is None:
            raise ValueError("Config file missing 'models' section.")

    def load_model(self, key: str):
        # Check if the model key exists
        cfg = self.models.get(key)
        if not cfg:
            raise ValueError(f"Model '{key}' not defined in config.yaml")

        provider = cfg.get("provider")
        if not provider:
            raise ValueError(f"Provider not specified for model '{key}'")

        # Load local YOLO model
        if provider == "local":
            weights = cfg.get("weights")
            if not weights or not os.path.exists(weights):
                # List available weight files in the weights directory
                base_dir = os.path.dirname(os.path.dirname(__file__))  # backend/
                weights_dir = os.path.join(base_dir, "weights")
                available = []
                if os.path.exists(weights_dir):
                    available = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]

                raise FileNotFoundError(
                    f"Weights not found for '{key}': {weights}\n"
                    f"Available weights: {available if available else 'None found'}"
                )

            print(f"[INFO] Loading local model '{key}' from {weights}")
            return YOLO(weights)

        # Placeholder for external LLM API models
        elif provider == "groq":
            model_name = cfg.get("model_name")
            if not model_name:
                raise ValueError(f"Groq model_name not specified for '{key}'")
            print(f"[INFO] Using Groq LLM model: {model_name}")
            return cfg  # return config dict for now

        else:
            raise ValueError(f"Unsupported provider '{provider}' for model '{key}'")
