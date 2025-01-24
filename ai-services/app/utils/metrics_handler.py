import os
import json
from app.utils.logger import Logger


class MetricsHandler:
    """
    Handles saving and loading metrics for models.
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.logger = Logger.get_logger(__name__)
        os.makedirs(self.save_dir, exist_ok=True)

    def save_metrics(self, metrics: dict, filename: str = "metrics_history.json"):
        """
        Saves metrics to a JSON file.
        """
        metrics_path = os.path.join(self.save_dir, filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved at: {metrics_path}")

    def load_metrics(self, filename: str = "metrics_history.json") -> dict:
        """
        Loads metrics from a JSON file.
        """
        metrics_path = os.path.join(self.save_dir, filename)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
        self.logger.warning(f"Metrics file not found at: {metrics_path}")
        return {}
