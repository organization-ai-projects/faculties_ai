# app/utils/metrics/json_metrics_storage.py
import json
import os
from typing import Any, Dict

from app.utils.metrics.interfaces.metrics_storage_interface import (
    MetricsStorageInterface,
)


class JsonMetricsStorage(MetricsStorageInterface):
    """
    Implémentation du stockage des métriques en JSON.
    """

    def save(self, data: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}
