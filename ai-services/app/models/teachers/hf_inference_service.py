import requests
from app.config.hyperparameters import HYPERPARAMETERS
from app.models.teachers.interfaces.inference_interface import InferenceInterface


class HuggingFaceInferenceService(InferenceInterface):
    """
    Service d'inférence utilisant l'API Hugging Face.
    """

    def __init__(self):
        """
        Initialise le service avec le modèle distant.
        Le modèle et l'URL sont récupérés depuis hyperparameters.py.
        """
        self.api_token = HYPERPARAMETERS["models"]["general_text_generation"]
        self.model_url = HYPERPARAMETERS["teachers"]["roles"]["basic_coding"]["model"]
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def predict(self, text: str) -> str:
        """
        Effectue une prédiction à l'aide de l'API Hugging Face.
        """
        payload = {"inputs": text}
        response = requests.post(self.model_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise ValueError(f"Error from Hugging Face API: {response.text}")

        return response.json()[0]["generated_text"]
