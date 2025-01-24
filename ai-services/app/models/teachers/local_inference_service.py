from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config.hyperparameters import HYPERPARAMETERS
from app.models.teachers.interfaces.inference_interface import InferenceInterface


class LocalInferenceService(InferenceInterface):
    """
    Service d'inférence utilisant un modèle local.
    """

    def __init__(self):
        """
        Initialise le modèle et le tokenizer en local.
        Le modèle est récupéré depuis hyperparameters.py.
        """
        model_name = HYPERPARAMETERS["teachers"]["roles"]["basic_coding"]["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def predict(self, text: str) -> str:
        """
        Effectue une prédiction à l'aide du modèle local.
        """
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
