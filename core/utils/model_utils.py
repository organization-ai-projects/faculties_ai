# core/utils/model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelUtils:
    """
    Classe dédiée à la gestion des modèles.
    Permet de charger un modèle et un tokenizer.
    """

    @staticmethod
    def load_model_and_tokenizer(model_name, hf_token):
        """
        Charge un modèle et son tokenizer depuis Hugging Face.
        """
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        return model, tokenizer
