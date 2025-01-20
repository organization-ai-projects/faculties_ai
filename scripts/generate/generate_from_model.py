from transformers import AutoModelForCausalLM, AutoTokenizer
from core import config
import logging

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/generate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_from_model(model_name, prompt, max_length=50, hf_token=None):
    """Génère du texte à partir d'un modèle donné."""
    logger.info(f"Chargement du modèle {model_name} pour la génération...")
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    # Tokenisation du prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Génération de texte
    logger.info("Génération en cours...")
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Texte généré : {generated_text}")
    return generated_text

def main():
    try:
        prompt = "Explique les principes de SOLID en programmation."
        hf_token = config.HF_TOKEN

        for model_name in config.HYPERPARAMETERS["models"]:
            print(f"\n=== Génération avec {model_name} ===")
            generated_text = generate_from_model(model_name, prompt, hf_token=hf_token)
            print(f"Résultat ({model_name}) : {generated_text}\n")
    except Exception as e:
        logger.error(f"Erreur lors de la génération : {e}")

if __name__ == "__main__":
    main()
