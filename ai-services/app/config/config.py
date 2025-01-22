import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Vérification du token Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("The Hugging Face token (HF_TOKEN) is missing. Check your .env file.")
else:
    print("Hugging Face token successfully loaded!")
