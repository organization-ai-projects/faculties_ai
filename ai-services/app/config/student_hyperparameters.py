import os
from dotenv import load_dotenv

# Charger les variables d'environnement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../../.env"))

# Création du répertoire des étudiants
os.makedirs(
    os.getenv("STUDENT_SAVE_DIR", os.path.join(BASE_DIR, "../models/students/")),
    exist_ok=True,
)

STUDENT_HYPERPARAMETERS = {
    "learning_rate": float(os.getenv("STUDENT_LEARNING_RATE", 3e-5)),
    "batch_size": int(os.getenv("STUDENT_BATCH_SIZE", 8)),
    "epochs": int(os.getenv("STUDENT_EPOCHS", 5)),
    "save_dir": os.getenv(
        "STUDENT_SAVE_DIR", os.path.join(BASE_DIR, "../models/students/")
    ),
    "base_model": {
        "type": os.getenv("STUDENT_BASE_MODEL", "simple_lstm"),
        "input_dim": int(os.getenv("STUDENT_INPUT_DIM", 128)),
        "hidden_dim": int(os.getenv("STUDENT_HIDDEN_DIM", 256)),
        "output_dim": int(os.getenv("STUDENT_OUTPUT_DIM", 128)),
    },
    "model_config": {
        "vocab_size": int(os.getenv("STUDENT_VOCAB_SIZE", 50257)),
        "hidden_size": int(os.getenv("STUDENT_HIDDEN_SIZE", 768)),
        "num_layers": int(os.getenv("STUDENT_NUM_LAYERS", 12)),
        "num_heads": int(os.getenv("STUDENT_NUM_HEADS", 12)),
        "max_seq_len": int(os.getenv("STUDENT_MAX_SEQ_LEN", 512)),
        "dropout": float(os.getenv("STUDENT_DROPOUT", 0.1)),
        "ff_multiplier": int(os.getenv("STUDENT_FF_MULTIPLIER", 4)),
    },
}
