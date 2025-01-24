import os
from fastapi import FastAPI, Depends
from app.routers import predictions, training, evaluation
from app.models.teachers.training_service import TrainingService
from app.utils.alert_system import AlertSystem
from app.utils.logger import Logger
from dotenv import load_dotenv

# Déterminer l'environnement (par défaut : dev)
APP_ENV = os.getenv("APP_ENV", "dev")

# Construire le chemin du fichier .env pour l'environnement actuel
dotenv_path = os.path.join(os.path.dirname(__file__), f"../.env.{APP_ENV}")

# Charger les variables d'environnement
load_dotenv(dotenv_path=dotenv_path)

# Initialiser le logger centralisé
logger = Logger.setup_logger(__name__)

# Vérification des configurations critiques
REQUIRED_ENV_VARS = ["DATASET_PATH", "TEACHER_SAVE_DIR", "STUDENT_SAVE_DIR"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise ValueError(f"Critical environment variable {var} is missing.")

# Chemins critiques
DATASET_PATH = os.getenv("DATASET_PATH")
TEACHER_SAVE_DIR = os.getenv("TEACHER_SAVE_DIR")
STUDENT_SAVE_DIR = os.getenv("STUDENT_SAVE_DIR")


# Fonction d'injection de dépendance : Système d'alerte
def get_alert_system() -> AlertSystem:
    """
    Retourne une instance d'AlertSystem pour gérer les alertes.
    """
    alert_folder = os.getenv("ALERT_FOLDER", "alerts")  # Chemin par défaut : "alerts"
    return AlertSystem(alert_folder=alert_folder)


# Fonction d'injection de dépendance : Service d'entraînement
def get_training_service(
    alert_system: AlertSystem = Depends(get_alert_system),
) -> TrainingService:
    """
    Retourne une instance de TrainingService configurée avec des dépendances injectées.
    """
    learning_rate = float(os.getenv("TEACHER_LEARNING_RATE", 5e-5))
    batch_size = int(os.getenv("TEACHER_BATCH_SIZE", 4))
    epochs = int(os.getenv("TEACHER_EPOCHS", 3))
    save_dir = TEACHER_SAVE_DIR

    return TrainingService(
        alert_system=alert_system,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        save_dir=save_dir,
    )


# Initialisation de l'application FastAPI
app = FastAPI(
    title="AI Services API",
    description="API pour gérer les prédictions, l'entraînement et l'évaluation des modèles IA.",
    version="1.0.0",
)

# Routes principales
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(training.router, prefix="/training", tags=["Training"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])


@app.get("/")
def root():
    """
    Endpoint principal pour vérifier que l'API fonctionne.
    """
    logger.info("API root accessed.")
    return {"message": "Welcome to the AI Services API"}


@app.get("/health/")
def health_check():
    """
    Endpoint de santé pour vérifier que le service est opérationnel.
    """
    logger.info("Health check accessed.")
    return {"status": "OK"}


@app.get("/config/")
def get_config():
    """
    Endpoint pour afficher les configurations actuelles (uniquement pour le debug).
    """
    logger.info("Configurations fetched.")
    return {
        "environment": APP_ENV,
        "dataset_path": DATASET_PATH,
        "teacher_save_dir": TEACHER_SAVE_DIR,
        "student_save_dir": STUDENT_SAVE_DIR,
    }
