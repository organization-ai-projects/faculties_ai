from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.models.teachers.training_service import TrainingService
from app.utils.logger import Logger

router = APIRouter()
logger = Logger.setup_logger(__name__)


def get_training_service() -> TrainingService:
    """
    Injection de dépendance pour récupérer une instance de TrainingService.
    """
    return TrainingService()


# Modèle pour les requêtes d'entraînement
class TrainingRequest(BaseModel):
    model_name: str
    epochs: int
    learning_rate: float


@router.post("/train")
def train_model(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_training_service),
):
    """
    Endpoint pour lancer l'entraînement d'un modèle IA.
    """
    try:
        logger.info(f"Lancement de l'entraînement pour le modèle {request.model_name}")
        message = training_service.train(
            model_name=request.model_name,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
        )
        logger.info(f"Entraînement terminé pour le modèle {request.model_name}")
        return {"message": message}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/role/{role_name}")
def train_role(
    role_name: str,
    epochs: int = None,
    learning_rate: float = None,
    training_service: TrainingService = Depends(get_training_service),
):
    """
    Endpoint pour lancer l'entraînement d'un rôle spécifique.
    """
    try:
        logger.info(f"Lancement de l'entraînement pour le rôle {role_name}")
        message = training_service.train_specific_role(
            role_name=role_name, epochs=epochs, learning_rate=learning_rate
        )
        logger.info(f"Entraînement terminé pour le rôle {role_name}")
        return {"message": message}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du rôle {role_name} : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/all_roles")
def train_all_roles(training_service: TrainingService = Depends(get_training_service)):
    """
    Endpoint pour lancer l'entraînement de tous les rôles des professeurs IA.
    """
    try:
        logger.info("Lancement de l'entraînement pour tous les rôles")
        message = training_service.train_all_roles()
        logger.info("Entraînement terminé pour tous les rôles")
        return {"message": message}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement de tous les rôles : {e}")
        raise HTTPException(status_code=500, detail=str(e))
