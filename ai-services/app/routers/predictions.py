from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator
from app.models.teachers.interfaces.inference_interface import InferenceInterface
from app.main import get_inference_service
from app.utils.logger import Logger
import time

# Initialisation du routeur
router = APIRouter()
logger = Logger.setup_logger(__name__)


# Modèle de requête pour la prédiction
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        title="Input Text",
        description="Texte pour lequel effectuer une prédiction",
        min_length=1,
        max_length=5000,
    )
    max_length: int = Field(
        100, title="Max Length", description="Longueur maximale de la réponse générée."
    )
    temperature: float = Field(
        1.0, title="Temperature", description="Niveau de créativité pour la génération."
    )

    @validator("temperature")
    def validate_temperature(cls, value):
        if not 0 <= value <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        return value


@router.post("/")
def predict(
    request: PredictionRequest,
    inference_service: InferenceInterface = Depends(get_inference_service),
):
    """
    Endpoint pour effectuer une prédiction sur un texte donné.
    Utilise le service d'inférence injecté (local ou distant).
    """
    try:
        start_time = time.time()
        logger.info(
            f"Prediction request received: {request.text[:50]}... (truncated for log)"
        )

        # Appeler le service d'inférence avec les paramètres
        result = inference_service.predict(
            text=request.text,
            max_length=request.max_length,
            temperature=request.temperature,
        )

        execution_time = round(time.time() - start_time, 2)
        logger.info(f"Prediction result: {result}")
        logger.info(f"Prediction completed in {execution_time} seconds")

        return {
            "input": request.text,
            "prediction": result,
            "model_metadata": {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "execution_time": f"{execution_time}s",
            },
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
