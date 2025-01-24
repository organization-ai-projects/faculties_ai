from fastapi import APIRouter, Depends
from app.models.students.student_trainer import StudentTrainer
from app.utils.logger import Logger
from app.config.hyperparameters import HYPERPARAMETERS

router = APIRouter()
logger = Logger.setup_logger(__name__)


def get_student_evaluation_service() -> StudentTrainer:
    """
    Injection de dépendance pour récupérer une instance de StudentTrainer.
    """
    dataset_path = HYPERPARAMETERS["dataset_path"]
    return StudentTrainer(dataset_path=dataset_path)


@router.get("/students/{student_id}/evaluate")
def evaluate_student(
    student_id: str,
    student_service: StudentTrainer = Depends(get_student_evaluation_service),
):
    """
    Endpoint pour évaluer un élève IA individuellement.
    """
    try:
        logger.info(f"Évaluation de l'élève {student_id} commencée.")
        evaluation_result = student_service.evaluate_student(student_id)
        logger.info(f"Évaluation de l'élève {student_id} terminée avec succès.")
        return {"student_id": student_id, "evaluation": evaluation_result}
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation de l'élève {student_id} : {e}")
        return {
            "error": f"Une erreur est survenue lors de l'évaluation de l'élève {student_id}."
        }


@router.get("/students/evaluate_all")
def evaluate_all_students(
    student_service: StudentTrainer = Depends(get_student_evaluation_service),
):
    """
    Endpoint pour évaluer tous les élèves IA collectivement.
    """
    try:
        logger.info("Évaluation collective des élèves commencée.")
        evaluations = student_service.evaluate_all_students()
        logger.info("Évaluation collective des élèves terminée avec succès.")
        return {"evaluations": evaluations}
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation collective des élèves : {e}")
        return {
            "error": "Une erreur est survenue lors de l'évaluation collective des élèves."
        }


@router.get("/students/compare/{student_id1}/{student_id2}")
def compare_students(
    student_id1: str,
    student_id2: str,
    student_service: StudentTrainer = Depends(get_student_evaluation_service),
):
    """
    Endpoint pour comparer deux élèves IA.
    """
    try:
        logger.info(
            f"Comparaison entre les élèves {student_id1} et {student_id2} commencée."
        )
        comparison_result = student_service.compare_students(student_id1, student_id2)
        logger.info(
            f"Comparaison entre les élèves {student_id1} et {student_id2} terminée avec succès."
        )
        return {
            "student_id1": student_id1,
            "student_id2": student_id2,
            "comparison": comparison_result,
        }
    except Exception as e:
        logger.error(
            f"Erreur lors de la comparaison entre les élèves {student_id1} et {student_id2} : {e}"
        )
        return {"error": "Une erreur est survenue lors de la comparaison des élèves."}
