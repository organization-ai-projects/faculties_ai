import torch
from app.utils.logger import Logger


class ProfessorCollaborator:
    """
    Handles collaboration logic between students and professors.
    """

    def __init__(self):
        self.logger = Logger.get_logger(__name__)

    def collaborate(
        self, student_model: torch.nn.Module, professor_model: torch.nn.Module
    ):
        """
        Adjusts the student model using knowledge from the professor model.
        """
        self.logger.info(
            f"Collaborating with professor: {professor_model.__class__.__name__}"
        )
        with torch.no_grad():
            for student_param, prof_param in zip(
                student_model.parameters(), professor_model.parameters()
            ):
                student_param.data += 0.1 * prof_param.data
