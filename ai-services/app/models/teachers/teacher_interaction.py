from typing import List
from app.utils.logger import Logger
from app.models.teachers.teacher_base import TeacherBase
from app.alerts.alert_system import (
    AlertSystem,
)  # Assuming this exists in the new structure


class TeacherInteraction:
    """
    Classe responsable de l'interaction entre les professeurs et les étudiants.
    Les professeurs enseignent aux étudiants, évaluent leurs performances, et détectent
    les incohérences.
    """

    def __init__(self, teachers: List[TeacherBase], alert_system: AlertSystem):
        """
        Initialise l'interaction entre un ensemble de professeurs et les étudiants.
        :param teachers: Liste d'instances des professeurs (de type TeacherBase).
        :param alert_system: Système d'alerte pour détecter les incohérences ou incertitudes.
        """
        if not all(isinstance(teacher, TeacherBase) for teacher in teachers):
            raise TypeError("All teachers must be instances of TeacherBase")

        self.teachers = teachers
        self.alert_system = alert_system
        self.logger = Logger.setup_logger(__name__)
        self.logger.info("TeacherInteraction initialized successfully.")

    def teach_students(self, student_models: List[TeacherBase]):
        """
        Permet aux professeurs d'enseigner aux étudiants.
        Chaque professeur utilise ses connaissances pour entraîner un ou plusieurs modèles
        d'étudiants.
        :param student_models: Liste des modèles d'étudiants à entraîner.
        """
        self.logger.info("Teaching students...")
        try:
            for teacher in self.teachers:
                for student in student_models:
                    teacher.teach(
                        student
                    )  # Méthode spécifique au professeur pour entraîner un étudiant
        except Exception as e:
            self.logger.error(f"Error during student teaching: {e}")
            raise

    def evaluate_students(self, student_models: List[TeacherBase]):
        """
        Évalue les performances des étudiants.
        :param student_models: Liste des modèles d'étudiants à évaluer.
        """
        self.logger.info("Evaluating students...")
        try:
            for student in student_models:
                for teacher in self.teachers:
                    performance = teacher.evaluate_student(student)
                    self.logger.info(
                        f"Performance de l'étudiant {student.__class__.__name__} "
                        f"par {teacher.__class__.__name__} : {performance}"
                    )
        except Exception as e:
            self.logger.error(f"Error during student evaluation: {e}")
            raise

    def detect_incoherence_and_alert(self, student_models: List[TeacherBase]):
        """
        Détecte les incohérences dans les retours des étudiants et alerte si nécessaire.
        :param student_models: Liste des modèles d'étudiants à vérifier.
        """
        self.logger.info("Detecting incoherences in student feedback...")
        try:
            for student in student_models:
                responses = [
                    teacher.get_student_feedback(student) for teacher in self.teachers
                ]
                if self.is_incoherent(responses):
                    self.logger.warning(
                        f"Incoherence detected in feedback for student "
                        f"{student.__class__.__name__}. Alert sent..."
                    )
                    self.alert_system.send_alert(
                        student, responses
                    )  # Envoi de l'alerte au superviseur
        except Exception as e:
            self.logger.error(f"Error during incoherence detection: {e}")
            raise

    def is_incoherent(self, responses: List[str]) -> bool:
        """
        Vérifie si les réponses des professeurs sur un étudiant sont incohérentes.
        Cela peut être basé sur des divergences dans les évaluations.
        :param responses: Liste des réponses des professeurs sur un étudiant.
        :return: True si une incohérence est détectée, False sinon.
        """
        return len(set(responses)) > 1
