from app.models.teachers.teacher_trainer import TeacherTrainer
from app.models.teachers.interfaces.training_interface import TrainingInterface
from app.utils.logger import Logger
from app.config.hyperparameters import HYPERPARAMETERS


class TrainingService(TrainingInterface):
    """
    Implémentation du service d'entraînement IA basé sur TeacherTrainer.
    """

    def __init__(self, trainer: TeacherTrainer, default_hyperparameters=None):
        """
        Initialise le service d'entraînement avec une instance de TeacherTrainer
        et des hyperparamètres par défaut.
        """
        self.trainer = trainer
        self.logger = Logger.setup_logger(__name__)
        self.default_hyperparameters = (
            default_hyperparameters or HYPERPARAMETERS["teachers"]
        )

    def train_all_roles(self) -> str:
        """
        Enchaîne l'entraînement pour tous les rôles des professeurs définis
        dans les hyperparamètres.
        """
        try:
            roles = self.default_hyperparameters["roles"]

            for role_name, role_params in roles.items():
                model_name = role_params["model"]
                task = role_params["task"]

                self.logger.info(f"Starting training for role '{role_name}'")
                self.logger.info(
                    f"Model: {model_name}, Task: {task}, Epochs: "
                    f"{self.default_hyperparameters['epochs']}, Learning Rate: "
                    f"{self.default_hyperparameters['learning_rate']}"
                )

                # Configure le trainer pour ce rôle
                self.trainer.configure(model_name=model_name, task=task)

                # Lancer l'entraînement pour ce rôle
                self.trainer.train(
                    epochs=self.default_hyperparameters["epochs"],
                    learning_rate=self.default_hyperparameters["learning_rate"],
                )

                self.logger.info(f"Training completed for role '{role_name}'")

            return "All roles trained successfully."

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def train_specific_role(
        self, role_name: str, epochs: int = None, learning_rate: float = None
    ) -> str:
        """
        Lancement de l'entraînement pour un rôle spécifique avec les paramètres
        spécifiés ou par défaut.
        :param role_name: Le rôle du professeur (ex: 'basic_coding', 'debugging').
        :param epochs: Nombre d'époques à utiliser (facultatif).
        :param learning_rate: Taux d'apprentissage à utiliser (facultatif).
        """
        try:
            roles = self.default_hyperparameters["roles"]
            if role_name not in roles:
                raise ValueError(f"Role '{role_name}' not found in teacher roles.")

            role_params = roles[role_name]
            model_name = role_params["model"]
            task = role_params["task"]

            # Utiliser les valeurs par défaut si aucun paramètre n'est spécifié
            epochs = epochs or self.default_hyperparameters["epochs"]
            learning_rate = (
                learning_rate or self.default_hyperparameters["learning_rate"]
            )

            self.logger.info(f"Starting training for specific role '{role_name}'")
            self.logger.info(
                f"Model: {model_name}, Task: {task}, Epochs: {epochs}, "
                f"Learning Rate: {learning_rate}"
            )

            # Configure le trainer pour ce rôle
            self.trainer.configure(model_name=model_name, task=task)

            # Lancer l'entraînement pour ce rôle
            self.trainer.train(epochs=epochs, learning_rate=learning_rate)

            self.logger.info(f"Training completed for role '{role_name}'")

            return f"Training completed for role '{role_name}'."

        except Exception as e:
            self.logger.error(f"Training failed for role '{role_name}': {e}")
            raise
