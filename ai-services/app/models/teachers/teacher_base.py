import os  # Import manquant ajouté
import torch
import torch.nn as nn
from typing import Optional, List
from app.components.transformer_layer import TransformerLayer
from app.components.mlp_mixer import MLPMixer
from app.components.memory_module import MemoryModule
from app.components.cnn import CNN1D
from app.components.rmsnorm import RMSNorm
from app.components.rotary_position_embedding import RotaryPositionEmbedding
from app.components.gated_feedforward import GatedFeedForward
from app.components.flash_mha import FlashMHA, FLASH_AVAILABLE
from app.utils.logger import Logger

class TeacherBase:
    """
    Base class for AI teachers, managing model handling and interactions.
    """

    def __init__(self, teacher_model: Optional[nn.Module] = None):
        """
        Initializes an AI teacher with a dedicated model.
        :param teacher_model: the teacher model, defaulting to an OptimizedStudentTransformer if not provided.
        """
        self.teacher_model = teacher_model if teacher_model else self._create_default_model()
        self.logger = Logger.setup_logger(__name__)
        self.logger.info("TeacherBase model initialized successfully.")

    def _create_default_model(self) -> nn.Module:
        """
        Creates and returns the default teacher model.
        :return: The default teacher model.
        """
        vocab_size = 50257
        hidden_size = 768
        num_layers = 6
        num_heads = 12
        max_seq_len = 128
        dropout = 0.1
        sd_prob = 0.1
        ff_multiplier = 4

        return nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.Dropout(dropout),
            *[
                TransformerLayer(
                    hidden_size,
                    num_heads,
                    dropout,
                    sd_prob * (layer_idx / (num_layers - 1)) if num_layers > 1 else 0.0,
                    use_flash_attn=FLASH_AVAILABLE,
                    ff_multiplier=ff_multiplier,
                )
                for layer_idx in range(num_layers)
            ],
            MLPMixer(hidden_size, max_seq_len),
            GatedFeedForward(hidden_size, ff_multiplier, dropout),
            CNN1D(hidden_size),
            MemoryModule(hidden_size),
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

    def learn_from_students(self, feedback: List[str]):
        """
        Allows a teacher to learn from students by analyzing their feedback and adjusting weights.
        :param feedback: Feedback from students.
        """
        self.logger.info("Learning from students' feedback...")
        self._adjust_strategy_based_on_feedback(feedback)

    def _adjust_strategy_based_on_feedback(self, feedback: List[str]):
        """
        Adjusts the teacher's training strategy based on feedback from students.
        :param feedback: Feedback from students.
        """
        self.logger.info("Adjusting strategy based on students' feedback...")
        # Extend with logic to adapt teacher's strategy
        pass

    def learn_from_peers(self, peers: List['TeacherBase']):
        """
        Allows a teacher to learn from other teachers based on their expertise.
        :param peers: Other teachers.
        """
        self.logger.info("Learning from other teachers...")
        self._adjust_strategy_based_on_peers(peers)

    def _adjust_strategy_based_on_peers(self, peers: List['TeacherBase']):
        """
        Adjusts a teacher's strategy based on the strategies of other teachers.
        :param peers: Other teachers.
        """
        self.logger.info("Adjusting strategy based on other teachers...")
        # Extend with logic for collaboration between teachers
        pass

    def save_teacher_model(self, output_dir: str):
        """
        Saves the teacher's model.
        :param output_dir: Directory to save the model.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "teacher_model.pth")
            torch.save(self.teacher_model.state_dict(), model_path)
            self.logger.info(f"Teacher model saved successfully at {model_path}.")
        except Exception as e:
            self.logger.error(f"Error saving teacher model: {e}")
            raise

    def get_responses(self, student_model: nn.Module) -> List[str]:
        """
        Gets the responses of a teacher for a given student.
        :param student_model: The student model.
        :return: List of responses.
        """
        try:
            responses = ["Response A", "Response B"]
            self.logger.info("Responses generated successfully.")
            return responses
        except Exception as e:
            self.logger.error(f"Error generating responses: {e}")
            raise
