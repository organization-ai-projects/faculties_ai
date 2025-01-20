# core/utils/checkpoint_utils.py

import os
import torch
from transformers import AutoModelForCausalLM

class CheckpointUtils:
    """
    Classe dédiée à la gestion des checkpoints.
    Permet de sauvegarder et de restaurer les modèles à partir de checkpoints.
    """

    @staticmethod
    def load_checkpoint(model_name, checkpoint_dir):
        """
        Charge un modèle à partir d'un checkpoint.
        """
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        print(f"Modèle {model_name} chargé depuis le checkpoint {checkpoint_dir}.")
        return model

    @staticmethod
    def save_checkpoint(model, checkpoint_dir):
        """
        Sauvegarde un modèle dans un répertoire de checkpoint.
        """
        model.save_pretrained(checkpoint_dir)
        print(f"Modèle sauvegardé dans {checkpoint_dir}.")
