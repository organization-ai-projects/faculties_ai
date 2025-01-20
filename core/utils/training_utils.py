import torch
import numpy as np

def compute_metrics(predictions, labels):
    """
    Calcule les métriques principales pour évaluer le modèle.
    :param predictions: Les prédictions du modèle
    :param labels: Les labels attendus
    :return: Un dictionnaire des métriques calculées
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return {"accuracy": accuracy}

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir):
    """
    Sauvegarde un point de contrôle pour un modèle.
    :param model: Le modèle à sauvegarder
    :param optimizer: L'optimiseur associé
    :param scheduler: Le planificateur de taux d'apprentissage (scheduler)
    :param epoch: L'époque actuelle
    :param checkpoint_dir: Le répertoire pour sauvegarder le checkpoint
    """
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Charge un point de contrôle pour un modèle.
    :param checkpoint_path: Chemin du checkpoint à charger
    :param model: Le modèle à restaurer
    :param optimizer: (optionnel) L'optimiseur à restaurer
    :param scheduler: (optionnel) Le scheduler à restaurer
    :return: L'époque chargée
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return checkpoint['epoch']
