import os
import shutil


def reset_dataset(original_path, target_path):
    """
    Copie un dataset depuis une source d'origine vers un chemin cible.
    :param original_path: Chemin du dataset original.
    :param target_path: Chemin où réinitialiser le dataset.
    """
    if os.path.exists(original_path):
        shutil.copy(original_path, target_path)
        print(f"Dataset reset from {original_path} to {target_path}")
    else:
        print(f"Original dataset not found: {original_path}")
