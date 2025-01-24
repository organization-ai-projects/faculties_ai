import logging
import os
import shutil
import stat

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_directory(directory, file_extensions=None):
    """
    Supprime les fichiers spécifiques et réinitialise un répertoire.
    :param directory: Chemin du répertoire à nettoyer
    :param file_extensions: Liste des extensions à supprimer
    """
    try:
        if not directory or not os.path.exists(directory):
            logger.warning(f"Directory does not exist or is not defined: {directory}")
            return

        any_file_deleted = False
        keep_dirs = set()

        # Parcourir les fichiers de bas en haut
        for root, dirs, files in os.walk(directory, topdown=False):
            has_files_to_keep = False

            # Vérifier les fichiers
            for file in files:
                file_path = os.path.join(root, file)
                if file_extensions is None or any(
                    file.endswith(ext) for ext in file_extensions
                ):
                    try:
                        os.chmod(file_path, stat.S_IWRITE)
                        os.remove(file_path)
                        any_file_deleted = True
                        logger.info(f"Deleted file: {file_path}")
                    except OSError as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
                else:
                    has_files_to_keep = True

            # Marquer les répertoires à conserver
            if has_files_to_keep:
                keep_dirs.add(root)
                if root != directory:
                    keep_dirs.add(os.path.dirname(root))

        # Nettoyer les sous-répertoires vides non marqués
        for root, dirs, _ in os.walk(directory, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if dir_path not in keep_dirs:
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except OSError:
                        pass

        # Recréer le répertoire principal si nécessaire
        if any_file_deleted and os.path.exists(directory) and not os.listdir(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
            logger.info(f"Cleaned and recreated directory: {directory}")

    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}")
        raise
