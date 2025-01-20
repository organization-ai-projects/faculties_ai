import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


class Logger:
    """
    Classe utilitaire pour configurer un logger global.
    """

    @staticmethod
    def setup_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB par défaut
        backup_count: int = 3,  # Nombre de sauvegardes des anciens logs
    ) -> logging.Logger:
        """
        Configure un logger avec console et fichier optionnels.
        
        :param name: Nom du logger.
        :param level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        :param log_file: Chemin vers le fichier de log (facultatif).
        :param max_file_size: Taille maximale d'un fichier de log avant rotation.
        :param backup_count: Nombre de sauvegardes des anciens logs.
        :return: Instance configurée de logging.Logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Format des logs
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler pour la console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler pour le fichier (si log_file est défini)
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
