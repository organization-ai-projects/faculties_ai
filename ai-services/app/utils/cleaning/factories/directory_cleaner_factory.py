import logging

from app.utils.cleaning.directory_cleaner import DirectoryCleaner


class DirectoryCleanerFactory:
    """
    Factory pour créer des instances de DirectoryCleaner.
    """

    @staticmethod
    def create() -> DirectoryCleaner:
        """
        Crée une instance de DirectoryCleaner avec un logger configuré.

        :return: Instance configurée de DirectoryCleaner.
        """
        logger = DirectoryCleanerFactory._setup_logger()
        return DirectoryCleaner(logger=logger)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """
        Configure et retourne un logger.

        :return: Logger configuré pour DirectoryCleaner.
        """
        logger = logging.getLogger("DirectoryCleaner")

        # Évite les doublons dans les handlers
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
