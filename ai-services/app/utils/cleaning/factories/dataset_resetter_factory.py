import logging

from app.utils.cleaning.dataset_resetter import DatasetResetter


class DatasetResetterFactory:
    """
    Factory pour créer des instances de DatasetResetter.
    """

    @staticmethod
    def create() -> DatasetResetter:
        """
        Crée une instance de DatasetResetter avec un logger configuré.

        :return: Instance de DatasetResetter.
        """
        logger = logging.getLogger("DatasetResetter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return DatasetResetter(logger=logger)
