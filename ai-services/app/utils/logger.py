import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
from typing import Optional, Dict


class Logger:
    """
    Utility class to configure a global logger with support for console, file, and email logging.
    """

    @staticmethod
    def setup_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 3,
        log_format: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        email_alerts: bool = False,
        email_config: Optional[Dict] = None,
    ) -> logging.Logger:
        """
        Configures a logger with console, file, and optional email handlers.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not logger.hasHandlers():
            formatter = logging.Formatter(log_format, datefmt=date_format)
            console_handler = Logger._get_console_handler(formatter)
            logger.addHandler(console_handler)

            if log_file:
                file_handler = Logger._get_file_handler(
                    log_file, max_file_size, backup_count, formatter
                )
                logger.addHandler(file_handler)

            if email_alerts and email_config:
                email_handler = Logger._get_email_handler(email_config, formatter)
                logger.addHandler(email_handler)

        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Returns a logger instance for the given name.
        """
        return Logger.setup_logger(name=name)

    @staticmethod
    def _get_console_handler(formatter: logging.Formatter) -> logging.StreamHandler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        return console_handler

    @staticmethod
    def _get_file_handler(
        log_file: str,
        max_file_size: int,
        backup_count: int,
        formatter: logging.Formatter,
    ) -> RotatingFileHandler:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        return file_handler

    @staticmethod
    def _get_email_handler(
        email_config: Dict, formatter: logging.Formatter
    ) -> SMTPHandler:
        required_keys = {"mailhost", "fromaddr", "toaddrs", "subject"}
        if not required_keys.issubset(email_config.keys()):
            raise ValueError(
                f"Email configuration must include: {', '.join(required_keys)}"
            )

        email_handler = SMTPHandler(
            mailhost=email_config["mailhost"],
            fromaddr=email_config["fromaddr"],
            toaddrs=email_config["toaddrs"],
            subject=email_config["subject"],
            credentials=email_config.get("credentials"),
            secure=email_config.get("secure"),
        )
        email_handler.setLevel(logging.ERROR)
        email_handler.setFormatter(formatter)
        return email_handler
