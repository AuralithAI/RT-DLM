import logging
import os
from datetime import datetime

class Logging:
    """Custom logging utility class for standardized logging across the project."""

    _logger = logging.getLogger(__name__)  
    _username = os.getlogin()  

    @classmethod
    def configure(cls, level=logging.INFO, log_file=None):
        """Configures the logger with a specific level, format, and optional file output."""
        log_format = f"[%(asctime)s] [{cls._username}] [%(levelname)s] -- %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        cls._logger.setLevel(level)
        cls._logger.handlers.clear()
        cls._logger.addHandler(handler)

        # Optional: Add file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)

    @classmethod
    def debug(cls, message: str):
        """Logs a debug message."""
        cls._logger.debug(message)

    @classmethod
    def info(cls, message: str):
        """Logs an info message."""
        cls._logger.info(message)

    @classmethod
    def warning(cls, message: str):
        """Logs a warning message."""
        cls._logger.warning(message)

    @classmethod
    def error(cls, message: str):
        """Logs an error message."""
        cls._logger.error(message)

    @classmethod
    def critical(cls, message: str):
        """Logs a critical message."""
        cls._logger.critical(message)
