import logging
import getpass
from datetime import datetime
from pathlib import Path

class Logging:
    """Custom logging utility class for standardized logging across the project."""

    _logger = logging.getLogger(__name__)  
    _username = getpass.getuser()
    _log_file = Path(__file__).parent.parent / "rt_dlm.log"

    @classmethod
    def configure(cls, level=logging.DEBUG, log_file=None):
        """Configures the logger with DEBUG level, format, and file output.
        
        Args:
            level: Logging level (default: DEBUG)
            log_file: Optional path to log file (default: rt_dlm.log in project root)
        """
        if log_file is None:
            log_file = cls._log_file
            
        log_format = f"[%(asctime)s] [{cls._username}] [%(levelname)s] -- %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        cls._logger.setLevel(level)
        cls._logger.handlers.clear()
        cls._logger.addHandler(console_handler)

        # File handler - always add for persistent logging
        try:
            file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            cls._logger.addHandler(file_handler)
            cls._logger.debug(f"Logging to file: {log_file}")
        except Exception as e:
            cls._logger.warning(f"Could not create log file {log_file}: {e}")

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


# Auto-configure with DEBUG level and file handler on import
Logging.configure(level=logging.DEBUG)
