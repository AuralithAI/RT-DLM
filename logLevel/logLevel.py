import logging

class Logging:
    """Custom logging utility class for standardized logging across the project."""
    
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def configure(cls, level=logging.DEBUG, log_format='[%(levelname)s] -- %(message)s'):
        """Configures the logger with a specific level and format."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        cls._logger.setLevel(level)
        cls._logger.handlers.clear()
        cls._logger.addHandler(handler)

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