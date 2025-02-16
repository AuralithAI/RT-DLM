import logging

class Logging:
    """Custom logging utility class for standardized logging across the project."""

    # Configure logging format
    logging.basicConfig(
        format='[%(levelname)s] -- %(message)s',
        level=logging.DEBUG
    )

    @staticmethod
    def debug(message: str):
        """Logs a debug message."""
        logging.debug(message)

    @staticmethod
    def info(message: str):
        """Logs an info message."""
        logging.info(message)

    @staticmethod
    def warning(message: str):
        """Logs a warning message."""
        logging.warning(message)

    @staticmethod
    def error(message: str):
        """Logs an error message."""
        logging.error(message)

    @staticmethod
    def critical(message: str):
        """Logs a critical message."""
        logging.critical(message)