import logging
import sys

from pythonjsonlogger import jsonlogger


def setup_logger(
    name: str | None = None,
    level: str = "INFO",
    use_json: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with JSON or text formatting.

    Args:
        name: Logger name (use __name__ in modules)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: If True and python-json-logger installed, output JSON

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)

    if use_json and jsonlogger:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False

    return logger
