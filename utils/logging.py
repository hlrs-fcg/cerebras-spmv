from loguru import logger
import sys

def setup_logging(level: str = "DEBUG"):
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)

__all__ = ["logger", "setup_logging"]
