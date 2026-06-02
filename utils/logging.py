import sys

try:
    from loguru import logger

    def setup_logging(level: str = "DEBUG"):
        logger.remove()
        logger.add(sys.stderr, level=level, colorize=True)

except ImportError:
    import logging as _logging

    class _Logger:
        """Loguru-compatible wrapper around stdlib logging for worker-node compatibility."""

        def __init__(self):
            self._log = _logging.getLogger("spmv")

        def opt(self, *, exception: bool = False, **_):
            log, ei = self._log, exception
            class _Proxy:
                def warning(self, msg, *a, **kw): log.warning(msg, *a, exc_info=ei, **kw)
                def error(self, msg, *a, **kw):   log.error(msg, *a, exc_info=ei, **kw)
            return _Proxy()

        def debug(self, msg, *a, **kw):    self._log.debug(msg, *a, **kw)
        def info(self, msg, *a, **kw):     self._log.info(msg, *a, **kw)
        def warning(self, msg, *a, **kw):  self._log.warning(msg, *a, **kw)
        def error(self, msg, *a, **kw):    self._log.error(msg, *a, **kw)
        def critical(self, msg, *a, **kw): self._log.critical(msg, *a, **kw)

    logger = _Logger()

    def setup_logging(level: str = "DEBUG"):
        _logging.basicConfig(
            level=getattr(_logging, level.upper(), _logging.DEBUG),
            format="%(levelname)s: %(message)s",
            stream=sys.stderr,
        )


__all__ = ["logger", "setup_logging"]
