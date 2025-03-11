import logging
from contextvars import ContextVar


def loggers_utils(log_name: str) -> logging.Logger:
    """Instance of Python Logger Class for local logging.

    Args:
        log_name (str): Name of the logger instance.

    Returns:
        logging.Logger: Instance of the logger.
    """
    logger = logging.getLogger(log_name)
    return logger


class AppFilter(logging.Filter):
    def __init__(self, correlation_id_ctx_var, name: str = "") -> None:
        """
        Initialize the AppFilter with a correlation ID context variable.

        Args:
            correlation_id_ctx_var (ContextVar): A context variable that stores the correlation ID.
            name (str): Optional. The name of the filter. Defaults to an empty string.
        """
        self._correlation_id_ctx_var = correlation_id_ctx_var
        super().__init__(name)

    def filter(self, record):
        """
        Modifies the logging record to include a correlation ID.
        """
        record.correlation_id = self._correlation_id_ctx_var.get()
        return True


class CustomHandler(logging.StreamHandler):
    """
    A custom logging handler that processes multiline string records into separate log entries.
    """

    def __init__(self):
        super(CustomHandler, self).__init__()

    def emit(self, record):
        try:
            messages = record.msg.split("\n")
            for message in messages:
                record.msg = message
                super(CustomHandler, self).emit(record)
        except Exception:
            super(CustomHandler, self).emit(record)


def setup_logging(correlation_id_ctx_var: ContextVar):
    """
    Sets up local logging configuration with correlation ID support.
    """
    formatter = logging.Formatter(
        "%(levelname)s - %(name)s - [%(correlation_id)s] - %(message)s"
    )
    handler = CustomHandler()
    handler.addFilter(AppFilter(correlation_id_ctx_var))
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
