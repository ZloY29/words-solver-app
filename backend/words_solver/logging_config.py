import logging
import os
import sys
import uuid
from contextvars import ContextVar

request_id_var = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        rid = request_id_var.get()
        record.request_id = rid if rid is not None else "-"
        return True


def configure_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    fmt = "%(asctime)s %(levelname)s request_id=%(request_id)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    # IMPORTANT: attach filter to handler so every record gets request_id
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


def set_request_id(value=None):
    if value is None:
        value = uuid.uuid4().hex[:12]
    request_id_var.set(value)
    return value
