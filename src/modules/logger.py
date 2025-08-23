"""
Logging management layer

- dotenv file: define the logging level with "LOG_LEVEL" key
- define get_logger() that returns the logger
- requires:
    - structlog
    - python-dotenv
"""

import logging
import os
import sys

import structlog
from dotenv import load_dotenv

load_dotenv()


def get_logger(name):
    # Configure logging underlying system
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
    # Struct logger setup
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.dev.ConsoleRenderer()
                if sys.stdout.isatty()
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(name)
