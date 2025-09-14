"""
Logging management layer

- dotenv file: define the logging level with "LOG_LEVEL" key
- define get_logger() that returns the logger
- requires:
    - structlog
    - python-dotenv
    - colorama
"""

import logging
import os
import sys

import colorama
import structlog
from dotenv import load_dotenv
from structlog.dev import Column, KeyValueColumnFormatter
from structlog.processors import CallsiteParameter

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
            structlog.processors.TimeStamper(fmt="%Y-%M-%d %H:%M:%S"),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.MODULE,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.add_log_level,  # This adds level coloring
            (
                structlog.dev.ConsoleRenderer(
                    columns=[
                        Column(
                            "timestamp",
                            KeyValueColumnFormatter(
                                key_style=None,
                                value_style=colorama.Fore.BLACK + colorama.Style.BRIGHT,
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix="[",
                                postfix="] ",  # Note the space after ]
                            ),
                        ),
                        Column(
                            "level",
                            KeyValueColumnFormatter(
                                key_style=None,
                                value_style=colorama.Style.BRIGHT,  # Let the level coloring processor handle this
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix="[",
                                postfix="] ",  # Note the space after ]
                            ),
                        ),
                        Column(
                            "module",
                            KeyValueColumnFormatter(
                                key_style=None,
                                value_style=colorama.Style.BRIGHT,
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix="[",
                                postfix="",  # No space after, we want tight formatting
                            ),
                        ),
                        Column(
                            "func_name",
                            KeyValueColumnFormatter(
                                key_style=None,
                                value_style=colorama.Style.BRIGHT,
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix=":",
                                postfix="",  # No space after
                            ),
                        ),
                        Column(
                            "lineno",
                            KeyValueColumnFormatter(
                                key_style=None,
                                value_style=colorama.Style.BRIGHT,
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix=":",
                                postfix="] ",  # Space after ] to separate from event
                            ),
                        ),
                        Column(
                            "event",
                            KeyValueColumnFormatter(
                                key_style=None,
                                value_style=colorama.Style.BRIGHT + colorama.Fore.BLUE,
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix="",
                                postfix="",
                            ),
                        ),
                        Column(
                            "",  # This handles any remaining key-value pairs
                            KeyValueColumnFormatter(
                                key_style=colorama.Fore.CYAN,
                                value_style=colorama.Fore.GREEN,
                                reset_style=colorama.Style.RESET_ALL,
                                value_repr=str,
                                prefix=" ",  # Space before additional key-value pairs
                                postfix="",
                            ),
                        ),
                    ]
                )
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
