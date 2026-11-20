"""The MetaKat worker's settings.

Config holds what the worker is built from: doc_worker.config.Config reads and
checks every setting of DocAPI's worker wrapper (the API, the directories, the
job history, the worker's mail, ALIVE reports and outage mail; DocAPI's
doc_worker/.env_example documents them), and Config adds MetaKat's processing
settings, each passed to the MetakatWorker parameter of the same name, and
MetaKat's defaults where they differ from DocAPI's.

LoggingConfig holds MetaKat's own process log, worker.log, which main() sets up
before the worker exists. Each is built once, from the environment and any
command-line overrides (Config({"BASE_DIR": ...})), and is then only read.
"""
import logging
import os
import socket
import time
from typing import Mapping, Optional

from doc_worker.config import Config as DocWorkerConfig


class MetakatWorkerFormatter(logging.Formatter):
    """Not masked, like the worker's mail, which DocAPI sends as written: both are for
    operators. Secrets are kept out where they are logged
    (metakat.logging_utils.redacted_for_logging)."""
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        record.hostname = socket.gethostname()
        return super().format(record)


class Config(DocWorkerConfig):
    def read(self):
        super().read()
        # MetaKat's defaults for DocAPI's settings
        ################################################################################################################
        self.API_URL = self.API_URL or "https://metakat.smart.lib.cas.cz"
        self.BASE_DIR = self.BASE_DIR or "./metakat_worker_data"
        # The sender's name in the worker's mail, when mail is set up.
        if self.MAIL_SERVER and not self.MAIL_SENDER_NAME:
            self.MAIL_SENDER_NAME = "MetaKat worker"

        # MetaKat processing
        ################################################################################################################
        # The image extensions a job may contain; a job with another one fails.
        self.ALLOWED_IMAGE_EXTENSIONS = set(self._list("ALLOWED_IMAGE_EXTENSIONS")
                                            or [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
        # Keep an interactive result.pdf and metakat.mods.txt, every MODS record in one
        # file, beside result.zip in the job's directory, for inspection.
        self.STORE_METAKAT_PDF = self._bool("STORE_METAKAT_PDF", False)
        self.STORE_MODS_OVERVIEW = self._bool("STORE_MODS_OVERVIEW", False)

    def check(self):
        super().check()
        if not self.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError("ALLOWED_IMAGE_EXTENSIONS must name at least one extension")

    def create_dirs(self):
        for directory in (self.JOBS_DIR or os.path.join(self.BASE_DIR, "jobs"),
                          self.ENGINES_DIR or os.path.join(self.BASE_DIR, "engines")):
            os.makedirs(directory, exist_ok=True)


class LoggingConfig:
    """The process log: worker.log in LOGGING_DIR, rotated at UTC midnight, and the console.

    The log of each job, worker.log in the job's directory, is DocAPI's wrapper's.
    An unset or empty variable keeps the default, as in Config.
    """
    LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def __init__(self, base_dir: str, overrides: Optional[Mapping[str, str]] = None):
        """base_dir is the worker's BASE_DIR; overrides replace the environment variables of their names."""
        overrides = dict(overrides or {})
        unknown = set(overrides) - {"LOGGING_DIR", "LOGGING_CONSOLE_LEVEL", "LOGGING_FILE_LEVEL"}
        if unknown:
            raise TypeError(f"Unknown settings: {', '.join(sorted(unknown))}")

        def read(key: str, default: str) -> str:
            return (overrides[key] if key in overrides else os.getenv(key, "")) or default

        # The directory of worker.log.
        self.LOGGING_DIR = read("LOGGING_DIR", os.path.join(base_dir, "logs"))
        # How much goes to the console (the screen session) and into worker.log.
        self.LOGGING_CONSOLE_LEVEL = read("LOGGING_CONSOLE_LEVEL", "INFO")
        self.LOGGING_FILE_LEVEL = read("LOGGING_FILE_LEVEL", "INFO")
        for name in ("LOGGING_CONSOLE_LEVEL", "LOGGING_FILE_LEVEL"):
            if getattr(self, name) not in self.LEVELS:
                raise ValueError(f"{name} must be one of {', '.join(self.LEVELS)}, not {getattr(self, name)!r}")

    @property
    def LOGGING_CONFIG(self) -> dict:
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'base': {
                    '()': MetakatWorkerFormatter,
                    'format': '%(asctime)s : %(name)s : %(hostname)s : %(levelname)s : %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.LOGGING_CONSOLE_LEVEL,
                    'formatter': 'base',
                    'stream': 'ext://sys.stdout'
                },
                'file_log': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'level': self.LOGGING_FILE_LEVEL,
                    'when': 'midnight',
                    'utc': True,
                    'formatter': 'base',
                    'filename': os.path.join(self.LOGGING_DIR, 'worker.log')
                }
            },
            'loggers': {
                'root': {
                    'level': 'DEBUG',
                    'handlers': [
                        'console',
                        'file_log',
                    ]
                },
                'doc_api.exception_logger': {
                    'level': 'DEBUG',
                    'handlers': [
                        'file_log'
                    ]
                },
                'multipart.multipart': {
                    'level': 'INFO'
                }
            }
        }

    def create_dirs(self):
        os.makedirs(self.LOGGING_DIR, exist_ok=True)
