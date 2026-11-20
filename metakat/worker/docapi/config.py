"""The MetaKat worker's settings: DocAPI's worker settings plus MetaKat's own.

doc_worker.config.Config reads and checks every setting of DocAPI's worker
wrapper (the API, the directories, the worker's mail, ALIVE reports and outage
mail); DocAPI's doc_worker/.env_example documents them. This class adds the
settings MetaKat's processing and logging need, and MetaKat's defaults where
they differ from DocAPI's. A Config is built once, from the environment and any
command-line overrides (Config({"BASE_DIR": ...})), and is then only read.
"""
import logging
import os
import socket
import time

from doc_worker.config import Config as DocWorkerConfig


class MetakatWorkerFormatter(logging.Formatter):
    """Not masked: the worker log is for operators. Secrets are kept out where they are
    logged (metakat.logging_utils.redacted_for_logging); DocAPI masks worker mail."""
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

        # Logging
        ################################################################################################################
        self.LOGGING_CONSOLE_LEVEL = self._str("LOGGING_CONSOLE_LEVEL", "INFO")
        self.LOGGING_FILE_LEVEL = self._str("LOGGING_FILE_LEVEL", "INFO")
        # The directory of worker.log.
        self.LOGGING_DIR = self._str("LOGGING_DIR", os.path.join(self.BASE_DIR, "logs"))
        # The job history (DocAPI's JOB_HISTORY_PATH) is kept beside worker.log unless
        # set; set but empty keeps none.
        if not self._is_set("JOB_HISTORY_PATH"):
            self.JOB_HISTORY_PATH = os.path.join(self.LOGGING_DIR, "job_history.jsonl")

    def check(self):
        super().check()
        if not self.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError("ALLOWED_IMAGE_EXTENSIONS must name at least one extension")
        levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        for name in ("LOGGING_CONSOLE_LEVEL", "LOGGING_FILE_LEVEL"):
            if getattr(self, name) not in levels:
                raise ValueError(f"{name} must be one of {', '.join(levels)}, not {getattr(self, name)!r}")

    def _is_set(self, key: str) -> bool:
        """Whether key is given at all, even empty: as an override or in the environment."""
        # setting_names() reads the settings on a probe without overrides.
        return key in getattr(self, "_overrides", {}) or key in os.environ

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
        for directory in (self.JOBS_DIR or os.path.join(self.BASE_DIR, "jobs"),
                          self.ENGINES_DIR or os.path.join(self.BASE_DIR, "engines"),
                          self.LOGGING_DIR):
            os.makedirs(directory, exist_ok=True)
