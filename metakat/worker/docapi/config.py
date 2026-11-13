import json
import os
import logging
import time
import socket
import math

class MetakatWorkerFormatter(logging.Formatter):
    """Not masked: the worker log is for operators. Secrets are kept out where they are
    logged (metakat.logging_utils.redacted_for_logging); DocAPI masks worker mail."""
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        record.hostname = socket.gethostname()
        return super().format(record)


TRUE_VALUES = {"true", "1"}


class Config:
    def __init__(self):
        # THIS MUST BE CHANGED IN PRODUCTION
        self.API_URL = os.getenv("API_URL", "https://metakat.smart.lib.cas.cz")
        self.WORKER_KEY = os.getenv("WORKER_KEY", f"metakat.defaultworkerkid.defaultworkerkey")

        self.ALLOWED_IMAGE_EXTENSIONS = set(os.getenv("ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.tif,.tiff").split(","))

        self.BASE_DIR = os.getenv("BASE_DIR", "./metakat_worker_data")
        self.JOBS_DIR = os.getenv("JOBS_DIR", os.path.join(self.BASE_DIR, "jobs"))
        self.ENGINES_DIR = os.getenv("ENGINES_DIR", os.path.join(self.BASE_DIR, "engines"))

        self.POLLING_INTERVAL = int(os.getenv("POLLING_INTERVAL", "5"))

        self.CLEANUP_JOB_DIR = self._env_bool("CLEANUP_JOB_DIR", False)
        self.CLEANUP_OLD_ENGINES = self._env_bool("CLEANUP_OLD_ENGINES", False)
        self.STORE_METAKAT_PDF = self._env_bool("STORE_METAKAT_PDF", False)
        self.STORE_MODS_OVERVIEW = self._env_bool("STORE_MODS_OVERVIEW", False)

        # Prefer normal API uploads when both sources exist, unless enabled.
        self.PREFER_IMAGE_URL = self._env_bool("PREFER_IMAGE_URL", False)
        self.PREFER_ALTO_URL = self._env_bool("PREFER_ALTO_URL", False)
        self.PREFER_PAGE_URL = self._env_bool("PREFER_PAGE_URL", False)

        # Worker health mail only; DocAPI sends job outcome notifications.
        self.MAIL_STARTUP = self._env_bool("MAIL_STARTUP", True)
        # Alive interval 0 disables periodic reports. Failures use the mailer directly.
        self.MAIL_ALIVE_INTERVAL_SECONDS = self._nonnegative_seconds("MAIL_ALIVE_INTERVAL_SECONDS", 86400)
        # How health and failure mail names this worker, e.g. "MetaKat worker prod";
        # empty uses the class name (MetakatWorker) and socket.gethostname().
        self.WORKER_IDENTITY = os.getenv("WORKER_IDENTITY") or None
        self.WORKER_HOSTNAME = os.getenv("WORKER_HOSTNAME") or None
        # Python format string with {identity}, {hostname}, {status} (ALIVE/FAILURE)
        # and {time} (UTC datetime, e.g. {time:%Y-%m-%d}).
        self.MAIL_SUBJECT = os.getenv("MAIL_SUBJECT") or "[{identity}] {status} ({hostname})"

        # MAILING configuration
        ################################################################################################################
        # without MAIL_SERVER and MAIL_SENDER_MAIL the worker sends no emails
        self.MAIL_SERVER = os.getenv("MAIL_SERVER", None)
        self.MAIL_PORT = int(os.getenv("MAIL_PORT", "25"))
        self.MAIL_TIMEOUT_SECONDS = float(os.getenv("MAIL_TIMEOUT_SECONDS", "60"))
        if not math.isfinite(self.MAIL_TIMEOUT_SECONDS) or self.MAIL_TIMEOUT_SECONDS <= 0:
            raise ValueError("MAIL_TIMEOUT_SECONDS must be a finite, positive number")
        # auto (ssl with a password, none without), none, ssl or starttls
        self.MAIL_SECURITY = os.getenv("MAIL_SECURITY", "auto")
        self.MAIL_SENDER_MAIL = os.getenv("MAIL_SENDER_MAIL", None)
        self.MAIL_SENDER_NAME = os.getenv("MAIL_SENDER_NAME", "MetaKat worker")
        # login name for the server, MAIL_SENDER_MAIL by default; used only with a password
        self.MAIL_USERNAME = os.getenv("MAIL_USERNAME", None)
        self.MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", None)
        # who receives emails sent without receivers of their own, comma separated
        self.MAIL_RECEIVER_MAILS = [mail.strip() for mail in os.getenv("MAIL_RECEIVER_MAILS", "").split(",")
                                    if mail.strip()]

        # LOGGING configuration
        ################################################################################################################
        self.LOGGING_CONSOLE_LEVEL = os.getenv("LOGGING_CONSOLE_LEVEL", logging.INFO)
        self.LOGGING_FILE_LEVEL = os.getenv("LOGGING_FILE_LEVEL", logging.INFO)
        self.LOGGING_DIR = os.getenv("LOGGING_DIR", os.path.join(self.BASE_DIR, "logs"))
        # Empty disables history; otherwise default beside the general logs.
        self._job_history_path = os.getenv("JOB_HISTORY_PATH")

    @property
    def JOB_HISTORY_PATH(self):
        if self._job_history_path is None:
            return os.path.join(self.LOGGING_DIR, "job_history.jsonl")
        return self._job_history_path or None

    @property
    def LOGGING_CONFIG(self):
        """Assembled on access rather than stored.

        The worker applies command-line overrides to this object after it has
        been constructed, so a dictionary built in __init__ would describe the
        environment rather than the settings actually in force.
        """
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
                    'filename': os.path.join(self.LOGGING_DIR, f'worker.log')
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


    def _env_bool(self, key: str, default: bool = False) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.strip().lower() in TRUE_VALUES

    @staticmethod
    def _nonnegative_seconds(key: str, default: float) -> float:
        value = float(os.getenv(key, str(default)))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite, nonnegative number")
        return value

    def create_dirs(self):
        os.makedirs(self.JOBS_DIR, exist_ok=True)
        os.makedirs(self.ENGINES_DIR, exist_ok=True)
        os.makedirs(self.LOGGING_DIR, exist_ok=True)


config = Config()
