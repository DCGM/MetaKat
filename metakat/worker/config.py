import json
import os
import logging
import time
import socket

class MetakatWorkerFormatter(logging.Formatter):
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

        self.BASE_DIR = os.getenv("BASE_DIR", "./metakat_worker_data")
        self.JOBS_DIR = os.getenv("JOBS_DIR", os.path.join(self.BASE_DIR, "jobs"))
        self.ENGINES_DIR = os.getenv("ENGINES_DIR", os.path.join(self.BASE_DIR, "engines"))

        self.POLLING_INTERVAL = int(os.getenv("POLLING_INTERVAL", "5"))

        self.CLEANUP_JOB_DIR = self._env_bool("CLEANUP_JOB_DIR", False)
        self.CLEANUP_OLD_ENGINES = self._env_bool("CLEANUP_OLD_ENGINES", False)

        # LOGGING configuration
        ################################################################################################################
        self.LOGGING_CONSOLE_LEVEL = os.getenv("LOGGING_CONSOLE_LEVEL", logging.INFO)
        self.LOGGING_FILE_LEVEL = os.getenv("LOGGING_FILE_LEVEL", logging.INFO)
        self.LOGGING_DIR = os.getenv("LOGGING_DIR", os.path.join(self.BASE_DIR, "logs"))
        self.LOGGING_CONFIG = {
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
                    'filename': os.path.join(self.LOGGING_DIR, f'server.log')
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

    def create_dirs(self):
        os.makedirs(self.JOBS_DIR, exist_ok=True)
        os.makedirs(self.ENGINES_DIR, exist_ok=True)
        os.makedirs(self.LOGGING_DIR, exist_ok=True)


config = Config()

