import os
import json
import functools
import socket
import subprocess
import tempfile
import shutil
import time
from dataclasses import replace
from pathlib import Path
import logging.config

from metakat.worker.docapi.config import Config, LoggingConfig

import argparse
from typing import Optional

from doc_worker.doc_worker_wrapper import DocWorkerWrapper, WorkerResponse
from doc_api.api.schemas.base_objects import Job
import doc_api

import metakat

from metakat.process_batch import process_batch
from metakat.engine_config import prepare_engine_config, require_config_mapping
from metakat.common.vlm.local_server import local_model_locations
from metakat.worker.docapi.job_summary import render_log, render_text, summarize_job


logger = logging.getLogger(__name__)


class MetakatWorker(DocWorkerWrapper):
    # MetaKat's processing settings, with Config's defaults; from_config() passes
    # Config's ALLOWED_IMAGE_EXTENSIONS, STORE_METAKAT_PDF and STORE_MODS_OVERVIEW.
    allowed_image_extensions = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    store_metakat_pdf = False
    store_mods_overview = False

    def __init__(self, *args, allowed_image_extensions=None, store_metakat_pdf: bool = False,
                 store_mods_overview: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if allowed_image_extensions is not None:
            self.allowed_image_extensions = frozenset(allowed_image_extensions)
        self.store_metakat_pdf = store_metakat_pdf
        self.store_mods_overview = store_mods_overview

    def process_job(self, job: Job, images_dir: str, result_dir: str, alto_dir: Optional[str] = None,
                    page_xml_dir: Optional[str] = None, meta_file: Optional[str] = None,
                    engine_dir: Optional[str] = None) -> WorkerResponse:
        """Run MetaKat on the job, then add the job summary to the response as its final logs.

        The wrapper sends everything logged meanwhile into the job's worker.log.
        """
        started = time.monotonic()
        response = self._run_metakat(job, images_dir, result_dir, alto_dir, page_xml_dir, meta_file, engine_dir)
        return self._add_summary(job, response, job_dir=str(Path(result_dir).parent),
                                 duration_s=time.monotonic() - started)

    def _run_metakat(self,
                    job: Job,
                    images_dir: str,
                    result_dir: str,
                    alto_dir: Optional[str] = None,
                    page_xml_dir: Optional[str] = None,
                    meta_file: Optional[str] = None,
                    engine_dir: Optional[str] = None) -> WorkerResponse:
        """Run MetaKat's pipeline (process_batch) on the job's images and ALTO, writing the result into result_dir.

        Args:
            job: The job object containing job metadata
            images_dir: Directory path containing the downloaded images
            result_dir: Directory path where processing results should be saved
            alto_dir: Optional directory path containing ALTO XML files
            page_xml_dir: Optional directory path containing PAGE XML files
            meta_file: Optional path to the metadata JSON envelope, or to a
                plain ProArc JSON, which is accepted as proarc_json
            engine_dir: Optional directory path containing engine files
            
        Returns:
            WorkerResponse indicating success or failure
        """
        try:
            if alto_dir is None:
                logger.error("ALTO files are required")
                return WorkerResponse.fail(final_log="ALTO files are required",
                                           final_log_user="ALTO files are required for this engine.")
            
            # Create temporary directory with symlinks to images and ALTO files
            tmp_batch_dir = tempfile.mkdtemp(prefix="metakat_batch_")
            
            try:
                logger.info(f"Creating temporary batch directory: {tmp_batch_dir}")

                ordered_images = sorted(job.images, key=lambda img: img.order)
                ordered_image_filenames = [img.name for img in ordered_images]

                # Create symlinks for image files
                images_path = Path(images_dir)
                for image_file in images_path.iterdir():
                    if image_file.name not in ordered_image_filenames:
                        return WorkerResponse.fail(final_log=f"Image file not listed in job images: {image_file.name}",
                                                   final_log_user="An input image is not listed in the job definition.")
                    ext = image_file.suffix.lower()
                    if ext not in self.allowed_image_extensions:
                        return WorkerResponse.fail(final_log=f"Image file with unsupported extension found: {image_file.name}, "
                                                   f"allowed extensions are: {', '.join(sorted(self.allowed_image_extensions))}",
                                                   final_log_user="An input image has an unsupported extension.")
                    if image_file.is_file():
                        symlink_path = Path(tmp_batch_dir) / image_file.name
                        symlink_path.symlink_to(image_file.resolve())
                
                # Create symlinks for ALTO files
                alto_path = Path(alto_dir)
                for alto_file in alto_path.iterdir():
                    if alto_file.is_file():
                        symlink_path = Path(tmp_batch_dir) / alto_file.name
                        symlink_path.symlink_to(alto_file.resolve())

                metadata = self._load_metadata_envelope(meta_file)
                if engine_dir is None:
                    raise ValueError("Downloaded engine directory is required")
                # A job may choose or replace an API model, but a local model
                # is part of the engine: its settings start a process here.
                local_models = local_model_locations(metadata["engine_config_override"])
                if local_models:
                    raise ValueError(
                        "A job's engine configuration override must not set up a local model: "
                        + ", ".join(local_models)
                    )
                engine_config = prepare_engine_config(
                    require_config_mapping(
                        job.engine_definition,
                        "Job engine definition",
                    ),
                    override=metadata["engine_config_override"],
                    base_dir=engine_dir,
                    require_within_base=True,
                )

                process_batch(
                    batch_dir=tmp_batch_dir,
                    engine_config=engine_config,
                    metakat_data=metadata["metakat_json"],
                    proarc_data=metadata["proarc_json"],
                    ordered_image_filenames=ordered_image_filenames,
                    engine_name=job.engine_name,
                    engine_version=job.engine_version,
                    engine_dir=engine_dir,
                    output_metakat_json=os.path.join(result_dir, "metakat.json"),
                    output_mods_dir=os.path.join(result_dir, "mods"),
                    # For inspection only, so kept beside result.zip like the
                    # PDF rather than uploaded with the result.
                    output_mods_overview=(
                        os.path.join(Path(result_dir).parent, "metakat.mods.txt")
                        if self.store_mods_overview
                        else None
                    ),
                    output_metakat_pdf=(
                        os.path.join(Path(result_dir).parent, "result.pdf")
                        if self.store_metakat_pdf
                        else None
                    ),
                    allowed_image_extensions=set(self.allowed_image_extensions))

                return WorkerResponse.ok()
            
            finally:
                # Clean up temporary directory
                if os.path.exists(tmp_batch_dir):
                    logger.info(f"Cleaning up temporary batch directory: {tmp_batch_dir}")
                    shutil.rmtree(tmp_batch_dir)
            
        except Exception as e:
            logger.exception("MetakatWorker processing failed")
            return WorkerResponse.error(final_log="MetakatWorker processing failed",
                                        final_log_user="Processing failed. Please try again later.", exception=e)

    def _add_summary(self, job: Job, response: WorkerResponse,
                     job_dir: str, duration_s: float) -> WorkerResponse:
        """Add the job summary to the response as its final logs, which the wrapper sends to DocAPI.

        The summary is built from the job, the response, the job's directory (its
        input, result and meta.json) and the engine definition. final_log gets the
        full summary, for admins; final_log_user gets the summary without the error
        details, followed by the response's own message for the user. The status is
        kept. The summary also goes to the worker log. If the summary cannot be
        built, the response is returned unchanged, so a broken summary never changes
        the outcome.
        """
        try:
            metadata = None
            meta_file = Path(job_dir) / "meta.json"
            if meta_file.is_file():
                try:
                    metadata = self._load_metadata_envelope(str(meta_file))
                except Exception:
                    logger.warning("Job metadata could not be read for the job summary", exc_info=True)
            definition = job.engine_definition if isinstance(job.engine_definition, dict) else {}
            # The wrapper adds diagnostics from exception/error_adapter_response.
            # Summarize the message here so those details appear only once.
            summary = summarize_job(
                job, replace(response, exception=None, error_adapter_response=None), duration_s,
                job_dir=Path(job_dir),
                has_chapter_stage="chapter" in definition,
                worker=worker_description(),
                metadata=metadata,
            )
            logger.info(render_log(summary))
            user_report = render_text(summary, with_error=False)
            if response.final_log_user:
                user_report += "\n" + response.final_log_user
            return replace(response, final_log=render_text(summary), final_log_user=user_report)
        except Exception:
            # A broken summary must not change the processing outcome.
            logger.exception("Could not prepare the final job summary")
            return response

    @staticmethod
    def _load_metadata_envelope(meta_file: Optional[str]) -> dict:
        keys = {
            "metakat_json",
            "proarc_json",
            "engine_config_override",
        }
        if meta_file is None:
            return {key: None for key in keys}
        with open(meta_file, "r", encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("Job metadata envelope must be a JSON object")
        if value and not value.keys() & keys:
            # Jobs that predate the envelope send the plain ProArc
            # packageInfo.json as the meta file. It carries none of the
            # envelope keys - only its own "type" and "objects" - so treat the
            # whole document as proarc_json instead of rejecting those as
            # unknown envelope keys. An empty object stays an empty envelope:
            # a ProArc JSON always has at least its own required keys, so {}
            # cannot be one.
            logger.info(
                "Job metadata has none of the envelope keys; processing the "
                "whole document as a plain ProArc JSON"
            )
            return {
                key: (value if key == "proarc_json" else None)
                for key in keys
            }
        unknown = value.keys() - keys
        if unknown:
            raise ValueError(
                "Unknown job metadata envelope key(s): "
                + ", ".join(sorted(unknown))
            )
        result = {key: value.get(key) for key in keys}
        for key, item in result.items():
            if item is not None and not isinstance(item, dict):
                raise ValueError(f"Job metadata {key!r} must be an object or null")
        return result


@functools.lru_cache(maxsize=None)
def worker_description() -> str:
    """The MetaKat and DocAPI commits the worker runs and its host, read once, when the worker starts."""
    def commit(directory: Path) -> str:
        try:
            run = functools.partial(subprocess.run, cwd=directory, capture_output=True, text=True, timeout=10)
            head = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
            dirty = run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
            return (head + ("+dirty" if dirty else "")) if head else "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    return (f"metakat {commit(Path(metakat.__file__).parent)}, "
            f"docapi {commit(Path(doc_api.__file__).parent)}, {socket.gethostname()}")


def _extension_set(value: str) -> set:
    """Parse a comma-separated extension list the way the environment variable is."""
    extensions = {item.strip() for item in value.split(",") if item.strip()}
    if not extensions:
        raise argparse.ArgumentTypeError("expected at least one extension")
    return extensions


def _setting_text(value) -> str:
    """A command-line value as Config reads it from the environment."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (set, frozenset)):
        return ",".join(sorted(value))
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--api-url",
        help="Base URL of the DocAPI server"
    )
    parser.add_argument(
        "--api-key",
        help="API worker key for authentication; prefer WORKER_KEY_FILE, which keeps the key off the command line"
    )
    
    # Directory arguments
    parser.add_argument(
        "--base-dir",
        help="Base directory for jobs and engines (creates subdirectories)"
    )
    parser.add_argument(
        "--jobs-dir",
        help="Directory for job data (overrides base-dir/jobs)"
    )
    parser.add_argument(
        "--engines-dir",
        help="Directory for engine files (overrides base-dir/engines)"
    )
    
    # Worker configuration
    parser.add_argument(
        "--polling-interval",
        type=float,
        help="Time in seconds to wait between job requests"
    )
    parser.add_argument(
        "--cleanup-job-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Remove job directory after successful processing"
    )
    parser.add_argument(
        "--cleanup-old-engines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Remove old engine versions when downloading new ones"
    )
    parser.add_argument(
        "--store-metakat-pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store an interactive result.pdf beside result.zip"
    )
    parser.add_argument(
        "--store-mods-overview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store metakat.mods.txt, every MODS record in one file, beside result.zip"
    )
    parser.add_argument(
        "--allowed-image-extensions",
        type=_extension_set,
        metavar="EXT[,EXT...]",
        help="Comma-separated image extensions a job may contain"
    )

    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level (default: LOGGING_CONSOLE_LEVEL, or INFO)"
    )
    parser.add_argument(
        "--log-file-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for worker.log (default: LOGGING_FILE_LEVEL, or INFO)"
    )
    parser.add_argument(
        "--logging-dir",
        help="Directory for worker.log (overrides base-dir/logs)"
    )
    
    args = parser.parse_args()

    # An option overrides the environment when it was supplied, which is not the
    # same question as whether its value is truthy: 0 is a legitimate polling
    # interval and would be discarded by `or`. Config takes them as text, as in
    # the environment, and checks them with the rest.
    overrides = {}
    for argument_name, setting in (
        ("api_url", "API_URL"),
        ("base_dir", "BASE_DIR"),
        ("jobs_dir", "JOBS_DIR"),
        ("engines_dir", "ENGINES_DIR"),
        ("polling_interval", "POLLING_INTERVAL"),
        ("cleanup_job_dir", "CLEANUP_JOB_DIR"),
        ("cleanup_old_engines", "CLEANUP_OLD_ENGINES"),
        ("store_metakat_pdf", "STORE_METAKAT_PDF"),
        ("store_mods_overview", "STORE_MODS_OVERVIEW"),
        ("allowed_image_extensions", "ALLOWED_IMAGE_EXTENSIONS"),
        ("log_level", "LOGGING_CONSOLE_LEVEL"),
        ("log_file_level", "LOGGING_FILE_LEVEL"),
        ("logging_dir", "LOGGING_DIR"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            overrides[setting] = _setting_text(value)
    if args.api_key is not None:
        # A key given here replaces a key file from the environment as well.
        overrides.update(WORKER_KEY=args.api_key, WORKER_KEY_FILE="")
    logging_overrides = {name: overrides.pop(name)
                         for name in ("LOGGING_DIR", "LOGGING_CONSOLE_LEVEL", "LOGGING_FILE_LEVEL")
                         if name in overrides}
    try:
        config = Config(overrides)
        logging_config = LoggingConfig(config.BASE_DIR, logging_overrides)
    except ValueError as error:
        parser.error(str(error))

    # Directories and logging are set up from the final settings.
    config.create_dirs()
    logging_config.create_dirs()
    logging.config.dictConfig(logging_config.LOGGING_CONFIG)

    if config.STORE_METAKAT_PDF and config.CLEANUP_JOB_DIR:
        logger.warning(
            "STORE_METAKAT_PDF and CLEANUP_JOB_DIR are both enabled; "
            "the locally stored result.pdf will be removed after upload"
        )
    if config.STORE_MODS_OVERVIEW and config.CLEANUP_JOB_DIR:
        logger.warning(
            "STORE_MODS_OVERVIEW and CLEANUP_JOB_DIR are both enabled; "
            "the locally stored metakat.mods.txt will be removed after upload"
        )

    worker = MetakatWorker.from_config(config)

    logger.info(f"Starting MetakatWorker connecting to {config.API_URL}")
    logger.info(f"Worker: {worker_description()}")
    logger.info(f"API timeouts: connect {config.API_CONNECT_TIMEOUT_SECONDS:g} s, "
                f"read {config.API_READ_TIMEOUT_SECONDS:g} s")
    logger.info(f"Base directory: {config.BASE_DIR}")
    logger.info(f"Jobs directory: {config.JOBS_DIR or os.path.join(config.BASE_DIR, 'jobs')}")
    logger.info(f"Engines directory: {config.ENGINES_DIR or os.path.join(config.BASE_DIR, 'engines')}")
    logger.info(f"Log directory: {logging_config.LOGGING_DIR}")
    logger.info(f"Job history: {config.JOB_HISTORY_PATH or 'off'}")
    logger.info(f"Store MetaKat PDF: {config.STORE_METAKAT_PDF}")
    logger.info(f"Store MODS overview: {config.STORE_MODS_OVERVIEW}")
    if config.MAIL_SERVER:
        logger.info(f"Mail: {config.MAIL_SENDER_MAIL} via {config.MAIL_SERVER}:{config.MAIL_PORT} "
                    f"({config.MAIL_SECURITY}), receivers: {', '.join(config.MAIL_RECEIVER_MAILS)}")
        logger.info(f"Mail limits: {config.MAIL_SEND_RECEIVER_LIMIT} per subject and receiver per "
                    f"{config.MAIL_SEND_RECEIVER_WINDOW_SECONDS:g} s, {config.MAIL_SEND_SERVER_LIMIT} in total per "
                    f"{config.MAIL_SEND_SERVER_WINDOW_SECONDS:g} s (0 = no limit)")
        logger.info(f"ALIVE mail: {'at startup and ' if config.ALIVE_MAIL_STARTUP else ''}"
                    f"every {config.ALIVE_MAIL_INTERVAL_SECONDS:g} s (0 = none); "
                    f"DocAPI outage mail after {config.OUTAGE_GRACE_SECONDS:g} s idle, "
                    f"reminders every {config.OUTAGE_MAIL_INTERVAL_SECONDS:g} s")
    else:
        logger.info("Mail: not set up (MAIL_SERVER), the worker sends no mail")

    worker.start()


if __name__ == "__main__":
    main()
