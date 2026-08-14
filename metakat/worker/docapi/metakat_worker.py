import os
import json
import tempfile
import shutil
from pathlib import Path
import logging.config

from metakat.worker.docapi.config import config
config.create_dirs()
logging.config.dictConfig(config.LOGGING_CONFIG)

import argparse
from typing import Optional

from doc_worker.doc_worker_wrapper import DocWorkerWrapper, WorkerResponse
from doc_api.api.schemas.base_objects import Job
from doc_api.connector import Connector

from metakat.process_batch import process_batch
from metakat.engine_config import prepare_engine_config, require_config_mapping


logger = logging.getLogger(__name__)


class MetakatWorker(DocWorkerWrapper):

    def process_job(self,
                    job: Job,
                    job_log_file_handler: logging.FileHandler,
                    images_dir: str,
                    result_dir: str,
                    alto_dir: Optional[str] = None,
                    page_xml_dir: Optional[str] = None,
                    meta_file: Optional[str] = None,
                    engine_dir: Optional[str] = None) -> WorkerResponse:
        """        
        Args:
            job: The job object containing job metadata
            job_log_file_handler: File handler for logging job processing into job-specific log file
            images_dir: Directory path containing the downloaded images
            result_dir: Directory path where processing results should be saved
            alto_dir: Optional directory path containing ALTO XML files
            page_xml_dir: Optional directory path containing PAGE XML files
            meta_file: Optional path to the metadata JSON envelope
            engine_dir: Optional directory path containing engine files
            
        Returns:
            WorkerResponse indicating success or failure
        """
        root_logger = logging.getLogger()

        try:
            root_logger.addHandler(job_log_file_handler)

            if alto_dir is None:
                logger.error("ALTO files are required")
                return WorkerResponse.fail("ALTO files are required")
            
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
                        return WorkerResponse.fail(f"Image file not listed in job images: {image_file.name}")
                    ext = image_file.suffix.lower()
                    if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
                        return WorkerResponse.fail(f"Image file with unsupported extension found: {image_file.name}, "
                                                   f"allowed extensions are: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS)}")
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
                    output_metakat_json=os.path.join(result_dir, "metakat.json"),
                    output_metakat_pdf=(
                        os.path.join(Path(result_dir).parent, "result.pdf")
                        if config.STORE_METAKAT_PDF
                        else None
                    ),
                    allowed_image_extensions=config.ALLOWED_IMAGE_EXTENSIONS)

                return WorkerResponse.ok()
            
            finally:
                # Clean up temporary directory
                if os.path.exists(tmp_batch_dir):
                    logger.info(f"Cleaning up temporary batch directory: {tmp_batch_dir}")
                    shutil.rmtree(tmp_batch_dir)
            
        except Exception as e:
            logger.exception("MetakatWorker processing failed")
            return WorkerResponse.fail("MetakatWorker processing failed", exception=e)

        finally:
            root_logger.removeHandler(job_log_file_handler)
            job_log_file_handler.close()

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
        help="API worker key for authentication"
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
        action="store_true",
        help="Remove job directory after successful processing"
    )
    parser.add_argument(
        "--cleanup-old-engines",
        action="store_true",
        help="Remove old engine versions when downloading new ones"
    )
    parser.add_argument(
        "--store-metakat-pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store an interactive result.pdf beside result.zip"
    )

    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()

    config.API_URL = args.api_url or config.API_URL
    config.WORKER_KEY = args.api_key or config.WORKER_KEY
    config.BASE_DIR = args.base_dir or config.BASE_DIR
    config.JOBS_DIR = args.jobs_dir or config.JOBS_DIR
    config.ENGINES_DIR = args.engines_dir or config.ENGINES_DIR
    config.POLLING_INTERVAL = args.polling_interval or config.POLLING_INTERVAL
    config.CLEANUP_JOB_DIR = args.cleanup_job_dir or config.CLEANUP_JOB_DIR
    config.CLEANUP_OLD_ENGINES = args.cleanup_old_engines or config.CLEANUP_OLD_ENGINES
    if args.store_metakat_pdf is not None:
        config.STORE_METAKAT_PDF = args.store_metakat_pdf
    config.LOGGING_CONSOLE_LEVEL = args.log_level or config.LOGGING_CONSOLE_LEVEL

    if config.STORE_METAKAT_PDF and config.CLEANUP_JOB_DIR:
        logger.warning(
            "STORE_METAKAT_PDF and CLEANUP_JOB_DIR are both enabled; "
            "the locally stored result.pdf will be removed after upload"
        )
        
    # Validate directory arguments
    if not config.BASE_DIR and (not config.JOBS_DIR or not config.ENGINES_DIR):
        parser.error("Either --base-dir or both --jobs-dir and --engines-dir must be specified")
    
    # Create connector
    connector = Connector(api_key=config.WORKER_KEY)
    
    # Create and start worker
    worker = MetakatWorker(
        api_url=config.API_URL,
        connector=connector,
        base_dir=config.BASE_DIR,
        jobs_dir=config.JOBS_DIR,
        engines_dir=config.ENGINES_DIR,
        polling_interval=config.POLLING_INTERVAL,
        cleanup_job_dir=config.CLEANUP_JOB_DIR,
        cleanup_old_engines=config.CLEANUP_OLD_ENGINES
    )
    
    logger.info(f"Starting MetakatWorker connecting to {config.API_URL}")
    logger.info(f"Base directory: {config.BASE_DIR}")
    logger.info(f"Jobs directory: {config.JOBS_DIR}")
    logger.info(f"Engines directory: {config.ENGINES_DIR}")
    logger.info(f"Store MetaKat PDF: {config.STORE_METAKAT_PDF}")
    
    worker.start()


if __name__ == "__main__":
    main()
