import os
import tempfile
import shutil
from pathlib import Path
import logging, logging.config

from metakat.worker.config import config
config.create_dirs()
logging.config.dictConfig(config.LOGGING_CONFIG)

import argparse
from typing import Optional

from doc_worker.doc_worker_wrapper import DocWorkerWrapper, WorkerResponse
from doc_api.api.schemas.base_objects import Job
from doc_api.connector import Connector

from metakat.process_batch import process_batch


logger = logging.getLogger(__name__)


class MetakatWorker(DocWorkerWrapper):

    def process_job(self, 
                    job: Job,
                    images_dir: str,
                    results_dir: str,
                    alto_dir: Optional[str] = None,
                    page_xml_dir: Optional[str] = None,
                    meta_file: Optional[str] = None,
                    engine_dir: Optional[str] = None) -> WorkerResponse:
        """        
        Args:
            job: The job object containing job metadata
            images_dir: Directory path containing the downloaded images
            results_dir: Directory path where processing results should be saved
            alto_dir: Optional directory path containing ALTO XML files
            page_xml_dir: Optional directory path containing PAGE XML files
            meta_file: Optional path to the meta.json file
            engine_dir: Optional directory path containing engine files
            
        Returns:
            WorkerResponse indicating success or failure
        """
        try:
            if alto_dir is None:
                logger.error("ALTO files are required")
                return WorkerResponse.fail("ALTO files are required")
            
            # Create temporary directory with symlinks to images and ALTO files
            tmp_batch_dir = tempfile.mkdtemp(prefix="metakat_batch_")
            
            try:
                logger.info(f"Creating temporary batch directory: {tmp_batch_dir}")
                
                # Create symlinks for image files
                images_path = Path(images_dir)
                for image_file in images_path.iterdir():
                    if image_file.is_file():
                        symlink_path = Path(tmp_batch_dir) / image_file.name
                        symlink_path.symlink_to(image_file.resolve())
                
                # Create symlinks for ALTO files
                alto_path = Path(alto_dir)
                for alto_file in alto_path.iterdir():
                    if alto_file.is_file():
                        symlink_path = Path(tmp_batch_dir) / alto_file.name
                        symlink_path.symlink_to(alto_file.resolve())

                # Construct engine paths
                page_type_core_path = self._get_engine_path(
                    engine_dir, 'page_type_core_engine', job.engine_definition
                )
                page_type_bind_path = self._get_engine_path(
                    engine_dir, 'page_type_bind_engine', job.engine_definition
                )
                biblio_core_path = self._get_engine_path(
                    engine_dir, 'biblio_core_engine', job.engine_definition
                )
                biblio_bind_path = self._get_engine_path(
                    engine_dir, 'biblio_bind_engine', job.engine_definition
                )
                chapter_core_path = self._get_engine_path(
                    engine_dir, 'chapter_core_engine', job.engine_definition
                )
                chapter_bind_path = self._get_engine_path(
                    engine_dir, 'chapter_bind_engine', job.engine_definition
                )

                process_batch(
                    batch_dir=tmp_batch_dir,
                    proarc_json=meta_file,
                    page_type_core_engine=page_type_core_path,
                    page_type_bind_engine=page_type_bind_path,
                    biblio_core_engine=biblio_core_path,
                    biblio_bind_engine=biblio_bind_path,
                    chapter_core_engine=chapter_core_path,
                    chapter_bind_engine=chapter_bind_path,
                    output_metakat_json=os.path.join(results_dir, "metakat.json"))

                return WorkerResponse.ok()
            
            finally:
                # Clean up temporary directory
                if os.path.exists(tmp_batch_dir):
                    logger.info(f"Cleaning up temporary batch directory: {tmp_batch_dir}")
                    shutil.rmtree(tmp_batch_dir)
            
        except Exception as e:
            logger.exception("MetakatWorker processing failed")
            return WorkerResponse.fail("MetakatWorker processing failed", exception=e)


    def _get_engine_path(self, engine_dir: str, engine_key: str, engine_definition: dict) -> str:
            """
            Construct the full path to an engine based on the engine definition key.
            
            The key format is expected to be: {category}_{engine_type}_engine
            For example: 'page_type_core_engine', 'biblio_bind_engine', 'chapter_core_engine'
            
            Args:
                engine_dir: Base engine directory
                engine_key: Key from engine_definition (e.g., 'page_type_core_engine')
                engine_definition: Dictionary containing engine definitions
                
            Returns:
                Full path to the engine directory
            """
            # Parse the key to extract category and engine_type
            # Remove '_engine' suffix and split
            key_parts = engine_key.replace('_engine', '').split('_')
            
            # Last part is the engine type (core/bind)
            engine_type = key_parts[-1]
            
            # Everything before that is the category
            category = '_'.join(key_parts[:-1])
            
            # Get the engine name from the definition
            engine_name = engine_definition[engine_key]
            
            return os.path.join(engine_dir, category, engine_type, engine_name)


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
    config.LOGGING_CONSOLE_LEVEL = args.log_level or config.LOGGING_CONSOLE_LEVEL
        
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
    
    worker.start()


if __name__ == "__main__":
    main()
