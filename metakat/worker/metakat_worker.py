
from metakat.worker.config import config
import logging, logging.config
logging.config.dictConfig(config.LOGGING_CONFIG)

import argparse
from typing import Optional

from doc_worker.doc_worker_wrapper import DocWorkerWrapper, WorkerResponse
from doc_api.api.schemas.base_objects import Job
from doc_api.connector import Connector


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
            

            return WorkerResponse.ok()
            
        except Exception as e:
            logger.exception("MetakatWorker processing failed")
            return WorkerResponse.fail("MetakatWorker processing failed", exception=e)


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
