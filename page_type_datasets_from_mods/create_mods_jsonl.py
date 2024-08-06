import argparse
import logging
import sys
import time

from tools.mods_helper import get_mods_jsonl

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mods-dir', required=True, type=str)
    parser.add_argument('--output-jsonl-file', required=True, type=str)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('CREATE MODS TREE - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    mods_jsonl = get_mods_jsonl(args.mods_dir)
    with open(args.output_jsonl_file, 'w') as f:
        for line in mods_jsonl:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
