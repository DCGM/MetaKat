import argparse
import json
import logging
import sys
import time

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids-jsonl', required=True, type=str)
    parser.add_argument('--output-mapping-file', required=True, type=str)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('CREATE MAPPING - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    mapping = {}

    with open(args.ids_jsonl) as f:
        ids_json_lines = f.readlines()
        for i, line in enumerate(ids_json_lines):
            key, val = list(json.loads(line).items())[0]
            if key == 'uuid:1c6c2490-0167-11ea-af21-005056827e52':
                key = 'uuid:21740130-06fb-11ea-baca-005056825209'
            page_ids = process_instance(val)
            for page_id in page_ids:
                if page_id != key:
                    mapping[page_id] = key
            if (i + 1) % 5000 == 0:
                logger.info(f'Processing json lines: {i + 1}/{len(ids_json_lines)}')
        logger.info(f'Json lines processed: {i + 1}/{len(ids_json_lines)}')

    with open(args.output_mapping_file, 'w') as f:
        for page_id, doc_id in mapping.items():
            f.write(f'{page_id} {doc_id}\n')


def process_instance(val):
    if isinstance(val, dict):
        page_ids = []
        for key, val in val.items():
            page_ids += process_instance(val)
        return page_ids
    elif isinstance(val, list):
        return val


if __name__ == '__main__':
    main()
