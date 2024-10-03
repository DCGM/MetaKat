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

    with open(args.ids_jsonl) as f:
        ids_json_lines = f.readlines()
        neighbour_lines = []
        for i, line in enumerate(ids_json_lines):
            key, val = list(json.loads(line).items())[0]
            neighbour_lines += process_instance(key, val)
            if (i + 1) % 5000 == 0:
                logger.info(f'Processing json lines: {i + 1}/{len(ids_json_lines)}')
        logger.info(f'Json lines processed: {i + 1}/{len(ids_json_lines)}')

    with open(args.output_mapping_file, 'w') as f:
        f.writelines(neighbour_lines)


def process_instance(key, val):
    if isinstance(val, dict):
        page_ids = []
        for key2, val in val.items():
            page_ids += process_instance(key2, val)
        return page_ids
    elif isinstance(val, list):
        neighbour_lines = []
        try:
            val.remove(key)
        except ValueError:
            pass
        for i in range(len(val)):
            current_page = val[i]
            previous_page_count = i
            next_page_count = len(val) - i - 1
            if i == 0:
                previous_page = 'None'
            else:
                previous_page = val[i - 1]
                previous_page = f'{previous_page}.jpg'
            if i == len(val) - 1:
                next_page = 'None'
            else:
                next_page = val[i + 1]
                next_page = f'{next_page}.jpg'
            neighbour_lines.append(f'{current_page}.jpg {previous_page} {previous_page_count} {next_page} {next_page_count}\n')
        return neighbour_lines


if __name__ == '__main__':
    main()
