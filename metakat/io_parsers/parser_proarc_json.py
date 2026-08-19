import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Union
from uuid import UUID

from metakat.io_parsers.parser_mods import parse_mods
from metakat.schemas.base_objects import ProarcIO

logger = logging.getLogger(__name__)

_PID_PREFIX = "uuid:"


def _pid_to_uuid(pid: str) -> UUID:
    # Pid is validated (StringConstraints pattern=r"^uuid:.*") to always
    # carry this prefix before the UUID itself.
    return UUID(pid[len(_PID_PREFIX):])


def parse_proarc_json(data: dict) -> ProarcIO:
    """Validate a ProArc packageInfo.json dict into ProarcIO, with each object's
    id derived from its pid and its parsed fields (see parser_mods.parse_mods)
    filled in from its MODS metadata.
    """
    package = ProarcIO.model_validate(data)
    for obj in package.objects:
        obj.id = _pid_to_uuid(obj.pid)
        parsed = parse_mods(obj.metadata)
        for key, value in parsed.items():
            setattr(obj, key, value)
    return package


def parse_proarc_json_file(path: Union[str, Path]) -> ProarcIO:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_proarc_json(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-info-file', required=True, type=str)
    parser.add_argument('--output-file', required=True, type=str)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('PARSE PROARC JSON - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    package = parse_proarc_json_file(args.package_info_file)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(package.model_dump_json(indent=2))

    logger.info(f'Wrote parsed ProArc package ({len(package.objects)} objects) to {args.output_file}')


if __name__ == '__main__':
    main()
