import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union
from uuid import UUID

from pydantic import ValidationError

from metakat.io_parsers.parser_mods import parse_mods
from metakat.schemas.base_objects import ProarcIO

logger = logging.getLogger(__name__)

_PID_PREFIX = "uuid:"


def _pid_to_uuid(pid: str) -> Optional[UUID]:
    # Pid's pattern (StringConstraints pattern=r"^uuid:.*") is enforced by
    # ProarcIO validation, so the prefix is always present by the time we get
    # here. What the pattern does not guarantee is that the remainder is a
    # real UUID.
    try:
        return UUID(pid[len(_PID_PREFIX):])
    except ValueError:
        return None


def parse_proarc_json(data: Mapping[str, Any]) -> Optional[ProarcIO]:
    """Read a ProArc packageInfo.json dict into ProarcIO, best effort.

    This is the only gate for ProArc input: nothing else should validate a
    ProArc document into the pydantic model directly, because doing so skips
    both the pid-to-id derivation and the MODS parsing that every consumer of
    ProarcIO expects to have happened.

    Reading is best effort and never raises - a ProArc document that cannot be
    read must not stop the batch from being processed without it. Returns None
    when nothing usable could be read, so that callers hand the engines either
    a package with something in it or no package at all, never an empty one.

    An object is usable when its pid yields a valid UUID id; that identity is
    the minimum a record needs to be placed in the hierarchy at all, so a
    record without it is dropped rather than partially read. An object whose
    MODS cannot be parsed is kept with its identity and no catalog fields.
    """
    try:
        package = ProarcIO.model_validate(data)
    except ValidationError as error:
        logger.warning(
            "ProArc JSON is malformed, so reading it cannot be attempted at "
            "all; continuing without ProArc data: %s",
            error,
        )
        return None

    usable = []
    for obj in package.objects:
        object_id = _pid_to_uuid(obj.pid)
        if object_id is None:
            logger.warning(
                "Skipping unusable ProArc object: pid %r carries no valid "
                "UUID, so the record cannot be identified",
                obj.pid,
            )
            continue
        obj.id = object_id
        try:
            parsed = parse_mods(obj.metadata)
        except Exception as error:
            logger.warning(
                "Could not read the MODS metadata of ProArc object %s; "
                "keeping the record with its identity only: %s",
                obj.pid,
                error,
            )
        else:
            for key, value in parsed.items():
                setattr(obj, key, value)
        usable.append(obj)

    if not usable:
        logger.warning(
            "No usable record could be read from the ProArc JSON; continuing "
            "without ProArc data"
        )
        return None

    if len(usable) != len(package.objects):
        logger.warning(
            "Read %d of %d ProArc object(s); the rest were unusable",
            len(usable),
            len(package.objects),
        )
    package.objects = usable
    return package


def parse_proarc_json_file(path: Union[str, Path]) -> Optional[ProarcIO]:
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
    if package is None:
        # The warnings above already say what could not be read; nothing
        # usable came out, so there is no package to write.
        logger.error('Nothing usable could be read from %s', args.package_info_file)
        return 1

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(package.model_dump_json(indent=2))

    logger.info(f'Wrote parsed ProArc package ({len(package.objects)} objects) to {args.output_file}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
