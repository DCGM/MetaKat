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
    # ProarcIO validates the "uuid:" prefix too (Pid's StringConstraints
    # pattern), but the id has to be derived before there is a validated
    # model to read it off, so the prefix is checked here rather than assumed.
    # The pattern alone would not be enough either: it does not say that what
    # follows the prefix is a real UUID.
    if not pid.startswith(_PID_PREFIX):
        return None
    try:
        return UUID(pid[len(_PID_PREFIX):])
    except ValueError:
        return None


def _read_object(raw_object: Mapping[str, Any]) -> Optional[dict]:
    """One raw object plus its derived id and parsed MODS fields, or None when
    its pid yields no UUID. Still a plain dict: ProarcIO validates it later."""
    pid = raw_object.get("pid")
    object_id = _pid_to_uuid(pid) if isinstance(pid, str) else None
    if object_id is None:
        return None

    read = dict(raw_object)
    read["id"] = object_id
    metadata = raw_object.get("metadata")
    if isinstance(metadata, str):
        try:
            read.update(parse_mods(metadata))
        except Exception as error:
            logger.warning(
                "Could not read the MODS metadata of ProArc object %s; "
                "keeping the record with its identity only: %s",
                pid,
                error,
            )
    return read


def _read_document(data: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """`data` with every object's id and parsed MODS fields filled in.

    Returns `data` untouched when the raw shape cannot be walked, so that
    ProarcIO reports precisely what is wrong with it instead of this function
    guessing. Returns None when an object's pid yields no UUID.
    """
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list) or not all(
        isinstance(raw_object, Mapping) for raw_object in raw_objects
    ):
        return data

    read_objects = []
    for raw_object in raw_objects:
        read = _read_object(raw_object)
        if read is None:
            # Every record has to be identifiable or none of them can be
            # trusted: a hierarchy missing one of its nodes still looks like a
            # complete one to the engines downstream, which would then read it
            # as a different structure than the catalog actually describes.
            logger.warning(
                "ProArc object pid %r carries no valid UUID; discarding the "
                "whole document rather than passing down a hierarchy with a "
                "record missing",
                raw_object.get("pid"),
            )
            return None
        read_objects.append(read)
    return {**data, "objects": read_objects}


def parse_proarc_json(data: Mapping[str, Any]) -> Optional[ProarcIO]:
    """Read a ProArc packageInfo.json dict into ProarcIO, best effort.

    This is the only gate for ProArc input: nothing else may validate a ProArc
    document into the pydantic model directly, because doing so skips both the
    pid-to-id derivation and the MODS parsing that every consumer of ProarcIO
    expects to have happened.

    The document is assembled first and validated once, as a whole, so that
    ProarcIO actually guards what comes out of here. Filling fields in after
    validation would not: pydantic does not check assignment, so whatever the
    MODS parser produced would reach the engines unchecked.

    Reading is best effort and never raises - a ProArc document that cannot be
    read must not stop the batch from being processed without it. Returns None
    when nothing usable could be read, so that callers hand the engines either
    a package with something in it or no package at all, never an empty one.

    Every object must have a pid that yields a valid UUID. That identity is
    what places a record in the hierarchy, and a hierarchy with a record
    missing still looks complete further down the pipeline, so one unusable
    object discards the whole document. An object whose MODS cannot be parsed
    keeps its identity and loses only its catalog fields.
    """
    if not isinstance(data, Mapping):
        logger.warning(
            "ProArc JSON is malformed, so reading it cannot be attempted at "
            "all; continuing without ProArc data: expected an object, got %s",
            type(data).__name__,
        )
        return None

    document = _read_document(data)
    if document is None:
        return None

    try:
        package = ProarcIO.model_validate(document)
    except ValidationError as error:
        logger.warning(
            "ProArc JSON is malformed, so reading it cannot be attempted at "
            "all; continuing without ProArc data: %s",
            error,
        )
        return None

    if not package.objects:
        logger.warning(
            "No usable record could be read from the ProArc JSON; continuing "
            "without ProArc data"
        )
        return None

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
