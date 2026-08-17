import argparse
import json
import logging
import os.path
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Tuple, List, Optional, Set
from uuid import uuid4, UUID
import xml.etree.ElementTree as ET

from natsort import natsorted
from PIL import Image

from metakat.chapter.engines.bind.definitions import load_chapter_bind_engine
from metakat.page_number.engines.bind.definitions import (
    load_page_number_bind_engine,
)
from metakat.page_type.engines.bind.definitions import load_page_type_bind_engine
from metakat.biblio.engines.bind.definitions import load_biblio_bind_engine
from metakat.biblio.engines.core.definitions import check_biblio_core_engine
from metakat.chapter.engines.core.definitions import check_chapter_core_engine
from metakat.page_number.engines.core.definitions import (
    check_page_number_core_engine,
)
from metakat.page_type.engines.core.definitions import check_page_type_core_engine
from metakat.engine_config import (
    load_config_file,
    prepare_engine_config,
    require_config_mapping,
)
from metakat.logging_utils import redacted_for_logging

from metakat.schemas.base_objects import (
    MetakatIO,
    MetakatPage,
    MetakatPageDimensions,
    ProarcIO,
)
from metakat.tools.create_interactive_pdf import create_interactive_pdf

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-dir', type=str, required=True)
    parser.add_argument('--metakat-json', type=str)
    parser.add_argument('--proarc-json', type=str)
    parser.add_argument(
        '--engine-config',
        type=str,
        required=True,
        help='Path to the complete YAML or JSON engine configuration',
    )
    parser.add_argument(
        '--engine-config-override',
        type=str,
        help='Optional YAML or JSON configuration override',
    )
    parser.add_argument(
        '--set',
        dest='engine_config_assignments',
        nargs=2,
        action='append',
        metavar=('PATH', 'VALUE'),
        help='Override a colon-delimited engine config path; repeatable',
    )

    parser.add_argument('--allowed-image-extensions', type=str, nargs='*', default=['.jpg', '.jpeg', '.png', '.tif', '.tiff'])

    parser.add_argument('--output-metakat-json', type=str, help='Path to output Metakat JSON file')
    parser.add_argument('--output-metakat-pdf', type=str, help='Path to output interactive MetaKat PDF file')

    parser.add_argument('--logging-level', default=logging.INFO)

    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - PROCESS BATCH - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    engine_config_path = Path(args.engine_config).resolve()
    base_config = load_config_file(engine_config_path)
    override = (
        load_config_file(args.engine_config_override)
        if args.engine_config_override is not None
        else None
    )
    engine_config = prepare_engine_config(
        base_config,
        override=override,
        assignments=args.engine_config_assignments,
        base_dir=engine_config_path.parent,
    )

    process_batch(
        batch_dir=args.batch_dir,
        engine_config=engine_config,
        metakat_data=_load_json_file(args.metakat_json),
        proarc_data=_load_json_file(args.proarc_json),
        output_metakat_json=args.output_metakat_json,
        output_metakat_pdf=args.output_metakat_pdf,
        allowed_image_extensions=set(args.allowed_image_extensions)
    )
    

def process_batch(
    batch_dir: str,
    engine_config: Mapping[str, Any],
    metakat_data: Optional[Mapping[str, Any]] = None,
    proarc_data: Optional[Mapping[str, Any]] = None,
    ordered_image_filenames: Optional[List] = None,
    output_metakat_json: Optional[str] = None,
    output_metakat_pdf: Optional[str] = None,
    allowed_image_extensions: Optional[Set] = None,
) -> MetakatIO:
    """
    Process a batch directory and return the processed MetakatIO object.
    
    Args:
        batch_dir: Path to the batch directory
        engine_config: Final merged and path-resolved pipeline configuration
        metakat_data: Decoded input MetaKat JSON object
        proarc_data: Decoded input ProARC JSON object
        ordered_image_filenames: List of ordered image filenames in batch_dir (defaults to natsorted image files)
        output_metakat_json: Path to output Metakat JSON file
        output_metakat_pdf: Path to output interactive MetaKat PDF file
        allowed_image_extensions: Set of allowed image file extensions

    Returns:
        Processed MetakatIO object
    """
    if allowed_image_extensions is None:
        allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    pipeline_config = require_config_mapping(
        engine_config,
        "Pipeline engine config",
    )
    logger.info(
        "Starting MetaKat processing with engine pipeline configuration:\n%s",
        redacted_for_logging(pipeline_config),
    )

    _preflight_engine_requirements(pipeline_config)

    metakat_io, proarc_io = init_io(
        batch_dir=batch_dir,
        metakat_data=metakat_data,
        proarc_data=proarc_data,
        ordered_image_filenames=ordered_image_filenames,
        allowed_image_extensions=allowed_image_extensions
    )

    page_number = _engine_pair(pipeline_config, "page_number")
    if page_number is not None:
        page_number_bind_engine_obj = load_page_number_bind_engine(
            page_number["bind"],
            page_number["core"],
        )
        metakat_io = page_number_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io,
        )

    page_type = _engine_pair(pipeline_config, "page_type")
    if page_type is not None:
        page_type_bind_engine_obj = load_page_type_bind_engine(
            page_type["bind"],
            page_type["core"],
        )
        metakat_io = page_type_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    biblio = _engine_pair(pipeline_config, "biblio")
    if biblio is not None:
        biblio_bind_engine_obj = load_biblio_bind_engine(
            biblio["bind"],
            biblio["core"],
        )
        metakat_io = biblio_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    chapter = _engine_pair(pipeline_config, "chapter")
    if chapter is not None:
        chapter_bind_engine_obj = load_chapter_bind_engine(
            chapter["bind"],
            chapter["core"],
        )
        metakat_io = chapter_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    logger.info("")
    MetakatIO.model_validate_json(json.dumps(metakat_io.model_dump(mode="json")))
    logger.info("MetakatIO has been successfully validated")

    if output_metakat_pdf is not None:
        create_interactive_pdf(
            batch_dir,
            metakat_io,
            output_metakat_pdf,
        )

    if output_metakat_json is not None:
        with open(output_metakat_json, 'w') as f:
            json.dump(metakat_io.model_dump(mode="json"), f, indent=4, ensure_ascii=False)
        logger.info(f"MetakatIO saved to {output_metakat_json}")
    
    return metakat_io


def init_io(batch_dir: str,
            metakat_data: Optional[Mapping[str, Any]] = None,
            proarc_data: Optional[Mapping[str, Any]] = None,
            batch_id: UUID = uuid4(),
            ordered_image_filenames: Optional[List] = None,
            allowed_image_extensions: Optional[Set] = None) -> Tuple[MetakatIO, ProarcIO]:
    if allowed_image_extensions is None:
        allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    if metakat_data is not None:
        metakat_io = MetakatIO.model_validate(metakat_data)
    else:
        metakat_io = MetakatIO(batch_id=batch_id)

    if proarc_data is not None:
        proarc_io = ProarcIO.model_validate(proarc_data)
    else:
        proarc_io = None

    if metakat_io.page_to_image_mapping is None:
        metakat_io.page_to_image_mapping = {}
    if metakat_io.page_to_alto_mapping is None:
        metakat_io.page_to_alto_mapping = {}
    if metakat_io.page_to_xml_mapping is None:
        metakat_io.page_to_xml_mapping = {}

    metakat_pages_by_id = {
        element.id: element
        for element in metakat_io.elements
        if isinstance(element, MetakatPage)
    }

    if ordered_image_filenames is None:
        ordered_image_filenames = []
        for file_name in natsorted(os.listdir(batch_dir)):
            name, ext = os.path.splitext(file_name)
            ext = ext.lower()
            if ext in allowed_image_extensions:
                ordered_image_filenames.append(file_name)

    batch_index = 0
    for image_name in ordered_image_filenames:
        name, ext = os.path.splitext(image_name)
        image_path = os.path.join(batch_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_name} not found in batch directory {batch_dir}")
        if image_name in metakat_io.page_to_image_mapping.values():
            logger.debug(f"Image {image_name} already in MetaKatIO")
            page_id = next(pid for pid, img in metakat_io.page_to_image_mapping.items() if img == image_name)
            metakat_page = metakat_pages_by_id.get(page_id)
            if metakat_page is None:
                raise ValueError(
                    "MetaKat image mapping refers to a missing page: "
                    f"{page_id} -> {image_name}"
                )
        else:
            page_id = uuid4()
            metakat_page = MetakatPage(id=page_id,
                                       batch_id=metakat_io.batch_id,
                                       batch_index=batch_index,
                                       pageIndex=batch_index)
            metakat_io.elements.append(metakat_page)
            metakat_pages_by_id[page_id] = metakat_page
            metakat_io.page_to_image_mapping[page_id] = image_name
        metakat_page.imageDim = load_image_dimensions(image_path)
        metakat_page.altoDim = None
        batch_index += 1
        xml_name = f'{name}.xml'
        xml_path = os.path.join(batch_dir, xml_name)
        if os.path.exists(xml_path):
            xml_format = detect_xml_format(xml_path)
            if xml_format == 'INVALID_XML':
                logger.warning(f"Invalid XML format for {xml_name}, skipping")
                continue
            if xml_format == 'ALTO':
                metakat_page.altoDim = load_alto_dimensions(
                    xml_path,
                )
                warn_about_dimension_mismatch(metakat_page, image_name)
                if xml_name in metakat_io.page_to_alto_mapping.values():
                    logger.debug(f"ALTO {xml_name} already in MetaKatIO")
                    continue
                metakat_io.page_to_alto_mapping[page_id] = xml_name
            elif xml_format == 'PAGE':
                if xml_name in metakat_io.page_to_xml_mapping.values():
                    logger.debug(f"PAGE {xml_name} already in MetaKatIO")
                    continue
                metakat_io.page_to_xml_mapping[page_id] = xml_name
            else:
                logger.warning(f"Unknown XML format for {xml_name}, skipping")
                continue

    return metakat_io, proarc_io


def _preflight_engine_requirements(pipeline_config: Mapping[str, Any]) -> None:
    """Fail before any processing when an enabled engine cannot be loaded.

    Engine implementations are imported lazily, so a component whose optional
    dependencies are absent would otherwise only fail once the pipeline reached
    it, discarding the work of every component before it. Checking every enabled
    component up front keeps that failure immediate.

    Checking every enabled component up front also catches a misspelled engine
    name before any inference runs, instead of when the pipeline reaches it.
    """
    checks = (
        ("page_number", check_page_number_core_engine),
        ("page_type", check_page_type_core_engine),
        ("biblio", check_biblio_core_engine),
        ("chapter", check_chapter_core_engine),
    )
    for category, check in checks:
        component = _engine_pair(pipeline_config, category)
        if component is not None:
            check(component["core"])


def _engine_pair(
    pipeline_config: Mapping[str, Any],
    category: str,
) -> Optional[dict[str, Any]]:
    value = pipeline_config.get(category)
    if value is None:
        return None
    category_config = require_config_mapping(
        value,
        f"Pipeline {category} config",
    )
    core = category_config.get("core")
    bind = category_config.get("bind")
    if core is None and bind is None:
        return None
    if core is None or bind is None:
        raise ValueError(
            f"Pipeline {category} config must provide both 'core' and 'bind'"
        )
    return {
        "core": require_config_mapping(core, f"Pipeline {category}.core config"),
        "bind": require_config_mapping(bind, f"Pipeline {category}.bind config"),
    }


def _load_json_file(path: Optional[str]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_image_dimensions(
    image_path: str,
) -> Optional[MetakatPageDimensions]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return MetakatPageDimensions(width=width, height=height)
    except (OSError, ValueError) as error:
        logger.warning(
            "Unable to read image dimensions from %s: %s",
            image_path,
            error,
        )
        return None


def load_alto_dimensions(
    alto_path: str,
) -> Optional[MetakatPageDimensions]:
    try:
        tree = ET.parse(alto_path)
        page_element = next(
            (
                element
                for element in tree.getroot().iter()
                if element.tag.rsplit("}", 1)[-1] == "Page"
            ),
            None,
        )
        if page_element is None:
            logger.warning(
                "ALTO Page element is missing from %s",
                alto_path,
            )
            return None
        width = page_element.attrib.get("WIDTH")
        height = page_element.attrib.get("HEIGHT")
        if width is None or height is None:
            logger.warning(
                "ALTO page dimensions are missing from %s",
                alto_path,
            )
            return None
        return MetakatPageDimensions(
            width=float(width),
            height=float(height),
        )
    except (ET.ParseError, OSError, ValueError) as error:
        logger.warning(
            "Unable to read ALTO dimensions from %s: %s",
            alto_path,
            error,
        )
        return None


def warn_about_dimension_mismatch(
    page: MetakatPage,
    image_name: str,
) -> None:
    if page.imageDim is None or page.altoDim is None:
        return
    if page.imageDim == page.altoDim:
        return
    logger.warning(
        "Image and ALTO dimensions differ for page %s: image=%sx%s, "
        "ALTO=%sx%s",
        image_name,
        page.imageDim.width,
        page.imageDim.height,
        page.altoDim.width,
        page.altoDim.height,
    )


def detect_xml_format(xml_path: str) -> str:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tag = root.tag
        ns = tag.split('}')[0].strip('{') if '}' in tag else ''

        if root.tag.endswith('alto') or 'alto' in ns.lower():
            return 'ALTO'
        elif root.tag.endswith('PcGts') or 'primaresearch.org/PAGE' in ns:
            return 'PAGE'
        else:
            return 'UNKNOWN'
    except ET.ParseError:
        return 'INVALID_XML'


if __name__ == '__main__':
    main()
