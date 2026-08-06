import argparse
import json
import logging
import os.path
import sys
import time
from typing import Tuple, List, Optional, Set
from uuid import uuid4, UUID
import xml.etree.ElementTree as ET

from natsort import natsorted

from metakat.chapter.engines.bind.definitions import load_chapter_bind_engine
from metakat.page_number.engines.bind.definitions import (
    load_page_number_bind_engine,
)
from metakat.page_type.engines.bind.definitions import load_page_type_bind_engine
from metakat.biblio.engines.bind.definitions import load_biblio_bind_engine

from metakat.schemas.base_objects import MetakatIO, ProarcIO, MetakatPage

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-dir', type=str, required=True)
    parser.add_argument('--metakat-json', type=str)
    parser.add_argument('--proarc-json', type=str)

    parser.add_argument('--allowed-image-extensions', type=str, nargs='*', default=['.jpg', '.jpeg', '.png', '.tif', '.tiff'])

    parser.add_argument('--page-number-core-engine', type=str, help='Path to directory containing page number core engine')
    parser.add_argument('--page-number-bind-engine', type=str, help='Path to directory containing page number bind engine')
    parser.add_argument('--page-type-core-engine', type=str, help='Path to directory containing page type core engine')
    parser.add_argument('--page-type-bind-engine', type=str, help='Path to directory containing page type bind engine')
    parser.add_argument('--biblio-core-engine', type=str, help='Path to directory containing biblio core engine')
    parser.add_argument('--biblio-bind-engine', type=str, help='Path to directory containing biblio bind engine')
    parser.add_argument('--chapter-core-engine', type=str, help='Path to directory containing chapter core engine')
    parser.add_argument('--chapter-bind-engine', type=str, help='Path to directory containing chapter bind engine')

    parser.add_argument('--output-metakat-json', type=str, help='Path to output Metakat JSON file')

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

    logger.info(' '.join(sys.argv))

    process_batch(
        batch_dir=args.batch_dir,
        metakat_json=args.metakat_json,
        proarc_json=args.proarc_json,
        page_number_core_engine=args.page_number_core_engine,
        page_number_bind_engine=args.page_number_bind_engine,
        page_type_core_engine=args.page_type_core_engine,
        page_type_bind_engine=args.page_type_bind_engine,
        biblio_core_engine=args.biblio_core_engine,
        biblio_bind_engine=args.biblio_bind_engine,
        chapter_core_engine=args.chapter_core_engine,
        chapter_bind_engine=args.chapter_bind_engine,
        output_metakat_json=args.output_metakat_json,
        allowed_image_extensions=set(args.allowed_image_extensions)
    )
    

def process_batch(
    batch_dir: str,
    metakat_json: Optional[str] = None,
    proarc_json: Optional[str] = None,
    ordered_image_filenames: Optional[List] = None,
    page_number_core_engine: Optional[str] = None,
    page_number_bind_engine: Optional[str] = None,
    page_type_core_engine: Optional[str] = None,
    page_type_bind_engine: Optional[str] = None,
    biblio_core_engine: Optional[str] = None,
    biblio_bind_engine: Optional[str] = None,
    chapter_core_engine: Optional[str] = None,
    chapter_bind_engine: Optional[str] = None,
    output_metakat_json: Optional[str] = None,
    allowed_image_extensions: Optional[Set] = None,
) -> MetakatIO:
    """
    Process a batch directory and return the processed MetakatIO object.
    
    Args:
        batch_dir: Path to the batch directory
        metakat_json: Path to input Metakat JSON file
        proarc_json: Path to input ProARC JSON file
        ordered_image_filenames: List of ordered image filenames in batch_dir (defaults to natsorted image files)
        page_number_core_engine: Path to page number core engine directory
        page_number_bind_engine: Path to page number bind engine directory
        page_type_core_engine: Path to page type core engine directory
        page_type_bind_engine: Path to page type bind engine directory
        biblio_core_engine: Path to biblio core engine directory
        biblio_bind_engine: Path to biblio bind engine directory
        chapter_core_engine: Path to chapter core engine directory
        chapter_bind_engine: Path to chapter bind engine directory
        output_metakat_json: Path to output Metakat JSON file
        allowed_image_extensions: Set of allowed image file extensions

    Returns:
        Processed MetakatIO object
    """
    if allowed_image_extensions is None:
        allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


    metakat_io, proarc_io = init_io(
        batch_dir=batch_dir,
        metakat_json=metakat_json,
        proarc_json=proarc_json,
        ordered_image_filenames=ordered_image_filenames,
        allowed_image_extensions=allowed_image_extensions
    )

    if (
        page_number_bind_engine is not None
        and page_number_core_engine is not None
    ):
        page_number_bind_engine_obj = load_page_number_bind_engine(
            page_number_bind_engine,
            page_number_core_engine,
        )
        metakat_io = page_number_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io,
        )

    if page_type_bind_engine is not None and page_type_core_engine is not None:
        page_type_bind_engine_obj = load_page_type_bind_engine(
            page_type_bind_engine,
            page_type_core_engine
        )
        metakat_io = page_type_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    if biblio_bind_engine is not None and biblio_core_engine is not None:
        biblio_bind_engine_obj = load_biblio_bind_engine(biblio_bind_engine, biblio_core_engine)
        metakat_io = biblio_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    if chapter_bind_engine is not None and chapter_core_engine is not None:
        chapter_bind_engine_obj = load_chapter_bind_engine(
            chapter_bind_engine,
            chapter_core_engine
        )
        metakat_io = chapter_bind_engine_obj.process(
            batch_dir=batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    logger.info("")
    MetakatIO.model_validate_json(json.dumps(metakat_io.model_dump(mode="json")))
    logger.info("MetakatIO has been successfully validated")

    if output_metakat_json is not None:
        with open(output_metakat_json, 'w') as f:
            json.dump(metakat_io.model_dump(mode="json"), f, indent=4, ensure_ascii=False)
        logger.info(f"MetakatIO saved to {output_metakat_json}")
    
    return metakat_io


def init_io(batch_dir: str,
            metakat_json: Optional[str] = None,
            proarc_json: Optional[str] = None,
            batch_id: UUID = uuid4(),
            ordered_image_filenames: Optional[List] = None,
            allowed_image_extensions: Optional[Set] = None) -> Tuple[MetakatIO, ProarcIO]:
    if allowed_image_extensions is None:
        allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    if metakat_json is not None:
        with open(metakat_json, 'r', encoding='utf-8') as f:
            metakat_io = MetakatIO.model_validate_json(f.read())
    else:
        metakat_io = MetakatIO(batch_id=batch_id)

    if proarc_json is not None:
        with open(proarc_json, 'r', encoding='utf-8') as f:
            proarc_io = ProarcIO.model_validate_json(f.read())
    else:
        proarc_io = None

    if metakat_io.page_to_image_mapping is None:
        metakat_io.page_to_image_mapping = {}
    if metakat_io.page_to_alto_mapping is None:
        metakat_io.page_to_alto_mapping = {}
    if metakat_io.page_to_xml_mapping is None:
        metakat_io.page_to_xml_mapping = {}

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
        else:
            page_id = uuid4()
            metakat_page = MetakatPage(id=page_id,
                                       batch_id=metakat_io.batch_id,
                                       batch_index=batch_index,
                                       pageIndex=batch_index)
            metakat_io.elements.append(metakat_page)
            metakat_io.page_to_image_mapping[page_id] = image_name
        batch_index += 1
        xml_name = f'{name}.xml'
        xml_path = os.path.join(batch_dir, xml_name)
        if os.path.exists(xml_path):
            xml_format = detect_xml_format(xml_path)
            if xml_format == 'INVALID_XML':
                logger.warning(f"Invalid XML format for {xml_name}, skipping")
                continue
            if xml_format == 'ALTO':
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
