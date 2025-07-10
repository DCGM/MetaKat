import argparse
import json
import logging
import os.path
import sys
import time
from typing import Tuple
from uuid import uuid4, UUID
import xml.etree.ElementTree as ET

from natsort import natsorted

from chapter.engines.bind.definitions import load_chapter_bind_engine
from page_type.engines.bind.definitions import load_page_type_bind_engine
from metakat.biblio.engines.bind.definitions import load_biblio_bind_engine

from metakat.schemas.base_objects import MetakatIO, ProarcIO, MetakatPage

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-dir', type=str, required=True)
    parser.add_argument('--metakat-json', type=str)
    parser.add_argument('--proarc-json', type=str)

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

    metakat_io, proarc_io = init_io(
        batch_dir=args.batch_dir,
        metakat_json=args.metakat_json,
        proarc_json=args.proarc_json
    )

    if args.page_type_bind_engine is not None and args.page_type_core_engine is not None:
        page_type_bind_engine = load_page_type_bind_engine(
            args.page_type_bind_engine,
            args.page_type_core_engine
        )
        metakat_io = page_type_bind_engine.process(
            batch_dir=args.batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    if args.biblio_bind_engine is not None and args.biblio_core_engine is not None:
        biblio_bind_engine = load_biblio_bind_engine(args.biblio_bind_engine, args.biblio_core_engine)
        metakat_io = biblio_bind_engine.process(
            batch_dir=args.batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    if args.chapter_bind_engine is not None and args.chapter_core_engine is not None:
        chapter_bind_engine = load_chapter_bind_engine(
            args.chapter_bind_engine,
            args.chapter_core_engine
        )
        metakat_io = chapter_bind_engine.process(
            batch_dir=args.batch_dir,
            metakat_io=metakat_io,
            proarc_io=proarc_io
        )

    logger.info("")
    MetakatIO.model_validate_json(json.dumps(metakat_io.model_dump(mode="json")))
    logger.info("MetakatIO has been successfully validated")

    if args.output_metakat_json is not None:
        with open(args.output_metakat_json, 'w') as f:
            json.dump(metakat_io.model_dump(mode="json"), f, indent=4, ensure_ascii=False)
        logger.info(f"MetakatIO saved to {args.output_metakat_json}")





def init_io(batch_dir: str, metakat_json: str, proarc_json: str, batch_id: UUID = uuid4()) -> Tuple[MetakatIO, ProarcIO]:
    if metakat_json is not None:
        metakat_io = MetakatIO.from_json(metakat_json)
    else:
        metakat_io = MetakatIO(batch_id=batch_id)

    if proarc_json is not None:
        proarc_io = ProarcIO.from_json(proarc_json)
    else:
        proarc_io = None

    batch_index = 0
    if metakat_io.page_to_image_mapping is None:
        metakat_io.page_to_image_mapping = {}
    if metakat_io.page_to_alto_mapping is None:
        metakat_io.page_to_alto_mapping = {}
    if metakat_io.page_to_xml_mapping is None:
        metakat_io.page_to_xml_mapping = {}
    for image_name in natsorted(os.listdir(batch_dir)):
        name, ext = os.path.splitext(image_name)
        ext = ext.lower()
        if ext not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
            continue
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