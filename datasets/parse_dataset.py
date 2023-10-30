import argparse
import os
import sys
import logging
import time
import Levenshtein

from lxml import etree
from pero_ocr.core import layout


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mets', required=True, type=str)
    parser.add_argument('--page-xml-dir', required=True, type=str)

    parser.add_argument('--chapter', action='store_true')

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - PARSE DATASET - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler_sender = logging.StreamHandler()
    handler_sender.setFormatter(log_formatter)
    logger.addHandler(handler_sender)
    logger.propagate = False
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    if args.chapter:
        parse_chapters(mets=args.mets,
                       page_xml_dir=args.page_xml_dir,
                       logger=logger)


def parse_chapters(mets: str,
                   page_xml_dir: str,
                   max_title_transcription_relative_distance: float = 0.5,
                   logger: logging.Logger = logging.getLogger(__name__)):

    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    tree = etree.parse(mets)

    chapter_elements = tree.xpath(f"//mods:mods[contains(@ID, 'CHAPTER')]",
                                  namespaces=namespaces)
    page_number_elements = tree.xpath(f"//mets:structMap[contains(@TYPE, 'PHYSICAL')]",
                                      namespaces=namespaces)[0]
    mastercopy_elements = tree.xpath(f"//mets:fileGrp[contains(@ID, 'MC_IMGGRP')]",
                                      namespaces=namespaces)[0]

    mapped_chapter_elements_count = 0
    mapped_chapter_title_transcription_relative_similarity = 0

    for chapter_element in chapter_elements:
        logger.debug(f'{"CHAPTER:":>20s} {chapter_element.get("ID")}')
        title_element_content = chapter_element.xpath(f"descendant::mods:title/text()",
                                                   namespaces=namespaces)
        if (title_element_content is not None and title_element_content and title_element_content[0] is not None
                and title_element_content[0].strip() != ''):
            title = title_element_content[0].strip()
            logger.debug(f'{"TITLE:":>20s} {title}')
        else:
            logger.warning(f'EMPTY TITLE: {title_element_content}, SKIPPING\n')
            continue

        page_number_element_content = chapter_element.xpath(f"descendant::mods:number/text()",
                                                         namespaces=namespaces)
        try:
            page_number = page_number_element_content[0]
            page_number = page_number.split(',')[0].strip()
        except (IndexError, ValueError):
            logger.warning(f'INVALID OR MISSING PAGE NUMBER: {page_number_element_content}, SKIPPING\n')
            continue

        logger.debug(f'{"PAGE NUMBER:":>20s} {page_number}')

        connecting_page_element = page_number_elements.xpath(f"descendant::mets:div[@ORDERLABEL='{page_number}']/mets:fptr[contains(@FILEID, 'mc_')]",
                                                             namespaces=namespaces)
        if connecting_page_element is None or not connecting_page_element or connecting_page_element[0] is None:
            logger.warning(f'NO CONNECTING PAGE ELEMENT FOUND: {connecting_page_element}, SKIPPING\n')
            continue
        connecting_page_element = connecting_page_element[0]
        mastercopy_id = connecting_page_element.get("FILEID")
        logger.debug(f'{"MASTER COPY ID:":>20s} {mastercopy_id}')

        mastercopy_element = mastercopy_elements.xpath(f"descendant::mets:file[@ID='{mastercopy_id}']/mets:FLocat",
                                                        namespaces=namespaces)
        if mastercopy_element is None or not mastercopy_element or mastercopy_element[0] is None:
            logger.warning(f'NO MASTER COPY ELEMENT: {mastercopy_element}, SKIPPING\n')
            continue
        mastercopy_element = mastercopy_element[0]
        mastercopy_path = mastercopy_element.get("{http://www.w3.org/1999/xlink}href")
        logger.debug(f'{"MASTER COPY PATH:":>20s} {mastercopy_path}')

        page_xml_path = os.path.basename(mastercopy_path)
        page_xml_path = os.path.splitext(page_xml_path)[0]
        page_xml_path = f'{page_xml_path}.xml'
        page_xml_path = os.path.join(page_xml_dir, page_xml_path)
        logger.debug(f'{"PAGE XML PATH:":>20s} {page_xml_path}')

        page_layout = layout.PageLayout()
        page_layout.from_pagexml(page_xml_path)

        mapped_text_line = None
        for text_line in page_layout.lines_iterator():
            if len(text_line.transcription.strip()) == 0:
                continue
            edit_distance = Levenshtein.distance(text_line.transcription, title)
            if len(text_line.transcription) > len(title):
                title_transcription_relative_similarity = (len(text_line.transcription) - edit_distance) / len(
                    text_line.transcription)
            else:
                title_transcription_relative_similarity = (len(text_line.transcription) + edit_distance) / len(
                    text_line.transcription)
            if 1 - max_title_transcription_relative_distance <= title_transcription_relative_similarity <= 1 + max_title_transcription_relative_distance:
                logger.debug(f'{"MAPPED":>20s}')
                logger.debug(f'{"OCR:":>20s} {text_line.transcription}')
                logger.debug(f'{"METS:":>20s} {title}')
                logger.debug(f'{"ED:":>20s} {edit_distance}')
                logger.debug(f'{"RS:":>20s} {title_transcription_relative_similarity}')
                mapped_text_line = text_line
                mapped_chapter_title_transcription_relative_similarity += title_transcription_relative_similarity
                break

        if mapped_text_line is None:
            logger.debug(f'{"MAPPING FAILED":>20s}\n')
            continue
        else:
            mapped_chapter_elements_count += 1

        logger.debug('\n')

    logger.info(f'{"TOTAL MAPPED:":>20s} {mapped_chapter_elements_count}/{len(chapter_elements)}')
    if mapped_chapter_elements_count > 0:
        logger.info(f'{"ARS:":>20s} {mapped_chapter_title_transcription_relative_similarity / float(mapped_chapter_elements_count)}')


def print_lxml_element(e):
    print(etree.tostring(e, pretty_print=True))


if __name__ == '__main__':
    main()

