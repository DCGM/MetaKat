import argparse
import os
import sys
import logging
import time

import Levenshtein
import cv2
import numpy as np
from lxml import etree
from pero_ocr.core import layout
from pero_ocr import sequence_alignment


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mets', required=True, type=str)
    parser.add_argument('--page-xml-dir', required=True, type=str)
    parser.add_argument('--mastercopy-dir', type=str, help='If set, previews of mapped elements are render.')

    parser.add_argument('--chapter', action='store_true')
    parser.add_argument('--output-chapter-dir', type=str)

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
                       output_chapter_dir=args.output_chapter_dir,
                       mastercopy_dir=args.mastercopy_dir)


def parse_chapters(mets: str,
                   page_xml_dir: str,
                   max_title_transcription_relative_distance: float = 0.2,
                   output_chapter_dir: str = None,
                   mastercopy_dir: str = None):

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
        connect_chapter_out = connect_chapter(chapter_element,
                                              page_number_elements=page_number_elements,
                                              mastercopy_elements=mastercopy_elements,
                                              namespaces=namespaces)
        if connect_chapter_out is None:
            continue
        title, mastercopy_path = connect_chapter_out

        mastercopy_name = get_mastercopy_name(mastercopy_path)
        page_xml_path = f'{mastercopy_name}.xml'
        page_xml_path = os.path.join(page_xml_dir, page_xml_path)
        logger.debug(f'{"PAGE XML PATH:":>20s} {page_xml_path}')

        chapter_page_layout = layout.PageLayout()
        chapter_page_layout.from_pagexml(page_xml_path)

        map_chapter_out = map_chapter(chapter_page_layout, title)
        if map_chapter_out is None:
            continue
        mapped_lines, title_transcription_relative_similarity = map_chapter_out

        mapped_chapter_title_transcription_relative_similarity += title_transcription_relative_similarity
        mapped_chapter_elements_count += 1

        if output_chapter_dir is not None:
            save_mapped_chapter(mapped_lines,
                                mastercopy_path=mastercopy_path,
                                output_chapter_dir=output_chapter_dir,
                                mastercopy_dir=mastercopy_dir)

        logger.debug('')
        logger.debug('')

    logger.info(f'{"TOTAL MAPPED:":>20s} {mapped_chapter_elements_count}/{len(chapter_elements)}')
    if mapped_chapter_elements_count > 0:
        logger.info(f'{"ARS:":>20s} {mapped_chapter_title_transcription_relative_similarity / float(mapped_chapter_elements_count)}')


def connect_chapter(chapter_element, page_number_elements, mastercopy_elements, namespaces: dict) -> None | tuple[str, str]:
    logger.debug(f'{"CHAPTER:":>20s} {chapter_element.get("ID")}')
    title_element_content = chapter_element.xpath(f"descendant::mods:title/text()",
                                                  namespaces=namespaces)
    if (title_element_content is not None and title_element_content and title_element_content[0] is not None
            and title_element_content[0].strip() != ''):
        title = str(title_element_content[0].strip())
        logger.debug(f'{"TITLE:":>20s} {title}')
    else:
        logger.warning(f'EMPTY TITLE: {title_element_content}, SKIPPING\n')
        return None

    page_number_element_content = chapter_element.xpath(f"descendant::mods:number/text()",
                                                        namespaces=namespaces)
    try:
        page_number = page_number_element_content[0]
        page_number = page_number.split(',')[0].strip()
    except (IndexError, ValueError):
        logger.warning(f'INVALID OR MISSING PAGE NUMBER: {page_number_element_content}, SKIPPING\n')
        return None

    logger.debug(f'{"PAGE NUMBER:":>20s} {page_number}')

    connecting_page_element = page_number_elements.xpath(
        f"descendant::mets:div[@ORDERLABEL='{page_number}']/mets:fptr[contains(@FILEID, 'mc_')]",
        namespaces=namespaces)
    if connecting_page_element is None or not connecting_page_element or connecting_page_element[0] is None:
        logger.warning(f'NO CONNECTING PAGE ELEMENT FOUND: {connecting_page_element}, SKIPPING\n')
        return None
    connecting_page_element = connecting_page_element[0]
    mastercopy_id = connecting_page_element.get("FILEID")
    logger.debug(f'{"MASTER COPY ID:":>20s} {mastercopy_id}')

    mastercopy_element = mastercopy_elements.xpath(f"descendant::mets:file[@ID='{mastercopy_id}']/mets:FLocat",
                                                   namespaces=namespaces)
    if mastercopy_element is None or not mastercopy_element or mastercopy_element[0] is None:
        logger.warning(f'NO MASTER COPY ELEMENT: {mastercopy_element}, SKIPPING\n')
        return None
    mastercopy_element = mastercopy_element[0]
    mastercopy_path = str(mastercopy_element.get("{http://www.w3.org/1999/xlink}href"))
    logger.debug(f'{"MASTER COPY PATH:":>20s} {mastercopy_path}')

    return title, mastercopy_path


def map_chapter(page_layout: layout.PageLayout, title: str,
                max_title_transcription_relative_distance: float = 0.2,
                ):

    transcriptions = []
    n_char_to_n_line_mapping = []
    for n_line, text_line in enumerate(page_layout.lines_iterator()):
        transcriptions.append(text_line.transcription)
        n_char_to_n_line_mapping += [n_line] * (len(text_line.transcription) + 1)

    transcription = " ".join(transcriptions)
    logger.debug(f'{"OCR PAGE TRANSCRIPTION:":>20s} {transcription}')
    logger.debug(f'{"TITLE TO ALIGN:":>20s} {title}')

    align = sequence_alignment.levenshtein_alignment_substring(
        [x for x in transcription.lower()],
        [x for x in title.lower()])
    logger.debug(f'{"ALIGNMENT:":>20s} {align}')

    start_align_index = crop_alignment(align)
    end_align_index = len(align) - crop_alignment(align[::-1])

    if start_align_index is None or end_align_index is None:
        logger.debug(f'{"MAPPING FAILED, alignment was not successful":>20s}\n')
        return None

    title_transcription = transcription[start_align_index:end_align_index]
    title_transcription_edit_distance = Levenshtein.distance(title_transcription.lower(), title.lower())

    if len(title_transcription) > len(title):
        title_transcription_relative_similarity = (len(title_transcription) - title_transcription_edit_distance) / len(
            title_transcription)
    else:
        title_transcription_relative_similarity = (len(title_transcription) + title_transcription_edit_distance) / len(
            title_transcription)
    align_lines = list(page_layout.lines_iterator())[n_char_to_n_line_mapping[start_align_index]:
                                                     n_char_to_n_line_mapping[end_align_index] + 1]
    for align_line in align_lines:
        logger.debug(f'{"ALIGN LINES:":>20s} {align_line.transcription}')
    logger.debug(f'{"OCR:":>20s} {title_transcription}')
    logger.debug(f'{"METS:":>20s} {title}')

    if 1 - max_title_transcription_relative_distance <= title_transcription_relative_similarity <= 1 + max_title_transcription_relative_distance:
        logger.debug(f'{"MAPPED":>20s}')
        logger.debug(f'{"ED:":>20s} {title_transcription_edit_distance}')
        logger.debug(f'{"RS:":>20s} {title_transcription_relative_similarity}')
        return align_lines, title_transcription_relative_similarity
    else:
        logger.debug(f'{"MAPPING FAILED, relative distance is too high":>20s}\n')
        return None


def save_mapped_chapter(aligned_lines: list[layout.TextLine],
                        mastercopy_path: str,
                        output_chapter_dir: str,
                        mastercopy_dir: str = None):
    mastercopy_name = get_mastercopy_name(mastercopy_path)
    chapter_name = f'{mastercopy_name}.chapter'

    chapter_polygon = create_bounding_box_for_lines(aligned_lines)

    chapter_page_layout = layout.PageLayout(id=mastercopy_name)
    chapter_page_layout.regions.append(layout.RegionLayout(id=chapter_name,
                                                           polygon=chapter_polygon))

    page_xml_chapter_output_path = os.path.join(output_chapter_dir, 'page_xml.chapter')
    os.makedirs(page_xml_chapter_output_path, exist_ok=True)
    chapter_page_layout.to_pagexml(os.path.join(page_xml_chapter_output_path, f'{chapter_name}.xml'))

    if mastercopy_dir is not None:
        render_chapter_output_path = os.path.join(output_chapter_dir, 'render.chapter')
        os.makedirs(render_chapter_output_path, exist_ok=True)
        img = cv2.imread(os.path.join(mastercopy_dir, os.path.basename(mastercopy_path)))
        img = chapter_page_layout.render_to_image(img)
        cv2.imwrite(os.path.join(render_chapter_output_path, f'{chapter_name}.jpg'), img)


def create_bounding_box_for_lines(lines: list[layout.TextLine], pad: int = 40):
    max_top_height = 0
    max_bottom_height = 0
    left = 1000000000
    right = 0
    top = 1000000000
    bottom = 0
    for align_line in lines:
        if align_line.heights[0] > max_top_height:
            max_top_height = align_line.heights[0]
        if align_line.heights[1] > max_bottom_height:
            max_bottom_height = align_line.heights[1]
        if np.min(align_line.baseline[:, 0]) < left:
            left = align_line.baseline[0][0]
        if np.max(align_line.baseline[:, 0]) > right:
            right = align_line.baseline[-1][0]
        if np.min(align_line.baseline[:, 1]) < top:
            top = np.min(align_line.baseline[:, 1])
        if np.max(align_line.baseline[:, 1]) > bottom:
            bottom = np.max(align_line.baseline[:, 1])
    bbox = np.asarray([[left - pad, top - max_top_height - pad],
                       [right + pad, top - max_top_height - pad],
                       [right + pad, bottom + max_bottom_height + pad],
                       [left - pad, bottom + max_bottom_height + pad]])
    bbox = bbox.clip(min=0)
    return bbox


def get_mastercopy_name(mastercopy_path: str):
    mastercopy_name = os.path.basename(mastercopy_path)
    mastercopy_name = os.path.splitext(mastercopy_name)[0]
    return mastercopy_name


def crop_alignment(align, dim=1):
    for i, al in enumerate(align):
        if al[dim] is not None:
            return i


def print_lxml_element(e):
    logger.info(etree.tostring(e, pretty_print=True))


if __name__ == '__main__':
    main()

