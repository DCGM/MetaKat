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
                       output_chapter_dir=args.output_chapter_dir)


def parse_chapters(mets: str,
                   page_xml_dir: str,
                   max_title_transcription_relative_distance: float = 0.2,
                   output_chapter_dir: str = None):

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
            logger.debug(f'{"Alignment was not successful, MAPPING FAILED":>20s}\n')
            continue

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
            mapped_chapter_title_transcription_relative_similarity += title_transcription_relative_similarity
            mapped_chapter_elements_count += 1

            if output_chapter_dir is not None:
                page_xml_chapter_output_path = os.path.join(output_chapter_dir, 'page_xml')
                render_chapter_output_path = os.path.join(output_chapter_dir, 'render')
                os.makedirs(page_xml_chapter_output_path, exist_ok=True)
                os.makedirs(render_chapter_output_path, exist_ok=True)

                max_top_height = 0
                max_bottom_height = 0
                left = 10000000
                right = 0
                top = 1000000
                bottom = 0
                for align_line in align_lines:
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
                logger.debug(f'{"MAX TOP HEIGHT":>20s} {max_top_height}')
                logger.debug(f'{"MAX BOTTOM HEIGHT":>20s} {max_bottom_height}')
                logger.debug(f'{"LEFT":>20s} {left}')
                logger.debug(f'{"RIGHT":>20s} {right}')
                logger.debug(f'{"TOP":>20s} {top}')
                logger.debug(f'{"BOTTOM":>20s} {bottom}')

                pad = 40
                chapter_polygon = np.asarray([[left - pad, top - max_top_height - pad],
                                             [right + pad, top - max_top_height - pad],
                                             [right + pad, bottom + max_bottom_height + pad],
                                             [left - pad, bottom + max_bottom_height + pad]])
                chapter_polygon = chapter_polygon.clip(min=0)

                chapter_base_name = os.path.basename(mastercopy_path)
                chapter_base_name = os.path.splitext(chapter_base_name)[0]

                chapter_page_layout = layout.PageLayout(id=chapter_base_name)
                chapter_page_layout.regions.append(layout.RegionLayout(id=chapter_base_name,
                                                   polygon=chapter_polygon))

                chapter_page_layout.to_pagexml(os.path.join(page_xml_chapter_output_path, f'{chapter_base_name}.chapter.xml'))
                img = cv2.imread(os.path.join("/".join(mets.split('/')[:-1]), str(mastercopy_path)))
                img = chapter_page_layout.render_to_image(img)
                cv2.imwrite(os.path.join(render_chapter_output_path, f'{chapter_base_name}.chapter.jpg'), img)

        else:
            logger.debug(f'{"Relative distance is too high, MAPPING FAILED":>20s}\n')
            continue


        # if mapped_text_lines:
        #     logger.debug(f'{"MAPPED":>20s}')
        #     logger.debug(f'{"OCR:":>20s} {title_transcription}')
        #     logger.debug(f'{"METS:":>20s} {title}')
        #     logger.debug(f'{"ED:":>20s} {title_transcription_edit_distance}')
        #     logger.debug(f'{"RS:":>20s} {title_transcription_relative_similarity}')
        #     mapped_chapter_title_transcription_relative_similarity += np.asarray(title_transcription_relative_similarity).mean()
        #     mapped_chapter_elements_count += 1
        # else:
        #     logger.debug(f'{"MAPPING FAILED":>20s}\n')
        #     continue

        """
        mapped_text_lines = []
        title_transcription = ""
        title_transcription_edit_distance = 0
        title_transcription_relative_similarity = 0
        for text_line in page_layout.lines_iterator():
            if len(text_line.transcription.strip()) == 0:
                continue
            if title_transcription == "":
                title_transcription += text_line.transcription
            else:
                title_transcription += f' {text_line.transcription}'
            title_transcription_edit_distance = sequence_alignment.levenshtein_distance_substring(
                [x for x in title_transcription.lower()],
                [x for x in title.lower()]
            )
            if len(title_transcription) > len(title):
                title_transcription_relative_similarity = (len(title_transcription) - title_transcription_edit_distance) / len(title_transcription)
            else:
                title_transcription_relative_similarity = (len(title_transcription) + title_transcription_edit_distance) / len(title_transcription)
            if 1 - max_title_transcription_relative_distance <= title_transcription_relative_similarity <= 1 + max_title_transcription_relative_distance:
                mapped_text_lines.append(text_line)
            else:
                mapped_text_lines = []
                title_transcription = ""
                title_transcription_edit_distance = 0
                title_transcription_relative_similarity = 0
            if len(title_transcription) > len(title):
                break

        if mapped_text_lines:
            logger.debug(f'{"MAPPED":>20s}')
            logger.debug(f'{"OCR:":>20s} {title_transcription}')
            logger.debug(f'{"METS:":>20s} {title}')
            logger.debug(f'{"ED:":>20s} {title_transcription_edit_distance}')
            logger.debug(f'{"RS:":>20s} {title_transcription_relative_similarity}')
            mapped_chapter_title_transcription_relative_similarity += np.asarray(title_transcription_relative_similarity).mean()
            mapped_chapter_elements_count += 1
        else:
            logger.debug(f'{"MAPPING FAILED":>20s}\n')
            continue
            
        """

        logger.debug('')
        logger.debug('')

    logger.info(f'{"TOTAL MAPPED:":>20s} {mapped_chapter_elements_count}/{len(chapter_elements)}')
    if mapped_chapter_elements_count > 0:
        logger.info(f'{"ARS:":>20s} {mapped_chapter_title_transcription_relative_similarity / float(mapped_chapter_elements_count)}')


def crop_alignment(align, dim=1):
    for i, al in enumerate(align):
        if al[dim] is not None:
            return i


def print_lxml_element(e):
    logger.info(etree.tostring(e, pretty_print=True))


if __name__ == '__main__':
    main()

