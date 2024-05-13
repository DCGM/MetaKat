import argparse
import json
import os
import re
import sys
import logging
import time
from collections import defaultdict

import Levenshtein
import cv2
import numpy as np
from lxml import etree
from pero_ocr.core import layout
from pero_ocr import sequence_alignment
from pero_ocr.core.layout import TextLine

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mets', required=True, type=str)
    parser.add_argument('--page-xml-dir', required=True, type=str)
    parser.add_argument('--mastercopy-dir', type=str, help='If set, previews of mapped elements are render.')

    parser.add_argument('--chapters', action='store_true')
    parser.add_argument('--numbers', action='store_true')
    parser.add_argument('--table-of-contents', action='store_true')

    parser.add_argument('--output-render-dir', type=str)
    parser.add_argument('--output-label-studio-dir', type=str)
    parser.add_argument('--label-studio-predictions-must-include', type=str)
    parser.add_argument('--label-studio-predictions-always-include', type=str)

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

    # page_xml_dir=args.page_xml_dir,
    # output_chapter_dir=args.output_chapter_dir,

    dataset = parse_dataset(mets=args.mets,
                            page_xml_dir=args.page_xml_dir,
                            parse_chapters=args.chapters,
                            parse_numbers=args.numbers,
                            parse_table_of_contents=args.table_of_contents)

    if args.output_render_dir is not None:
        render_dataset(dataset, args.mastercopy_dir, args.output_render_dir)

    if args.output_label_studio_dir is not None:
        save_label_studio_storage(dataset,
                                  args.mastercopy_dir,
                                  args.output_label_studio_dir,
                                  must_include=[x.strip() for x in args.label_studio_predictions_must_include.split(",")],
                                  always_include=[x.strip() for x in args.label_studio_predictions_always_include.split(",")])


def parse_dataset(mets: str,
                  page_xml_dir: str,
                  parse_chapters=False,
                  max_chapter_transcription_relative_distance: float = 0.2,
                  parse_numbers=False,
                  max_number_transcription_relative_distance: float = 0.2,
                  parse_table_of_contents=False):

    dataset = defaultdict(dict)

    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    tree = etree.parse(mets)

    volume_elements = tree.xpath(f"//mods:mods[contains(@ID, 'VOLUME')]",
                                 namespaces=namespaces)
    volume_uuid_element_content = volume_elements[0].xpath(f"descendant::mods:identifier[contains(@type, 'uuid')]/text()",
                                                           namespaces=namespaces)[0]
    logger.debug(f'{"VOLUME UUID:":>20s} {volume_uuid_element_content}')
    logger.debug('')
    logger.debug('')

    mastercopy_elements = tree.xpath(f"//mets:fileGrp[contains(@ID, 'MC_IMGGRP')]",
                                     namespaces=namespaces)[0]

    if parse_chapters:
        chapter_elements = tree.xpath(f"//mods:mods[contains(@ID, 'CHAPTER')]",
                                      namespaces=namespaces)
        page_number_elements = tree.xpath(f"//mets:structMap[contains(@TYPE, 'PHYSICAL')]",
                                          namespaces=namespaces)[0]

        mapped_chapter_elements_count = 0
        mapped_chapter_transcription_relative_similarity = 0

        for chapter_element in chapter_elements:
            connect_chapter_out = connect_chapter(chapter_element,
                                                  page_number_elements=page_number_elements,
                                                  mastercopy_elements=mastercopy_elements,
                                                  namespaces=namespaces)
            if connect_chapter_out is None:
                continue
            chapter_id, chapter, mastercopy_path = connect_chapter_out
            mastercopy_name = get_mastercopy_name(mastercopy_path)

            page_xml_path = f'{mastercopy_name}.xml'
            page_xml_path = os.path.join(page_xml_dir, page_xml_path)
            logger.debug(f'{"PAGE XML PATH:":>20s} {page_xml_path}')

            chapter_page_layout = layout.PageLayout()
            chapter_page_layout.from_pagexml(page_xml_path)

            map_chapter_out = map_chapter(chapter_page_layout, chapter,
                                          max_chapter_transcription_relative_distance=max_chapter_transcription_relative_distance)
            if map_chapter_out is None:
                continue
            mapped_lines, chapter_transcription_relative_similarity = map_chapter_out

            mapped_chapter_transcription_relative_similarity += chapter_transcription_relative_similarity
            mapped_chapter_elements_count += 1

            chapter_bbox = create_bounding_box_for_lines(mapped_lines)

            dataset_page_id = get_dateset_page_id(volume_uuid=volume_uuid_element_content, mastercopy_path=mastercopy_path)
            dataset_chapter_id = f'{mastercopy_name}.chapter.{chapter_id}'
            dataset[dataset_page_id]['mastercopy_path'] = mastercopy_path
            if 'chapters' in dataset[dataset_page_id]:
                dataset[dataset_page_id]['chapters'].append([dataset_chapter_id, chapter, chapter_bbox])
            else:
                dataset[dataset_page_id]['chapters'] = [[dataset_chapter_id, chapter, chapter_bbox]]
            logger.debug('')
            logger.debug('')

        logger.info(f'{"TOTAL MAPPED CHAPTERS:":>20s} {mapped_chapter_elements_count}/{len(chapter_elements)}')
        if mapped_chapter_elements_count > 0:
            logger.info(f'{"ARS:":>20s} {mapped_chapter_transcription_relative_similarity / float(mapped_chapter_elements_count)}')
        logger.info('')

    if parse_numbers:
        number_elements = tree.xpath(f"//mets:div[boolean(@ORDERLABEL)][contains(@ID, 'PAGE')]",
                                     namespaces=namespaces)

        mapped_number_elements_count = 0
        mapped_number_transcription_relative_similarity = 0

        for number_element in number_elements:
            connect_number_out = connect_number(number_element, mastercopy_elements, namespaces)

            if connect_number_out is None:
                continue
            number_id, number, mastercopy_path = connect_number_out
            mastercopy_name = get_mastercopy_name(mastercopy_path)

            page_xml_path = f'{mastercopy_name}.xml'
            page_xml_path = os.path.join(page_xml_dir, page_xml_path)
            logger.debug(f'{"PAGE XML PATH:":>20s} {page_xml_path}')

            number_page_layout = layout.PageLayout()
            number_page_layout.from_pagexml(page_xml_path)

            map_number_out = map_number(
                number_page_layout, number,
                max_number_transcription_relative_distance=max_number_transcription_relative_distance)
            if map_number_out is None:
                continue
            mapped_lines, number_transcription_relative_similarity = map_number_out

            mapped_number_transcription_relative_similarity += number_transcription_relative_similarity
            mapped_number_elements_count += 1

            number_bbox = create_bounding_box_for_lines(mapped_lines)

            dataset_page_id = get_dateset_page_id(volume_uuid=volume_uuid_element_content,
                                                  mastercopy_path=mastercopy_path)
            dataset_number_id = f'{mastercopy_name}.number.{number_id}'
            dataset[dataset_page_id]['mastercopy_path'] = mastercopy_path
            if 'numbers' in dataset[dataset_page_id]:
                dataset[dataset_page_id]['numbers'].append([dataset_number_id, number, number_bbox])
            else:
                dataset[dataset_page_id]['numbers'] = [[dataset_number_id, number, number_bbox]]
            logger.debug('')
            logger.debug('')

        logger.info(f'{"TOTAL MAPPED NUMBERS:":>20s} {mapped_number_elements_count}/{len(number_elements)}')
        if mapped_number_elements_count > 0:
            logger.info(
                f'{"ARS:":>20s} {mapped_number_transcription_relative_similarity / float(mapped_number_elements_count)}')
        logger.info('')

    if parse_table_of_contents:
        table_of_contents_elements = tree.xpath(f"//mets:div[contains(@TYPE, 'tableOfContents')][contains(@ID, 'PAGE')]",
                                                namespaces=namespaces)

        mapped_table_of_contents_elements_count = 0

        for table_of_contents_element in table_of_contents_elements:
            connect_table_of_contents_out = connect_table_of_contents(table_of_contents_element, mastercopy_elements, namespaces)

            if connect_table_of_contents_out is None:
                continue
            table_of_contents_id, mastercopy_path = connect_table_of_contents_out
            mastercopy_name = get_mastercopy_name(mastercopy_path)

            mapped_table_of_contents_elements_count += 1

            dataset_page_id = get_dateset_page_id(volume_uuid=volume_uuid_element_content,
                                                  mastercopy_path=mastercopy_path)
            dataset_table_of_contents_id = f'{mastercopy_name}.table_of_contents.{table_of_contents_id}'
            dataset[dataset_page_id]['mastercopy_path'] = mastercopy_path
            if 'table_of_contents' in dataset[dataset_page_id]:
                dataset[dataset_page_id]['table_of_contents'].append([dataset_table_of_contents_id])
            else:
                dataset[dataset_page_id]['table_of_contents'] = [[dataset_table_of_contents_id]]
            logger.debug('')
            logger.debug('')

        logger.info(f'{"TOTAL MAPPED TABLE OF CONTENTS:":>20s} {mapped_table_of_contents_elements_count}/{len(table_of_contents_elements)}')
        logger.info('')

    return dataset


def connect_chapter(chapter_element, page_number_elements, mastercopy_elements, namespaces: dict) -> None | tuple[str, str, str]:
    chapter_id = chapter_element.get("ID")
    logger.debug(f'{"CHAPTER ID:":>20s} {chapter_id}')
    title_element_content = chapter_element.xpath(f"descendant::mods:title/text()",
                                                  namespaces=namespaces)
    if (title_element_content is not None and title_element_content and title_element_content[0] is not None
            and title_element_content[0].strip() != ''):
        chapter = str(title_element_content[0].strip())
        logger.debug(f'{"TITLE:":>20s} {chapter}')
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

    mastercopy_path = get_mastercopy_path(mastercopy_elements, mastercopy_id, namespaces)
    logger.debug(f'{"MASTER COPY PATH:":>20s} {mastercopy_path}')

    return chapter_id, chapter, mastercopy_path


def map_chapter(page_layout: layout.PageLayout, chapter: str,
                max_chapter_transcription_relative_distance: float = 0.2):

    transcriptions = []
    n_char_to_n_line_mapping = []
    for n_line, text_line in enumerate(page_layout.lines_iterator()):
        transcriptions.append(text_line.transcription)
        n_char_to_n_line_mapping += [n_line] * (len(text_line.transcription) + 1)

    transcription = " ".join(transcriptions)
    logger.debug(f'{"OCR PAGE TRANSCRIPTION:":>20s} {transcription}')
    logger.debug(f'{"CHAPTER TO ALIGN:":>20s} {chapter}')

    transcription_to_align = [x for x in transcription.lower()]
    chapter_to_align = [x for x in chapter.lower()]
    alignments = []
    while True:
        align = sequence_alignment.levenshtein_alignment_substring(transcription_to_align, chapter_to_align)
        logger.debug(f'{"ALIGNMENT:":>20s} {align}')
        start_align_index = crop_alignment(align)
        if start_align_index == len(transcription_to_align) - 1:
            break
        end_align_index = len(align) - crop_alignment(align[::-1])

        if start_align_index is None or end_align_index is None:
            logger.debug(f'{"MAPPING FAILED, alignment was not successful":>20s}\n')
            return None

        chapter_transcription = "".join(transcription_to_align[start_align_index:end_align_index])
        chapter_transcription_edit_distance = Levenshtein.distance(chapter_transcription.lower(), chapter.lower())

        match, relative_similarity = compare_orig_to_transcription(chapter,
                                                                   chapter_transcription,
                                                                   chapter_transcription_edit_distance,
                                                                   max_chapter_transcription_relative_distance)

        if match:
            align_lines = list(page_layout.lines_iterator())[n_char_to_n_line_mapping[start_align_index]:
                                                             n_char_to_n_line_mapping[end_align_index] + 1]

            chapter_height = 0
            for align_line in align_lines:
                chapter_height += align_line.heights[0] + align_line.heights[1]
            alignments.append([start_align_index, end_align_index,
                               chapter_transcription,
                               chapter_transcription_edit_distance,
                               relative_similarity,
                               chapter_height,
                               align_lines])
            transcription_to_align[start_align_index:end_align_index] = ['#'] * (end_align_index - start_align_index)
        else:
            break

    if alignments:
        logger.debug(f'{"ALIGNMENTS:":>20s} {[alignment[:-1] for alignment in alignments]}')
        alignments = sorted(alignments, key=lambda x: x[-2])
        final_alignment = alignments[-1]
        start_align_index, end_align_index, \
            chapter_transcription, \
            chapter_transcription_edit_distance, \
            chapter_transcription_relative_similarity, \
            _, final_aligned_lines = final_alignment
        for align_line in final_aligned_lines:
            logger.debug(f'{"ALIGN LINES:":>20s} {align_line.transcription}')
        logger.debug(f'{"OCR:":>20s} {chapter_transcription}')
        logger.debug(f'{"METS:":>20s} {chapter}')
        logger.debug(f'{"MAPPED":>20s}')
        logger.debug(f'{"ED:":>20s} {chapter_transcription_edit_distance}')
        logger.debug(f'{"RS:":>20s} {chapter_transcription_relative_similarity}')
        return final_aligned_lines, chapter_transcription_relative_similarity

    logger.debug(f'{"MAPPING FAILED, relative distance is too high":>20s}\n')
    return [TextLine(id='chapter_not_mapped')], 1


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

        label_studio_chapter_output_path = os.path.join(output_chapter_dir, 'label_studio.chapter')
        os.makedirs(label_studio_chapter_output_path, exist_ok=True)

        label_studio_json = create_label_studio_json(
            left_top=chapter_polygon[0],
            right_bottom=chapter_polygon[2],
            img_shape=img.shape,
            img_name=os.path.basename(mastercopy_path)
        )
        with open(os.path.join(label_studio_chapter_output_path, f'{mastercopy_name}.json'), 'w') as f:
            f.write(label_studio_json)


def connect_number(number_element, mastercopy_elements, namespaces: dict) -> None | tuple[str, str, str]:
    number_id = number_element.get('ID')
    number = number_element.get('ORDERLABEL')
    number = number.strip()
    if number == '':
        return None
    pattern = re.compile(r"""(^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$)""")
    if not re.match(pattern, number):
        try:
            int(number)
        except ValueError:
            return None
    logger.debug(f'{"NUMBER ID:":>20s} {number_id}')
    logger.debug(f'{"NUMBER:":>20s} {number}')
    mastercopy_id = number_element[0].get('FILEID')
    logger.debug(f'{"MASTER COPY ID:":>20s} {mastercopy_id}')
    mastercopy_path = get_mastercopy_path(mastercopy_elements, mastercopy_id, namespaces)
    logger.debug(f'{"MASTER COPY PATH:":>20s} {mastercopy_path}')
    return number_id, number, mastercopy_path


def map_number(page_layout: layout.PageLayout, number: str,
               max_number_transcription_relative_distance: float = 0.2):

    logger.debug(f'{"NUMBER TO ALIGN:":>20s} {number}')

    number_transcription_edit_distances = []
    for text_line in page_layout.lines_iterator():
        if text_line.transcription.strip() == '':
            continue
        number_transcription_edit_distances.append([text_line,
                                                    Levenshtein.distance(text_line.transcription.lower(), number.lower())])
    number_text_line, number_transcription_edit_distance = sorted(number_transcription_edit_distances, key=lambda x: x[-1])[0]

    match, relative_similarity = compare_orig_to_transcription(number,
                                                               number_text_line.transcription,
                                                               number_transcription_edit_distance,
                                                               max_number_transcription_relative_distance)
    if match:
        logger.debug(f'{"OCR:":>20s} {number_text_line.transcription}')
        logger.debug(f'{"METS:":>20s} {number}')
        logger.debug(f'{"MAPPED":>20s}')
        logger.debug(f'{"ED:":>20s} {number_transcription_edit_distance}')
        logger.debug(f'{"RS:":>20s} {relative_similarity}')
        return [number_text_line], relative_similarity

    logger.debug(f'{"MAPPING FAILED, relative distance is too high":>20s}\n')
    return [TextLine(id='number_not_mapped')], 1


def connect_table_of_contents(table_of_contents_element, mastercopy_elements, namespaces: dict) -> None | tuple[str, str]:
    table_of_contents_id = table_of_contents_element.get('ID')
    logger.debug(f'{"TABLE OF CONTENTS ID:":>20s} {table_of_contents_id}')
    mastercopy_id = table_of_contents_element[0].get('FILEID')
    logger.debug(f'{"MASTER COPY ID:":>20s} {mastercopy_id}')
    mastercopy_path = get_mastercopy_path(mastercopy_elements, mastercopy_id, namespaces)
    logger.debug(f'{"MASTER COPY PATH:":>20s} {mastercopy_path}')
    return table_of_contents_id, mastercopy_path


def create_bounding_box_for_lines(lines: list[layout.TextLine], pad: int = 25):
    if not lines:
        return
    if lines[0].id == 'chapter_not_mapped':
        return [[200, 200],
                [400, 200],
                [400, 280],
                [200, 280]]
    if lines[0].id == 'number_not_mapped':
        return [[50, 50],
                [100, 50],
                [100, 100],
                [50, 100]]
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


def render_dataset(dataset, mastercopy_dir, output_render_dir):
    os.makedirs(output_render_dir, exist_ok=True)
    for page_id, data in dataset.items():
        logger.debug(f'{"RENDERING:":>20s} {page_id}')
        if 'mastercopy_path' not in data:
            logger.warning(f'{"MASTERCOPY PATH MISSING, SKIPPING":>20s} {page_id}')
            continue
        mastercopy_name = os.path.basename(data['mastercopy_path'])
        mastercopy_path = os.path.join(mastercopy_dir, mastercopy_name)
        img = cv2.imread(mastercopy_path, cv2.IMREAD_COLOR)
        if 'chapters' in data:
            for _, _, chapter_bbox in data['chapters']:
                pts = np.asarray(chapter_bbox, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                img = cv2.polylines(img, pts=[pts],
                                    isClosed=True, color=(255, 0, 0), thickness=2)
        if 'numbers' in data:
            for _, _, number_bbox in data['numbers']:
                pts = np.asarray(number_bbox, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                img = cv2.polylines(img, pts=[pts],
                                    isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imwrite(os.path.join(output_render_dir, f'{mastercopy_name}.jpg'), img)


def save_label_studio_storage(dataset, mastercopy_dir, output_label_studio_dir, label_studio_project_name='digilinka',
                              must_include=None, always_include=None):
    images_dir = os.path.join(output_label_studio_dir, 'images')
    tasks_dir = os.path.join(output_label_studio_dir, 'tasks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tasks_dir, exist_ok=True)

    skipped_annotations = 0
    new_annotations = 0

    for page_id, data in dataset.items():
        skip = False
        if must_include is not None:
            always_in = False
            for a in always_include:
                if a in data:
                    always_in = True
            if not always_in:
                for m in must_include:
                    if m not in data:
                        skip = True
                        break
        if skip:
            continue
        logger.debug(f'{"PROCESSING:":>20s} {page_id}')
        if 'mastercopy_path' not in data:
            logger.warning(f'{"MASTERCOPY PATH MISSING, SKIPPING":>20s} {page_id}')
            continue
        mastercopy_name = os.path.basename(data['mastercopy_path'])
        mastercopy_path = os.path.join(mastercopy_dir, mastercopy_name)
        img = cv2.imread(mastercopy_path, cv2.IMREAD_COLOR)
        img_name = f'{mastercopy_name}.jpg'
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, img)

        task_name = f'{mastercopy_name}.json'
        task_path = os.path.join(tasks_dir, task_name)
        label_studio_dict = None
        existing_ids = []
        if os.path.exists(task_path):
            with open(task_path) as f:
                label_studio_dict = json.load(f)
            if 'predictions' in label_studio_dict and 'result' in label_studio_dict['predictions'][0]:
                for result in label_studio_dict['predictions'][0]['result']:
                    existing_ids.append(result['id'])
        results = []

        if 'chapters' in data:
            for chapter_id, chapter, chapter_bbox in data['chapters']:
                if chapter_id in existing_ids:
                    skipped_annotations += 1
                    continue
                x, y, width, height = get_label_studio_coords(chapter_bbox, img.shape)
                result = {
                            "id": chapter_id,
                            "meta": {
                                "text": [chapter]
                            },
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": img.shape[1],
                            "original_height": img.shape[0],
                            "image_rotation": 0,
                            "value": {
                              "rotation": 0,
                              "x": x,
                              "y": y,
                              "width": width,
                              "height": height,
                              "rectanglelabels": [
                                  "kapitola"
                                ]
                            }
                          }
                new_annotations += 1
                results.append(result)

        if 'numbers' in data:
            for number_id, number, number_bbox in data['numbers']:
                if number_id in existing_ids:
                    skipped_annotations += 1
                    continue
                x, y, width, height = get_label_studio_coords(number_bbox, img.shape)
                result = {
                    "id": number_id,
                    "meta": {
                        "text": [number]
                    },
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": img.shape[1],
                    "original_height": img.shape[0],
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "rectanglelabels": [
                            "cislo strany"
                        ]
                    }
                }
                new_annotations += 1
                results.append(result)

        if label_studio_dict is None:
            label_studio_dict = dict()
            label_studio_dict['data'] = {'image': f'/data/local-files/?d={label_studio_project_name}/images/{img_name}'}
            label_studio_dict['predictions'] = [{'model_version': "v0.1",
                                                 'score': "1.0",
                                                 'result': results}]
        else:
            label_studio_dict['predictions'][0]['result'] += results

        with open(task_path, 'w') as f:
            json.dump(label_studio_dict, f, indent=4)

    logger.info(f'SKIPPED ANNOTATIONS: {skipped_annotations}')
    logger.info(f'NEW ANNOTATIONS: {new_annotations}')


def get_label_studio_coords(bbox, img_shape):
    left_top = bbox[0]
    right_bottom = bbox[2]

    width = right_bottom[0] - left_top[0]
    height = right_bottom[1] - left_top[1]
    x = left_top[0]
    y = left_top[1]

    width /= img_shape[1]
    height /= img_shape[0]
    x /= img_shape[1]
    y /= img_shape[0]

    width *= 100
    height *= 100
    x *= 100
    y *= 100

    return x, y, width, height


def create_label_studio_json(left_top, right_bottom, img_shape, img_name, label_studio_project_name="digilinka"):
    label_studio_dict = dict()
    label_studio_dict['data'] = {'image': f'/data/local-files/?d={label_studio_project_name}/images/{img_name}'}
    predictions = []

    width = right_bottom[0] - left_top[0]
    height = right_bottom[1] - left_top[1]
    x = left_top[0]
    y = left_top[1]

    width /= img_shape[1]
    height /= img_shape[0]
    x /= img_shape[1]
    y /= img_shape[0]

    width *= 100
    height *= 100
    x *= 100
    y *= 100

    predictions.append({'model_version': "v0.1",
                        'score': "1.0",
                        'result': [
                            {
                                "id": "result0",
                                "type": "rectanglelabels",
                                "from_name": "label",
                                "to_name": "image",
                                "original_width": img_shape[1],
                                "original_height": img_shape[0],
                                "image_rotation": 0,
                                "value": {
                                    "rotation": 0,
                                    "x": x,
                                    "y": y,
                                    "width": width,
                                    "height": height,
                                    "rectanglelabels": [
                                        "kapitola"
                                    ]
                                }
                             }
                        ]})
    label_studio_dict['predictions'] = predictions
    return json.dumps(label_studio_dict, indent=4)


def get_mastercopy_name(mastercopy_path: str):
    mastercopy_name = os.path.basename(mastercopy_path)
    mastercopy_name = os.path.splitext(mastercopy_name)[0]
    return mastercopy_name


def get_dateset_page_id(volume_uuid: str, mastercopy_path: str):
    return f'{volume_uuid}.{os.path.basename(mastercopy_path)}'


def get_mastercopy_path(mastercopy_elements, mastercopy_id, namespaces):
    mastercopy_element = mastercopy_elements.xpath(f"descendant::mets:file[@ID='{mastercopy_id}']/mets:FLocat",
                                                   namespaces=namespaces)
    if mastercopy_element is None or not mastercopy_element or mastercopy_element[0] is None:
        logger.warning(f'NO MASTER COPY ELEMENT: {mastercopy_element}, SKIPPING\n')
        return None
    mastercopy_element = mastercopy_element[0]
    mastercopy_path = str(mastercopy_element.get("{http://www.w3.org/1999/xlink}href"))
    return mastercopy_path


def compare_orig_to_transcription(orig: str, transcription: str, transcription_edit_distance: float,
                                  max_transcription_relative_distance: float):
    if len(transcription) > len(orig):
        transcription_relative_similarity = ((len(transcription) - transcription_edit_distance) / len(transcription))
    else:
        transcription_relative_similarity = ((len(transcription) + transcription_edit_distance) / len(transcription))
    if 1 - max_transcription_relative_distance <= transcription_relative_similarity <= 1 + max_transcription_relative_distance:
        return True, transcription_relative_similarity
    return False, transcription_relative_similarity


def crop_alignment(align, dim=1):
    for i, al in enumerate(align):
        if al[dim] is not None:
            return i


def print_lxml_element(e):
    logger.info(etree.tostring(e, pretty_print=True))


if __name__ == '__main__':
    main()

