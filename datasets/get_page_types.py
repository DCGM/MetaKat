import argparse
import json
import logging
import os
import random
import sys
import time
from uuid import uuid4

import cv2
import numpy as np
from lxml import etree

from datasets.parse_dataset import get_mastercopy_path, get_mastercopy_name, get_dateset_page_id

logger = logging.getLogger(__name__)

page_type_classes = ('TitlePage,Table,TableOfContents,Index,Jacket,FrontEndSheet,FrontCover,BackEndSheet,BackCover,'
                     'Blank,Sheetmusic,Advertisement,Map,FrontJacket,FlyLeaf,ListOfIllustrations,Illustration,Spine,'
                     'CalibrationTable,Cover,Edge,ListOfTables,FrontEndPaper,BackEndPaper,ListOfMaps,Bibliography,'
                     'CustomInclude,Frontispiece,Errata,FragmentsOfBookbinding,BackEndpaper,FrontEndpaper,Preface,'
                     'Abstract,Dedication,Imprimatur,Impressum,Obituary,NormalPage')
page_type_classes = page_type_classes.split(',')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mets', required=True, type=str)
    parser.add_argument('--mastercopy-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - GET PAGE TYPE - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler_sender = logging.StreamHandler()
    handler_sender.setFormatter(log_formatter)
    logger.addHandler(handler_sender)
    logger.propagate = False
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    get_page_types(args.mets, args.mastercopy_dir, args.output_dir)


def get_page_types(mets, mastercopy_dir, output_dir):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    tree = etree.parse(mets)

    volume_elements = tree.xpath(f"//mods:mods[contains(@ID, 'VOLUME')]",
                                 namespaces=namespaces)
    volume_uuid_element_content = volume_elements[0].xpath(f"descendant::mods:identifier[contains(@type, 'uuid')]/text()",
                                                           namespaces=namespaces)[0]
    page_elements = tree.xpath(f"//mets:div[boolean(@TYPE)][contains(@ID, 'PAGE')]",
                               namespaces=namespaces)
    mastercopy_elements = tree.xpath(f"//mets:fileGrp[contains(@ID, 'MC_IMGGRP')]",
                                     namespaces=namespaces)[0]
    out = []
    page_types = []

    for page_element in page_elements:
        mastercopy_id = page_element[0].get('FILEID')
        logger.debug(f'{"MASTER COPY ID:":>20s} {mastercopy_id}')
        mastercopy_path = get_mastercopy_path(mastercopy_elements, mastercopy_id, namespaces)
        logger.debug(f'{"MASTER COPY PATH:":>20s} {mastercopy_path}')
        mastercopy_name = get_mastercopy_name(mastercopy_path)
        logger.debug(f'{"MASTER COPY NAME:":>20s} {mastercopy_name}')
        dataset_page_id = get_dateset_page_id(volume_uuid=volume_uuid_element_content,
                                              mastercopy_path=mastercopy_path)
        logger.info(f'{"DATASET PAGE ID:":>20s} {dataset_page_id}')
        page_type = page_element.get('TYPE')
        for page_type_class in page_type_classes:
            if page_type_class.lower() == page_type.lower():
                page_type = page_type_class
        logger.info(f'{"PAGE TYPE:":>20s} {page_type}')
        logger.info('')

        page_types.append(page_type)
        out.append([mastercopy_dir, mastercopy_path, output_dir, page_type, dataset_page_id])

    final_out_start_indexes = get_doc_pad_type_pages(page_types)
    final_out_end_indexes = list(len(page_types) - 1 - np.asarray(get_doc_pad_type_pages(page_types[::-1])))
    all_indexes = list(range(len(out)))
    random.shuffle(all_indexes)
    final_random_indexes = all_indexes[:10]
    final_no_normal_page_indexes = []
    for i, pt in enumerate(page_types):
        if pt != 'NormalPage':
            final_no_normal_page_indexes.append(i)
    final_out_indexes = final_out_start_indexes + final_out_end_indexes + final_random_indexes + final_no_normal_page_indexes
    final_out_indexes = sorted(list(set(final_out_indexes)))

    for i in final_out_indexes:
        export_page_to_output_dir_and_json(*out[i])



def get_doc_pad_type_pages(page_types, max_subsequent_normal_pages=5):
    final_out_indexes = []
    normal_page_counter = 0
    for i, pt in enumerate(page_types):
        if normal_page_counter == max_subsequent_normal_pages:
            final_out_indexes = final_out_indexes[:-max_subsequent_normal_pages]
            break

        if pt == 'NormalPage':
            normal_page_counter += 1
        else:
            normal_page_counter = 0

        final_out_indexes.append(i)

    return final_out_indexes


def export_page_to_output_dir_and_json(mastercopy_dir, mastercopy_path, output_dir, page_type, dataset_page_id, max_height=1000):
    logger.info(f'{"COPYING:":>20s} {mastercopy_path} - {dataset_page_id} - {page_type}')
    img = cv2.imread(os.path.join(mastercopy_dir, os.path.basename(mastercopy_path)))
    if img.shape[0] > max_height:
        img = cv2.resize(img, [int(img.shape[1] * (max_height / img.shape[0])), max_height])
    os.makedirs(os.path.join(output_dir, 'images', page_type), exist_ok=True)
    image_name = f'{dataset_page_id}.{page_type}.jpg'
    cv2.imwrite(os.path.join(output_dir, 'images', page_type, image_name), img)

    annotation_dict = {}
    annotation_dict['img_path'] = image_name
    annotation_dict['positions'] = [{'ignore': False,
                                     'uuid': str(uuid4()),
                                     'rect': [0, 0, img.shape[1], img.shape[0]]}]
    annotation_dict['path'] = image_name
    annotation_dict['image_name'] = image_name
    annotation_dict['aratio'] = 1.0
    annotation_dict['uuid'] = str(uuid4())

    annotation_json_path = os.path.join(output_dir, f'{page_type}.json')
    all_annotations = []
    if os.path.exists(annotation_json_path):
        with open(annotation_json_path) as f:
            all_annotations = json.load(f)
    all_annotations.append(annotation_dict)
    with open(annotation_json_path, 'w') as f:
        json.dump(all_annotations, f)


if __name__ == '__main__':
    main()
