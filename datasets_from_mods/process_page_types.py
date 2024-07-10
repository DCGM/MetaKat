import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from glob import glob
import multiprocessing as mp
from functools import partial
from uuid import uuid4
from multiprocessing_logging import install_mp_handler

import cv2
import numpy as np

from tools.mods_helper import get_page_type_from_page_mods, get_year_from_doc_mods, page_type_classes, \
    get_number_from_number_mods

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mods-dir', required=True, type=str)
    parser.add_argument('--ids-jsonl', required=True, type=str)
    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)

    parser.add_argument('--from-year', type=int)
    parser.add_argument('--to-year', type=int)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--book', action='store_true')
    group.add_argument('--periodic', action='store_true')

    parser.add_argument('--max-years-per-periodic', type=int)
    parser.add_argument('--max-numbers-per-year', type=int)

    parser.add_argument('--max-samples-per-class-per-doc', default=1000, type=int)
    parser.add_argument('--max-samples-per-class', default=10500, type=int)

    parser.add_argument('--existing-output-dir', type=str)
    parser.add_argument('--existing-counter', type=str)

    parser.add_argument('--invalid-doc-uuids', type=str)
    parser.add_argument('--invalid-page-uuids', type=str)

    parser.add_argument('--valid-doc-uuids', type=str)
    parser.add_argument('--valid-page-uuids', type=str)

    parser.add_argument('--export-orig-images', action='store_true')

    parser.add_argument('--process-undefined-page-types', action='store_true')

    parser.add_argument('--num-processes', default=4, type=int)

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    doc_type = 'book' if args.book else 'periodic'

    os.makedirs(args.output_dir, exist_ok=True)

    log_formatter = logging.Formatter('%(asctime)s - GET PAGE TYPE - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(log_formatter)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.setLevel(args.logging_level)
    if args.num_processes > 1:
        install_mp_handler(logger)

    logger.info(' '.join(sys.argv))

    if args.existing_output_dir is not None:
        args.invalid_doc_uuids = os.path.join(args.existing_output_dir, 'doc_uuids.txt')
        args.invalid_page_uuids = os.path.join(args.existing_output_dir, 'page_uuids.txt')
        args.existing_counter = os.path.join(args.existing_output_dir, 'counter.json')

    valid_page_uuids = None
    if args.valid_page_uuids is not None and os.path.exists(args.valid_page_uuids):
        with open(args.valid_page_uuids) as f:
            valid_page_uuids = f.readlines()
            valid_page_uuids = {uuid.strip(): 0 for uuid in valid_page_uuids if uuid.strip() != ''}

    invalid_page_uuids = None
    if args.invalid_page_uuids is not None and os.path.exists(args.invalid_page_uuids):
        with open(args.invalid_page_uuids) as f:
            invalid_page_uuids = f.readlines()
            invalid_page_uuids = {uuid.strip(): 0 for uuid in invalid_page_uuids if uuid.strip() != ''}

    doc_mods_dir_paths = glob(os.path.join(args.mods_dir, '*/'))
    doc_mods_dir_paths = [doc_mods_dir_path.rstrip('/') for doc_mods_dir_path in doc_mods_dir_paths]

    valid_doc_uuids = None
    if args.valid_doc_uuids is not None and os.path.exists(args.valid_doc_uuids):
        with open(args.valid_doc_uuids) as f:
            valid_doc_uuids = f.readlines()
            valid_doc_uuids = {uuid.strip(): 0 for uuid in valid_doc_uuids if uuid.strip() != ''}

    if valid_doc_uuids is not None:
        new_doc_mods_dir_paths = []
        for doc_mods_dir_path in doc_mods_dir_paths:
            doc_id = os.path.basename(doc_mods_dir_path)
            if doc_id in valid_doc_uuids:
                new_doc_mods_dir_paths.append(doc_mods_dir_path)
        doc_mods_dir_paths = new_doc_mods_dir_paths

    invalid_doc_uuids = None
    if args.invalid_doc_uuids is not None and os.path.exists(args.invalid_doc_uuids):
        with open(args.invalid_doc_uuids) as f:
            invalid_doc_uuids = f.readlines()
            invalid_doc_uuids = {uuid.strip(): 0 for uuid in invalid_doc_uuids if uuid.strip() != ''}

    if invalid_doc_uuids is not None:
        new_doc_mods_dir_paths = []
        for doc_mods_dir_path in doc_mods_dir_paths:
            doc_id = os.path.basename(doc_mods_dir_path)
            if doc_id not in invalid_doc_uuids:
                new_doc_mods_dir_paths.append(doc_mods_dir_path)
        doc_mods_dir_paths = new_doc_mods_dir_paths

    random.shuffle(doc_mods_dir_paths)

    old_counter = {}
    if args.existing_counter is not None and os.path.exists(args.existing_counter):
        with open(args.existing_counter) as f:
            old_counter = json.load(f)

    ids_jsonl = {}
    with open(args.ids_jsonl) as f:
        for line in f.readlines():
            key, val = list(json.loads(line).items())[0]
            ids_jsonl[key] = val

    now = time.time()
    if args.num_processes > 1:
        with mp.Manager() as manager:
            counter = manager.dict()
            counter['docs'] = 0
            for page_type_class in page_type_classes.values():
                if page_type_class in old_counter:
                    counter[page_type_class] = old_counter[page_type_class]
                else:
                    counter[page_type_class] = 0
            logger.info(counter)
            results = manager.dict()
            for page_type_class in page_type_classes.values():
                results[page_type_class] = manager.list()
            processed_doc_uuids = manager.list()
            processed_page_uuids = manager.list()
            process_page_types_multi = partial(process_doc_page_types,
                                               doc_type=doc_type,
                                               counter=counter,
                                               results=results,
                                               processed_doc_uuids=processed_doc_uuids,
                                               processed_page_uuids=processed_page_uuids,
                                               ids_jsonl=ids_jsonl,
                                               images_dir=args.images_dir,
                                               output_dir=args.output_dir,
                                               max_years_per_periodic=args.max_years_per_periodic,
                                               max_numbers_per_year=args.max_numbers_per_year,
                                               max_samples_per_class_per_doc=args.max_samples_per_class_per_doc,
                                               max_samples_per_class=args.max_samples_per_class,
                                               invalid_page_uuids=invalid_page_uuids,
                                               valid_page_uuids=valid_page_uuids,
                                               from_year=args.from_year,
                                               to_year=args.to_year,
                                               export_orig_images=args.export_orig_images,
                                               process_undefined_page_types=args.process_undefined_page_types)
            logger.info(args.num_processes)
            with manager.Pool(processes=args.num_processes) as pool:
                pool.map(process_page_types_multi, doc_mods_dir_paths)
            # `d` is a DictProxy object that can be converted to dict
            counter = dict(counter)
            results = dict(results)
            for r in results:
                results[r] = list(results[r])
            processed_doc_uuids = set(processed_doc_uuids)
            processed_page_uuids = set(processed_page_uuids)
    else:
        counter = {}
        counter['docs'] = 0
        for page_type_class in page_type_classes.values():
            if page_type_class in old_counter:
                counter[page_type_class] = old_counter[page_type_class]
            else:
                counter[page_type_class] = 0
        logger.info(counter)
        results = {}
        for page_type_class in page_type_classes.values():
            results[page_type_class] = []
        processed_doc_uuids = []
        processed_page_uuids = []
        for doc_mods_dir_path in doc_mods_dir_paths:
            process_doc_page_types(top_level_mods_dir_path=doc_mods_dir_path,
                                   doc_type=doc_type,
                                   counter=counter,
                                   results=results,
                                   processed_doc_uuids=processed_doc_uuids,
                                   processed_page_uuids=processed_page_uuids,
                                   ids_jsonl=ids_jsonl,
                                   images_dir=args.images_dir,
                                   output_dir=args.output_dir,
                                   max_years_per_periodic=args.max_years_per_periodic,
                                   max_numbers_per_year=args.max_numbers_per_year,
                                   max_samples_per_class_per_doc=args.max_samples_per_class_per_doc,
                                   max_samples_per_class=args.max_samples_per_class,
                                   invalid_page_uuids=invalid_page_uuids,
                                   valid_page_uuids=valid_page_uuids,
                                   from_year=args.from_year,
                                   to_year=args.to_year,
                                   export_orig_images=args.export_orig_images,
                                   process_undefined_page_types=args.process_undefined_page_types)

    logger.info(counter)

    os.makedirs(os.path.join(args.output_dir, 'ann'), exist_ok=True)

    processed_doc_uuids = list(set(processed_doc_uuids))
    processed_page_uuids = list(set(processed_page_uuids))

    for page_type, annotations in results.items():
        result_json_path = os.path.join(args.output_dir, 'ann', f'{page_type}.json')
        if os.path.exists(result_json_path):
            with open(result_json_path) as f:
                all_annotations = json.load(f)
            all_annotations.extend(annotations)
        else:
            all_annotations = annotations
        with open(result_json_path, 'w') as f:
            json.dump(all_annotations, f)

    if args.existing_output_dir is not None:
        existing_doc_uuids_path = os.path.join(args.existing_output_dir, 'doc_uuids.txt')
        if os.path.exists(existing_doc_uuids_path):
            with open(existing_doc_uuids_path) as f:
                processed_doc_uuids.extend([x.strip() for x in f.readlines()])
            processed_doc_uuids = list(set(processed_doc_uuids))
    doc_uuids_path = os.path.join(args.output_dir, 'doc_uuids.txt')
    with open(doc_uuids_path, 'w') as f:
        f.write('\n'.join(processed_doc_uuids))

    if args.existing_output_dir is not None:
        existing_page_uuids_path = os.path.join(args.existing_output_dir, 'page_uuids.txt')
        if os.path.exists(existing_page_uuids_path):
            with open(existing_page_uuids_path) as f:
                processed_page_uuids.extend([x.strip() for x in f.readlines()])
            processed_page_uuids = list(set(processed_page_uuids))
    page_uuids_path = os.path.join(args.output_dir, 'page_uuids.txt')
    with open(page_uuids_path, 'w') as f:
        f.write('\n'.join(processed_page_uuids))

    counter_path = os.path.join(args.output_dir, 'counter.json')
    counter.pop('docs')
    with open(counter_path, 'w') as f:
        json.dump(counter, f)

    logger.info(f'Elapsed time: {time.time() - now}')


def process_doc_page_types(top_level_mods_dir_path, doc_type, counter, results,
                           processed_doc_uuids, processed_page_uuids,
                           ids_jsonl, images_dir, output_dir,
                           max_years_per_periodic=1, max_numbers_per_year=1,
                           max_samples_per_class_per_doc=1, max_samples_per_class=10,
                           invalid_page_uuids=None,
                           valid_page_uuids=None,
                           from_year=None, to_year=None,
                           export_orig_images=False,
                           process_undefined_page_types=False):
    if top_level_mods_dir_path.endswith('.mods'):
        return

    doc_id = os.path.basename(top_level_mods_dir_path)

    is_periodic_dir = is_dir_periodic(top_level_mods_dir_path)
    if doc_type == 'periodic' and is_periodic_dir:
        year_mods_dir_paths = process_periodic(periodic_mods_dir_path=top_level_mods_dir_path,
                                               max_years_per_periodic=max_years_per_periodic,
                                               from_year=from_year,
                                               to_year=to_year)
        for year_mods_dir_path in year_mods_dir_paths:
            number_mods_dir_paths = process_year(year_mods_dir_path=year_mods_dir_path,
                                                 max_numbers_per_year=max_numbers_per_year)
            for number_mods_dir_path in number_mods_dir_paths:
                counter['docs'] += 1
                number_page_ids = ids_jsonl[os.path.basename(top_level_mods_dir_path)]\
                                           [os.path.basename(year_mods_dir_path)]\
                                           [os.path.basename(number_mods_dir_path)]
                doc_id_tmp = os.path.basename(top_level_mods_dir_path).replace('uuid:', '')
                doc_images_dir_path = os.path.join(images_dir, f'{doc_id_tmp}.images')
                if not os.path.exists(doc_images_dir_path):
                    doc_images_dir_path = os.path.join(images_dir, doc_id_tmp)
                if not os.path.exists(doc_images_dir_path):
                    logger.warning(f'Images dir not found: {doc_images_dir_path}|.images')
                    continue
                process_doc(doc_mods_dir_path=number_mods_dir_path,
                            counter=counter,
                            results=results,
                            processed_doc_uuids=processed_doc_uuids,
                            processed_page_uuids=processed_page_uuids,
                            doc_id=doc_id,
                            doc_mods_path=f'{number_mods_dir_path}.mods',
                            doc_page_ids=number_page_ids,
                            doc_images_dir_path=doc_images_dir_path,
                            output_dir=output_dir,
                            max_samples_per_class_per_doc=max_samples_per_class_per_doc,
                            max_samples_per_class=max_samples_per_class,
                            invalid_page_uuids=invalid_page_uuids,
                            valid_page_uuids=valid_page_uuids,
                            export_orig_images=export_orig_images,
                            process_undefined_page_types=process_undefined_page_types)

        return
    elif doc_type == 'book' and not is_periodic_dir:
        counter['docs'] += 1
        doc_id_tmp = doc_id.replace('uuid:', '')
        doc_images_dir_path = os.path.join(images_dir, f'{doc_id_tmp}.images')
        if not os.path.exists(doc_images_dir_path):
            doc_images_dir_path = os.path.join(images_dir, doc_id_tmp)
        if not os.path.exists(doc_images_dir_path):
            logger.warning(f'Images dir not found: {doc_images_dir_path}|.images')
            return
        # logger.info(f'Book: {doc}')
        doc_page_ids = ids_jsonl[os.path.basename(top_level_mods_dir_path)]
        process_doc(doc_mods_dir_path=top_level_mods_dir_path,
                    counter=counter,
                    results=results,
                    processed_doc_uuids=processed_doc_uuids,
                    processed_page_uuids=processed_page_uuids,
                    doc_id=doc_id,
                    doc_mods_path=f'{top_level_mods_dir_path}.mods',
                    doc_page_ids=doc_page_ids,
                    doc_images_dir_path=doc_images_dir_path,
                    output_dir=output_dir,
                    max_samples_per_class_per_doc=max_samples_per_class_per_doc,
                    max_samples_per_class=max_samples_per_class,
                    invalid_page_uuids=invalid_page_uuids,
                    valid_page_uuids=valid_page_uuids,
                    from_year=from_year,
                    to_year=to_year,
                    export_orig_images=export_orig_images,
                    process_undefined_page_types=process_undefined_page_types)


def process_periodic(periodic_mods_dir_path,
                     max_years_per_periodic=1,
                     from_year=None, to_year=None):
    years = OrderedDict()
    for year_mods_dir_path in glob(os.path.join(periodic_mods_dir_path, '*/')):
        year_mods_dir_path = year_mods_dir_path.rstrip('/')
        year_mods_path = f'{year_mods_dir_path}.mods'
        year_years = get_year_from_doc_mods(year_mods_path)
        if not are_years_valid(year_years, from_year, to_year):
            continue
        years[year_years[0]] = year_mods_dir_path
    if len(years) == 0:
        return []
    years = OrderedDict(sorted(years.items(), key=lambda x: x[0]))
    if max_years_per_periodic is None:
        max_years_per_periodic = len(years)
    years_chunked = list(chunks(list(years.values()), max_years_per_periodic))
    year_mods_dir_paths = [random.choice(chunk) for chunk in years_chunked if len(chunk) > 0]
    return year_mods_dir_paths


def process_year(year_mods_dir_path,
                 max_numbers_per_year=1):
    numbers = OrderedDict()
    for number_mods_dir_path in glob(os.path.join(year_mods_dir_path, '*/')):
        number_mods_dir_path = number_mods_dir_path.rstrip('/')
        number_mods_path = f'{number_mods_dir_path}.mods'
        number = get_number_from_number_mods(number_mods_path)
        if number is None:
            numbers[-1] = number_mods_dir_path
        else:
            numbers[number] = number_mods_dir_path
    if len(numbers) == 0:
        return []
    numbers = OrderedDict(sorted(numbers.items(), key=lambda x: x[0]))
    if max_numbers_per_year is None:
        max_numbers_per_year = len(numbers)
    numbers_chunked = list(chunks(list(numbers.values()), max_numbers_per_year))
    number_mods_dir_paths = [random.choice(chunk) for chunk in numbers_chunked if len(chunk) > 0]
    return number_mods_dir_paths


def process_doc(doc_mods_dir_path, counter, results,
                processed_doc_uuids, processed_page_uuids,
                doc_id, doc_mods_path, doc_page_ids, doc_images_dir_path, output_dir,
                max_samples_per_class_per_doc=1, max_samples_per_class=10,
                invalid_page_uuids=None,
                valid_page_uuids=None,
                from_year=None, to_year=None,
                export_orig_images=False,
                process_undefined_page_types=False):
    if counter['docs'] % 100 == 0:
        logger.info(counter)

    years = get_year_from_doc_mods(doc_mods_path)
    if not are_years_valid(years, from_year, to_year):
        return

    page_map = get_doc_page_map(doc_mods_dir_path=doc_mods_dir_path,
                                doc_page_ids=doc_page_ids,
                                doc_images_dir_path=doc_images_dir_path)
    if page_map is None:
        return

    add_page_type_to_page_map(page_map, add_undefined_page_type=process_undefined_page_types)

    valid_page_ids = defaultdict(list)
    for page_id, page_val in list(page_map.items()):
        if not page_val['invalid'] and \
            page_val['image_path'] is not None and \
            page_val['page_type'] is not None and \
                counter[page_val['page_type']] < max_samples_per_class and \
                (invalid_page_uuids is None or page_id not in invalid_page_uuids) and \
                (valid_page_uuids is None or page_id in valid_page_uuids):
            valid_page_ids[page_val['page_type']].append(page_id)

    if len(valid_page_ids) == 0:
        return

    for page_type, page_ids in list(valid_page_ids.items()):
        random.shuffle(page_ids)
        valid_page_ids[page_type] = page_ids[:max_samples_per_class_per_doc]

    valid_page_ids = {page_id for page_ids in valid_page_ids.values() for page_id in page_ids}

    page_map_list = list(page_map.items())
    for i, (page_id, page_val) in enumerate(page_map_list):
        if page_id not in valid_page_ids:
            continue
        previous_image_path = None
        previous_pages = 0
        next_image_path = None
        next_pages = 0
        if i > 0:
            previous_image_path = page_map_list[i - 1][1]['image_path']
            previous_pages = i
        if i < len(page_map_list) - 1:
            next_image_path = page_map_list[i + 1][1]['image_path']
            next_pages = len(page_map_list) - i - 1

        if previous_image_path is None:
            previous_image_path = 'black'
        if next_image_path is None:
            next_image_path = 'black'

        out = create_image_and_annotation(image_path=page_val['image_path'],
                                          page_id=page_id,
                                          max_height=1000,
                                          previous_image_path=previous_image_path,
                                          previous_pages=previous_pages,
                                          next_image_path=next_image_path,
                                          next_pages=next_pages)
        if out is not False and counter[page_val['page_type']] < max_samples_per_class:
            counter[page_val['page_type']] += 1
            img, annotations = out
            images_dir = os.path.join(output_dir, 'images', page_val['page_type'])
            os.makedirs(images_dir, exist_ok=True)
            cv2.imwrite(os.path.join(images_dir, f'{page_id}.jpg'), img)
            if export_orig_images:
                images_orig_dir = os.path.join(output_dir, 'images_orig', page_val['page_type'])
                os.makedirs(images_orig_dir, exist_ok=True)
                orig_image_path = os.path.join(images_orig_dir, f'{page_id}.jpg')
                if os.path.exists(page_val['image_path']):
                    shutil.copy2(page_val['image_path'], orig_image_path)
            logger.info(f'{doc_mods_dir_path} {page_id} {page_val["page_type"]}')
            results[page_val['page_type']].append(annotations)
            processed_doc_uuids.append(doc_id)
            processed_page_uuids.append(page_id)


def create_image_and_annotation(image_path,
                                page_id,
                                max_height=1000,
                                previous_image_path=None,
                                previous_pages=None,
                                next_image_path=None,
                                next_pages=None):
    img = cv2.imread(image_path)
    if img is None:
        return False
    if img.shape[0] > max_height:
        img = cv2.resize(img, [int(img.shape[1] * (max_height / img.shape[0])), max_height])
    xy_left_top = [0, 0]
    img_shape = img.shape

    if previous_image_path is not None:
        img_prev = get_side_image(previous_image_path, img_shape)
        img = np.hstack((img_prev, img))
        xy_left_top[0] = img_prev.shape[1]

        if previous_pages is not None:
            img = cv2.putText(img, str(previous_pages), (50, 170), cv2.FONT_HERSHEY_SIMPLEX,
                              6, (255, 0, 0), 12, cv2.LINE_AA)

    if next_image_path is not None:
        img_next = get_side_image(next_image_path, img_shape)
        img = np.hstack((img, img_next))

        if next_pages is not None:
            img = cv2.putText(img, str(next_pages), (img.shape[1] - img_next.shape[1] + 50, 170), cv2.FONT_HERSHEY_SIMPLEX,
                              6, (255, 0, 0), 12, cv2.LINE_AA)

    image_name = f'{page_id}.jpg'

    annotation_dict = {}
    annotation_dict['img_path'] = image_name
    annotation_dict['positions'] = [{'ignore': False,
                                     'uuid': str(uuid4()),
                                     'rect': [xy_left_top[0], xy_left_top[1], img_shape[1], img_shape[0]]}]
    annotation_dict['path'] = image_name
    annotation_dict['image_name'] = image_name
    annotation_dict['aratio'] = 1.0
    annotation_dict['uuid'] = page_id

    return img, annotation_dict


def get_side_image(image_path, img_shape, min_width=800):
    if image_path == 'black':
        img = np.full((img_shape[0], min_width, 3), 150.0)
    else:
        img = cv2.imread(image_path)
        if img is None:
            img = np.full((img_shape[0], min_width, 3), 150.0)
        else:
            img = cv2.resize(img, [int(img.shape[1] * (img_shape[0] / img.shape[0])), img_shape[0]])
    if img.shape[1] < min_width:
        img = np.hstack((img, np.full((img_shape[0], min_width - img.shape[1], 3), 150.0)))
    return img


def get_doc_page_map(doc_mods_dir_path, doc_page_ids, doc_images_dir_path):
    doc_id = os.path.basename(doc_mods_dir_path).replace('uuid:', '')
    mods = glob(os.path.join(doc_mods_dir_path, '*.mods'))
    if not doc_page_ids:
        return None
    page_map = OrderedDict()
    for i, page_id in enumerate(doc_page_ids):
        page_id = page_id.strip()
        image_path = os.path.join(doc_images_dir_path, f'{page_id}.jpg')
        if not os.path.exists(image_path):
            image_path = None
        page_map[page_id] = {'order': i, 'mods_path': '', 'image_path': image_path, 'invalid': False, 'page_type': None}
    if len(page_map) == 0:
        logger.warning(f'No page ids found for {doc_page_ids}')
        return None
    for mod in mods:
        page_id = os.path.splitext(os.path.basename(mod))[0]
        if page_id in page_map:
            page_map[page_id]['mods_path'] = mod
        else:
            logger.warning(f'mods {mod} not in {doc_id}.ids')
    return page_map


def add_page_type_to_page_map(page_map, add_undefined_page_type=False):
    for page_id, page_val in list(page_map.items()):
        if page_val['mods_path'] != '':
            page_map[page_id]['page_type'] = get_page_type_from_page_mods(page_val['mods_path'])
        elif add_undefined_page_type:
            page_map[page_id]['page_type'] = 'Undefined'


def are_years_valid(years, from_year, to_year):
    if from_year is None and to_year is None:
        return True
    if years is None:
        return False
    if from_year is not None and to_year is not None:
        if years[0] < from_year or years[1] > to_year:
            return False
    if from_year is not None:
        if years[0] < from_year:
            return False
    if to_year is not None:
        if years[1] > to_year:
            return False
    return True


def is_dir_periodic(dir_path):
    for dir_path in os.listdir(dir_path):
        if not dir_path.endswith('.mods'):
            return True


# https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


if __name__ == '__main__':
    main()
