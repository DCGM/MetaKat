import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def get_mods_jsonl(mods_dir, process_three_level_docs=False):
    mods_tree = {}
    logger.info(f'Creating mods tree from {mods_dir} for four level documents')
    four_level_paths = list(Path(mods_dir).glob('*/*/*/*.mods'))
    for i, m in enumerate(four_level_paths):
        m = str(m.relative_to(mods_dir))
        periodic, year, number, page = m.split('/')
        if periodic not in mods_tree:
            mods_tree[periodic] = {}
        if year not in mods_tree[periodic]:
            mods_tree[periodic][year] = {}
        if number not in mods_tree[periodic][year]:
            mods_tree[periodic][year][number] = list()
        mods_tree[periodic][year][number].append(page.replace('.mods', ''))
        if (i + 1) % 500 == 0:
            logger.info(f'{i + 1}/{len(four_level_paths)} files processed')

    if process_three_level_docs:
        logger.info(f'Creating mods tree from {mods_dir} for three level documents')
        three_level_paths = list(Path(mods_dir).glob('*/*/*.mods'))
        for i, m in enumerate(three_level_paths):
            m = str(m.relative_to(mods_dir))
            periodic, year, page = m.split('/')
            if periodic not in mods_tree:
                mods_tree[periodic] = {}
            if year not in mods_tree[periodic]:
                mods_tree[periodic][year] = list()
            if isinstance(mods_tree[periodic][year], list):
                mods_tree[periodic][year].append(page.replace('.mods', ''))
            if (i + 1) % 500 == 0:
                logger.info(f'{i + 1}/{len(three_level_paths)} files processed')

    logger.info(f'Creating mods tree from {mods_dir} for two level documents')
    two_level_paths = list(Path(mods_dir).glob('*/*.mods'))
    for i, m in enumerate(two_level_paths):
        m = str(m.relative_to(mods_dir))
        doc, page = m.split('/')
        if doc not in mods_tree:
            mods_tree[doc] = list()
        if isinstance(mods_tree[doc], list):
            mods_tree[doc].append(page.replace('.mods', ''))
        if (i + 1) % 500 == 0:
            logger.info(f'{i + 1}/{len(two_level_paths)} files processed')

    mods_jsonl = []
    for doc, val in mods_tree.items():
        jsonl = {doc: val}
        mods_jsonl.append(json.dumps(jsonl))

    return mods_jsonl


def get_year_from_doc_mods(mods_path, original_date=False):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    try:
        tree = ET.parse(mods_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    date_element = root.findall(f".//mods:namePart[@type='date']", namespaces=namespaces)
    date = None
    if len(date_element) != 0:
        date = date_element[0].text
    if date is None:
        issued_element = root.findall(f".//mods:dateIssued", namespaces=namespaces)
        if len(issued_element) != 0:
            date = issued_element[0].text
    if date is None:
        date_element = root.findall(f".//mods:date", namespaces=namespaces)
        if len(date_element) != 0:
            date = date_element[0].text
    if date is None:
        return None
    if original_date:
        return date
    # [] - for dates like [1938]
    # ? - for dates like 1938?
    start_end_year = [x.strip('[]?') for x in date.split('-')]
    try:
        int(start_end_year[0])
    except ValueError:
        return None
    try:
        int(start_end_year[1])
    except (ValueError, IndexError):
        return int(start_end_year[0]), int(start_end_year[0])
    return int(start_end_year[0]), int(start_end_year[1])


def get_periodic_frequency_from_doc_mods(mods_path):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    try:
        tree = ET.parse(mods_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    frequency_element = root.findall(f".//mods:frequency", namespaces=namespaces)
    frequency = None
    if len(frequency_element) != 0:
        frequency = frequency_element[0].text
    return frequency


def get_number_from_number_mods(mods_path):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    try:
        tree = ET.parse(mods_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    number_element = root.findall(f".//mods:partNumber", namespaces=namespaces)
    number = None
    if len(number_element) != 0:
        number = number_element[0].text
    try:
        number = int(number)
    except (ValueError, TypeError):
        return None
    return number


page_type_classes = ('Abstract,Advertisement,Appendix,BackCover,BackEndPaper,BackEndSheet,Bibliography,'
                     'Blank,CalibrationTable,Cover,CustomInclude,Dedication,Edge,Errata,FlyLeaf,'
                     'FragmentsOfBookbinding,FrontCover,FrontEndPaper,FrontEndSheet,FrontJacket,'
                     'Frontispiece,Illustration,Impressum,Imprimatur,Index,Jacket,ListOfIllustrations,'
                     'ListOfMaps,ListOfTables,Map,NormalPage,Obituary,Preface,SheetMusic,Spine,Table,'
                     'TableOfContents,TitlePage')
page_type_classes = page_type_classes.split(',')
page_type_classes = {x.lower(): x for x in page_type_classes}


def get_page_type_from_page_mods(mods_path):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    try:
        tree = ET.parse(mods_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    page_type_element = root.findall(f".//mods:part[@type]", namespaces=namespaces)
    if len(page_type_element) != 0:
        page_type = page_type_element[0].get('type')
        if page_type.lower() in page_type_classes:
            return page_type_classes[page_type.lower()]
    return None
