import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MODS_NS = "http://www.loc.gov/mods/v3"
NS = {"mods": MODS_NS}

# Roles that map onto a MetakatVolume/MetakatIssue field. Any other
# marcrelator code (e.g. "oth") has no matching field and is dropped.
ROLE_TERM_TO_FIELD = {
    "aut": "author",
    "ill": "illustrator",
    "pht": "photographer",
    "trl": "translator",
    "edt": "editor",
    "red": "redaktor",
}

# Placeholder values Czech MARC/MODS records use for "unknown" instead of
# omitting the element outright.
SKIP_VALUES = {
    "neuveden", "neuvedeno", "neznámý", "nezjištěn", "nezjištěno",
    "bez nakladatele", "nezname_cislo",
    "s. n.", "s. n.]", "[s. n.]", "[s. n.",
    "s. l.", "s. l.]", "[s. l.]", "[s. l.",
}

FIELD_NAMES = (
    "title", "subTitle", "partName", "partNumber", "dateIssued", "edition",
    "placeTerm", "publisher", "manufacturePublisher", "manufacturePlaceTerm",
    "author", "illustrator", "photographer", "translator", "editor",
    "redaktor", "seriesName", "seriesNumber",
)


def _clean(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if not text or text.lower() in SKIP_VALUES:
        return None
    return text


def _dedup(values: List[str]) -> Optional[List[str]]:
    return list(dict.fromkeys(values)) or None


def _find_mods_element(root: ET.Element) -> Optional[ET.Element]:
    mods = root.find("mods:mods", NS)
    if mods is not None:
        return mods
    if root.tag == f"{{{MODS_NS}}}mods":
        return root
    return None


def _parse_title_info(mods: ET.Element) -> Dict[str, Optional[str]]:
    title_infos = mods.findall("mods:titleInfo", NS)
    primary = next((ti for ti in title_infos if ti.get("usage") == "primary"), None)
    if primary is None:
        primary = next((ti for ti in title_infos if ti.get("type") is None), None)
    if primary is None and title_infos:
        primary = title_infos[0]

    if primary is None:
        return {"title": None, "subTitle": None, "partName": None, "partNumber": None}

    return {
        "title": _clean(primary.findtext("mods:title", namespaces=NS)),
        "subTitle": _clean(primary.findtext("mods:subTitle", namespaces=NS)),
        "partName": _clean(primary.findtext("mods:partName", namespaces=NS)),
        "partNumber": _clean(primary.findtext("mods:partNumber", namespaces=NS)),
    }


def _parse_series(mods: ET.Element) -> Dict[str, Optional[List[str]]]:
    series_name, series_number = [], []
    for related in mods.findall("mods:relatedItem", NS):
        if related.get("type") != "series":
            continue
        title_info = related.find("mods:titleInfo", NS)
        if title_info is None:
            continue
        name = _clean(title_info.findtext("mods:title", namespaces=NS))
        if name:
            series_name.append(name)
        number = _clean(title_info.findtext("mods:partNumber", namespaces=NS))
        if number:
            series_number.append(number)
    return {"seriesName": _dedup(series_name), "seriesNumber": _dedup(series_number)}


def _parse_names(mods: ET.Element) -> Dict[str, Optional[List[str]]]:
    values: Dict[str, List[str]] = {field: [] for field in ROLE_TERM_TO_FIELD.values()}
    for name in mods.findall("mods:name", NS):
        parts = [
            part.text.strip()
            for part in name.findall("mods:namePart", NS)
            if part.text and part.get("type") != "date"
        ]
        full_name = _clean(" ".join(parts))
        if not full_name:
            continue
        for role_term in name.findall(".//mods:roleTerm", NS):
            code = (role_term.text or "").strip().lower()
            field = ROLE_TERM_TO_FIELD.get(code)
            if field and full_name not in values[field]:
                values[field].append(full_name)
    return {field: _dedup(vals) for field, vals in values.items()}


def _parse_origin_info(mods: ET.Element) -> Dict[str, object]:
    publisher, manufacture_publisher = [], []
    manufacture_place = []
    place, date_issued = None, None

    for origin in mods.findall("mods:originInfo", NS):
        is_manufacture = origin.get("eventType") == "manufacture"

        for pub in origin.findall("mods:publisher", NS):
            value = _clean(pub.text)
            if value:
                (manufacture_publisher if is_manufacture else publisher).append(value)

        for place_term in origin.findall("mods:place/mods:placeTerm", NS):
            if place_term.get("type") != "text":
                continue
            value = _clean(place_term.text)
            if not value:
                continue
            if is_manufacture:
                manufacture_place.append(value)
            elif place is None:
                place = value

        if not is_manufacture and date_issued is None:
            date_issued = _clean(origin.findtext("mods:dateIssued", namespaces=NS))

    edition = _clean(mods.findtext("mods:originInfo/mods:edition", namespaces=NS))

    return {
        "placeTerm": place,
        "dateIssued": date_issued,
        "edition": edition,
        "publisher": _dedup(publisher),
        "manufacturePublisher": _dedup(manufacture_publisher),
        "manufacturePlaceTerm": _dedup(manufacture_place),
    }


def parse_mods(xml_text: str) -> Dict[str, object]:
    """Parse a single MODS record (optionally wrapped in modsCollection) into a flat
    dict whose keys/types match the corresponding fields on MetakatTitle/MetakatVolume/
    MetakatIssue, so results can be assigned directly onto those models.
    """
    result: Dict[str, object] = {field: None for field in FIELD_NAMES}

    root = ET.fromstring(xml_text)
    mods = _find_mods_element(root)
    if mods is None:
        return result

    result.update(_parse_title_info(mods))
    result.update(_parse_series(mods))
    result.update(_parse_names(mods))
    result.update(_parse_origin_info(mods))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mods-file', required=True, type=str)
    parser.add_argument('--output-file', required=True, type=str)
    parser.add_argument('--logging-level', default=logging.INFO)
    return parser.parse_args()


def main():
    args = parse_args()

    log_formatter = logging.Formatter('PARSE MODS - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    with open(args.mods_file, encoding='utf-8') as f:
        xml_text = f.read()

    result = parse_mods(xml_text)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f'Wrote parsed MODS fields to {args.output_file}')


if __name__ == '__main__':
    main()
