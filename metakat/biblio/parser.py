# autor: Marie Pařilová
# datum: 12.5.2025
# Nástroj pro poloautomatické zarovnání OCR a metadat titulních stran knih.
# Tato práce je součástí bakalářské práce.

import os
import json
import argparse
from lxml import etree
import xml.etree.ElementTree as ET
import ftfy
import re
from rapidfuzz import fuzz
import sys
import unicodedata
sys.path.append('/MetaKat/pero-ocr/pero_ocr/core')
from force_alignment import align_text_to_image
from pero_ocr.core import layout

radek = 1 # Pomocna promenna pro nezarovnane udaje
nezarovnane_annotace = []

def fix_text(text):
    '''Oprava textu na správný formát'''
    return ftfy.fix_text(text) if text else text

def normalize_text(text):
    '''Normalizace textu (odstranění interpunkce, převedení na malé písmena a ASCII)'''
    text = re.sub(r'[„“"\'.,;:!?(){}[\]]', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text.strip().lower()

def remove_numbers(text):
    """Odstraní všechna čísla z textu."""
    text = re.sub(r'-', '', text)
    return re.sub(r'\d+', '', text)

def parse_mods(file_path):
    '''Zpracování XML souboru ve formátu MODS a extrakce relevantních informací'''
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = etree.parse(file)
    root = tree.getroot()

    ns = {'mods': 'http://www.loc.gov/mods/v3'}
    data = {}

    def add_if_not_none(key, value):
        '''Přidá klíč-hodnotu do slovníku, pokud hodnota není None nebo "N/A"'''
        if value is not None and value != "N/A":
            data[key] = value

    # Název
    title = root.xpath(".//mods:titleInfo[not(ancestor::mods:relatedItem)]/mods:title", namespaces=ns)
    add_if_not_none('title', fix_text(title[0].text) if title else None)

    # Autoři
    authors = []
    for author in root.xpath(".//mods:name[@type='personal']", namespaces=ns):
        role_terms = author.xpath("mods:role/mods:roleTerm", namespaces=ns)
        if any(role.text == 'aut' for role in role_terms):
            name_parts = author.xpath("mods:namePart", namespaces=ns)
            name = " ".join(fix_text(part.text) for part in name_parts if part.text)
            name = remove_numbers(name)
            if name:
                authors.append(name)
    add_if_not_none('authors', authors if authors else None)

    # Překladatelé
    translators = []
    for translator in root.xpath(".//mods:name[@type='personal']", namespaces=ns):
        role_terms = translator.xpath("mods:role/mods:roleTerm", namespaces=ns)
        if any(role.text == 'trl' for role in role_terms):
            name_parts = translator.xpath("mods:namePart", namespaces=ns)
            name = " ".join(fix_text(part.text) for part in name_parts if part.text)
            name = remove_numbers(name)
            if name:
                translators.append(name)
    add_if_not_none('translators', translators if translators else None)

    # Ilustrátoři
    illustrators = []
    for illustrator in root.xpath(".//mods:name[@type='personal']", namespaces=ns):
        role_terms = illustrator.xpath("mods:role/mods:roleTerm", namespaces=ns)
        if any(role.text == 'ill' for role in role_terms):
            name_parts = illustrator.xpath("mods:namePart", namespaces=ns)
            name = " ".join(fix_text(part.text) for part in name_parts if part.text)
            name = remove_numbers(name)
            if name:
                illustrators.append(name)
    add_if_not_none('illustrators', illustrators if illustrators else None)

    # Editoři
    editors = []
    for editor in root.xpath(".//mods:name[@type='personal']", namespaces=ns):
        role_terms = editor.xpath("mods:role/mods:roleTerm", namespaces=ns)
        if any(role.text == 'edt' for role in role_terms):
            name_parts = editor.xpath("mods:namePart", namespaces=ns)
            name = " ".join(fix_text(part.text) for part in name_parts if part.text)
            name = remove_numbers(name)
            if name:
                editors.append(name)
    add_if_not_none('editors', editors if editors else None)

    # Datum vydání (Date Issued)
    date_issued_elem = root.find(".//mods:originInfo/mods:dateIssued", ns)
    if date_issued_elem is not None and date_issued_elem.text:
        date_issued_text = fix_text(date_issued_elem.text)
        date_issued_text = date_issued_text.replace("v roce ", "").strip()
        add_if_not_none('date_issued', date_issued_text)

    # Místo vydání
    place_elem = root.find(".//mods:originInfo/mods:place/mods:placeTerm[@type='text']", ns)
    if place_elem is not None and place_elem.text:
        place_text = fix_text(place_elem.text)
        place_text = place_text.replace("v ", "").strip()
        add_if_not_none('place_of_publication', place_text)

    # Nakladatel (Publisher)
    publisher_elem = root.find(".//mods:originInfo/mods:publisher", ns)
    if publisher_elem is not None and publisher_elem.text:
        publisher_text = fix_text(publisher_elem.text)
        publisher_text = publisher_text.replace("nakladatel,", "").replace("nákladem", "").strip()
        add_if_not_none('publisher', publisher_text)

    # Podtitul
    subtitle = root.xpath(".//mods:titleInfo/mods:subTitle", namespaces=ns)
    add_if_not_none('subtitle', fix_text(subtitle[0].text) if subtitle else None)

    # Vydání (Edition)
    edition_elem = root.find(".//mods:originInfo/mods:edition", ns)
    if edition_elem is not None and edition_elem.text:
        edition_text = fix_text(edition_elem.text)
        edition_text = edition_text.replace("vydání", "").replace("vyd.", "").strip()
        add_if_not_none('edition', edition_text)

    # Díl (Part Number)
    part_number = root.xpath(".//mods:titleInfo/mods:partNumber", namespaces=ns)
    for pn in part_number:
        add_if_not_none('part_number', fix_text(pn.text) if pn.text else None)

    # Název dílu (Part Name)
    part_name = root.xpath(".//mods:titleInfo/mods:partName", namespaces=ns)
    for pn in part_name:
        if pn.text:
            part_text = fix_text(pn.text)
            part_text = part_text.replace("díl", "").strip()
            add_if_not_none('part_name', part_text)

    # Tiskaři
    printers = []
    for printer in root.xpath(".//mods:name[@type='personal']", namespaces=ns):
        role_terms = printer.xpath("mods:role/mods:roleTerm", namespaces=ns)
        if any(role.text == 'prt' for role in role_terms):
            name_parts = printer.xpath("mods:namePart", namespaces=ns)
            name = " ".join(fix_text(part.text) for part in name_parts if part.text)
            name = remove_numbers(name)
            if name:
                printers.append(name)
    add_if_not_none('printers', printers if printers else None)

    # Tiskárna (Printer)
    printer_elements = root.xpath(".//mods:originInfo[@eventType='manufacture']/mods:publisher", namespaces=ns)
    printers = []
    for printer_elem in printer_elements:
        printer_text = fix_text(printer_elem.text) if printer_elem.text else None
        if printer_text:
            printer_text = printer_text.replace("tiskem, ", "").replace("vytiskla", "").strip()
            printers.append(printer_text)
    add_if_not_none('printers', printers if printers else None)

    # Místo tisku (Place of Print)
    place_elements = root.xpath(".//mods:originInfo[@eventType='manufacture']/mods:place/mods:placeTerm", namespaces=ns)
    places = []
    for place_elem in place_elements:
        place_text = fix_text(place_elem.text) if place_elem.text else None
        if place_text:
            place_text = place_text.replace("v ", "").strip()
            places.append(place_text)
    add_if_not_none('place_of_print', places if places else None)

    # Název série
    series_title = root.xpath(".//mods:relatedItem[@type='series']/mods:titleInfo/mods:title", namespaces=ns)
    add_if_not_none('series', fix_text(series_title[0].text) if series_title else None)

    # Číslo série (Series Part Number)
    series_part_number = root.xpath(".//mods:relatedItem[@type='series']/mods:titleInfo/mods:partNumber", namespaces=ns)
    for spn in series_part_number:
        add_if_not_none('series_part_number', fix_text(spn.text) if spn.text else None)

    return data

def assign_category(text, data):
    '''Přiřadí kategorii textu na základě shody s extrahovanými informacemi'''
    categories = {
        'titulek': ['title'],
        'misto vydani': ['place_of_publication'],
        'nakladatel': ['publisher'],
        'podtitulek': ['subtitle'],
        'autor': ['authors', 'author'],
        'prekladatel': ['translators'],
        'ilustrator': ['illustrators'],
        'dil': ['part_number', 'part'],
        'nazev dilu': ['part_name'],
        'datum vydani': ['date_issued'],
        'editor': ['editors'],
        'tiskar': ['printers'],
        'misto tisku': ['place_of_print'],
        'serie': ['series'],
        'vydani': ['edition'],
        'cislo serie': ['series_part_number']
    }
    
    best_match = None
    highest_score = 0
    threshold=90
    
    for category, keys in categories.items():
        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    value = " ".join(value)

                # Vypočítej podobnost mezi textem a hodnotou
                score = fuzz.partial_ratio(text.lower(), value.lower())
                # Pokud skóre překročí práh a je nejlepší, ulož kategorii
                if score > threshold and score > highest_score:
                    highest_score = score
                    best_match = category

    return best_match if best_match else "unknown"

def check_logits_and_get_coords(pl, search_text, image_width, image_height):
    """
    Spustí check_logits a vrátí souřadnice pro text (pouze podřetězec), pokud byl nalezen.

    :param pl: Objekt PageLayout.
    :param search_text: Hledaný text.
    :return: Souřadnice (nebo None, pokud nebyl text nalezen).
    """

    search_text = search_text.lower()  # case-insensitive.

    for line in pl.lines_iterator():
        line_text = line.transcription.lower()
        if search_text in line_text:
            # Najít začátek a konec hledaného textu v rámci celého textu řádku
            start_idx = line_text.index(search_text)
            end_idx = start_idx + len(search_text)

            # Zarovnat znaky na obrázek
            x_char_alignment = align_text_to_image(line, 511)
            x_start = x_char_alignment[start_idx] - 25
            x_end = x_char_alignment[end_idx - 1]

            # Y souřadnice se vypočítají z baseline a výšek textu
            y_center = (line.baseline[0][1] + line.baseline[-1][1]) / 2
            y_top = y_center - line.heights[0]
            y_bottom = y_center + line.heights[1]

            # Výpočet výšky a šířky pro podřetězec
            width = (x_end - x_start) + 10
            height = y_bottom - y_top
            return {
                "x": (x_start / image_width) * 100,
                "y": (y_top / image_height) * 100,
                "width": (width / image_width) * 110,
                "height": (height / image_height) * 100,
            }
    return None

def get_line_coordinates(line, image_width, image_height):
    """
    Získá normalizované souřadnice z OCR řádku.

    :param line: OCR řádek.
    :param image_width: Šířka obrázku.
    :param image_height: Výška obrázku.
    :return: Slovník s normalizovanými souřadnicemi.
    """
    x_min = min(p[0] for p in line.polygon)
    y_min = min(p[1] for p in line.polygon)
    x_max = max(p[0] for p in line.polygon)
    y_max = max(p[1] for p in line.polygon)

    return {
        "x": (x_min / image_width) * 100,
        "y": (y_min / image_height) * 100,
        "width": ((x_max - x_min) / image_width) * 100,
        "height": ((y_max - y_min) / image_height) * 100,
    }

def handle_text(search_text, annotations, ocr_lines, pl, image_width, image_height, key):
    """
    Hledá shodu pro text a rozhoduje, zda přiřadit kategorii, nebo spustit check_logits.

    :param search_text: Text k vyhledání.
    :param annotations: Seznam anotací (JSON).
    :param ocr_lines: OCR výstup.
    :param pl: Objekt PageLayout.
    :param image_width: Šířka obrázku.
    :param image_height: Výška obrázku.
    :param key: Klíč z MODS (pro kategorii).
    """

    threshold = 90
    best_coords = None
    min_ocr_length = 3  # Minimální délka OCR textu
    single_box = ['title', 'subtitle', 'part', 'part_name', 'series'] # Texty
    completed_data = {}
    normalized_search_text = normalize_text(search_text)
    zarovnani = False
    global radek
    global nezarovnane_annotace

    if key in single_box and key not in completed_data:
        completed_data[key] = {"completed": False, "full_text": None}

    for line, ocr_text in ocr_lines:
        if search_text is not None:
            best_coords = None
            normalized_ocr_text = normalize_text(ocr_text)

            # text příliš krátký
            if len(normalized_ocr_text) < min_ocr_length:
                continue

            if key in single_box and completed_data[key]["completed"]:
                continue

            if (normalized_ocr_text == normalized_search_text):
                best_coords = get_line_coordinates(line, image_width, image_height)
                category = assign_category(search_text, {key: search_text})
                if key in single_box:
                    if completed_data[key]["full_text"]:
                        completed_data[key]["full_text"] += " " + normalized_ocr_text
                    else:
                        completed_data[key]["full_text"] = normalized_ocr_text

            else:
                score = fuzz.partial_ratio(normalized_search_text, normalized_ocr_text) # substringy v radku
                score_changed_words = fuzz.token_set_ratio(normalized_search_text, normalized_ocr_text) # prohozena slova

                if score < score_changed_words and score_changed_words > threshold:
                    words = normalized_search_text.split()
                    normalized_search_text = " ".join(reversed(words))

                if score > threshold or score_changed_words > threshold:
                    category = assign_category(normalized_search_text, {key: normalized_search_text})
                    if category == 'titulek' or category == 'podtitulek':
                        best_coords = get_line_coordinates(line, image_width, image_height)
                    else:
                        best_coords = check_logits_and_get_coords(pl, normalized_search_text, image_width, image_height)
                        if best_coords:
                            pass
                        else:
                            best_coords = get_line_coordinates(line, image_width, image_height)
                    if key in single_box:
                        if completed_data[key]["full_text"]:
                            completed_data[key]["full_text"] += " " + normalized_ocr_text
                        else:
                            completed_data[key]["full_text"] = normalized_ocr_text

        if key in single_box and key in completed_data:
            if (completed_data[key]["full_text"]) == normalized_search_text:
                completed_data[key]["completed"] = True
        
        if best_coords is not None:
            zarovnani = True
            annotations.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": best_coords["x"],
                    "y": best_coords["y"],
                    "width": best_coords["width"],
                    "height": best_coords["height"],
                    "rectanglelabels": [category]
                },
                "meta": {
                    "text": [search_text]
                }
            })

    if not zarovnani:
        # nepodařilo se zarovnat
        category = assign_category(search_text, {key: search_text})
        y_offset = radek * 5
        ann = ({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": 1,
                "y": y_offset,
                "width": 10,
                "height": 2,
                "rectanglelabels": [category]
            },
            "meta": {
                "text": [search_text]
            }
        })
        nezarovnane_annotace.append(ann)
        radek += 1

def process_and_save_json(input_json, output_path, xml_path):
    annotations = input_json["annotations"][0]["result"]
    
    merged_annotations = merge_annotations(annotations)
    filtered_annotations = check_count_of_boxes(merged_annotations)
    filtered_annotations.extend(nezarovnane_annotace)

    result = {
        "data": input_json["data"],
        "predictions": [
            {
                "result": filtered_annotations
            }
        ]
    }
    
    # Uložení JSON
    output_path = os.path.join(output_path, os.path.splitext(os.path.basename(xml_path))[0] + ".json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4) 

def merge_annotations(annotations, scale_factor=2):
    merged = []

    for item in annotations:
        label = item['value']['rectanglelabels'][0]
        x = item['value']['x']
        y = item['value']['y']
        width = item['value']['width']
        height = item['value']['height']
        meta_text = item.get('meta', {}).get('text', [])

        x2 = x + width
        y2 = y + height

        merged_this = False
        for existing in merged:
            if existing["rectanglelabels"][0] == label:
                reference_height = min(height, existing["height"])

                if label in {"titulek", "podtitulek", "nakladatel", "vydavatel", "tiskar", "nazev dilu"}:
                    max_distance = reference_height * scale_factor
                    vertical_distance = abs(existing["y2"] - y)
                    horizontal_overlap = max(0, min(existing["x2"], x2) - max(existing["x"], x))

                    if vertical_distance <= max_distance and horizontal_overlap > 0:
                        existing["x"] = min(existing["x"], x)
                        existing["y"] = min(existing["y"], y)
                        existing["x2"] = max(existing["x2"], x2)
                        existing["y2"] = max(existing["y2"], y2)
                        existing["height"] = max(existing["height"], height)
                        existing["meta_text"].extend(meta_text)
                        existing["meta_text"] = list(set(existing["meta_text"]))
                        merged_this = True
                        break

                else:
                    horizontal_overlap = max(0, min(existing["x2"], x2) - max(existing["x"], x))
                    vertical_overlap = max(0, min(existing["y2"], y2) - max(existing["y"], y))

                    total_vertical_height = max(existing["y2"], y2) - min(existing["y"], y)
                    vertical_overlap_ratio = vertical_overlap / total_vertical_height

                    if horizontal_overlap > 0 and vertical_overlap_ratio >= 0.8:
                        existing["x"] = min(existing["x"], x)
                        existing["y"] = min(existing["y"], y)
                        existing["x2"] = max(existing["x2"], x2)
                        existing["y2"] = max(existing["y2"], y2)
                        existing["height"] = max(existing["height"], height)
                        existing["meta_text"].extend(meta_text)
                        existing["meta_text"] = list(set(existing["meta_text"]))
                        merged_this = True
                        break

        if not merged_this:
            merged.append({
                "x": x,
                "y": y,
                "x2": x2,
                "y2": y2,
                "height": height,
                "rectanglelabels": [label],
                "meta_text": meta_text[:]
            })

    merged_annotations = []
    for box in merged:
        merged_annotations.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": box["x"],
                "y": box["y"],
                "width": box["x2"] - box["x"],
                "height": box["y2"] - box["y"],
                "rectanglelabels": box["rectanglelabels"]
            },
            "meta": {
                "text": box["meta_text"]
            }
        })

    return merged_annotations

def check_count_of_boxes(annotations):
    """
    Filtruje anotace tak, aby pro určité kategorie v `single_box` zůstala pouze jedna anotace.
    Pro některé kategorie vybírá boxy níže, pro ostatní výše na stránce.

    :param annotations: List anotací (seznam slovníků obsahujících informace o anotacích).
    :return: List filtrovaných anotací.
    """
    lower_priority = ['nakladatel', 'misto vydani', 'datum vydani']

    upper_priority = ['titulek', 'podtitulek', 'vydani', 'dil', 'nazev dilu', 'serie', 'cislo serie']

    multiple_boxes = ['editor', 'prekladatel', 'ilustrator', 'autor', 'tiskar']

    filtered_single_boxes = {}

    for annotation in annotations:
        category = annotation["value"]["rectanglelabels"][0]
        y = annotation["value"]["y"]

        if category in lower_priority:
            # spodní (nižší) box
            if category in filtered_single_boxes:
                if y > filtered_single_boxes[category]["value"]["y"]:  # Vyber box níže
                    filtered_single_boxes[category] = annotation
            else:
                filtered_single_boxes[category] = annotation

        elif category in upper_priority:
            # horní (vyšší) box
            if category in filtered_single_boxes:
                if y < filtered_single_boxes[category]["value"]["y"]:  # Vyber box výše
                    filtered_single_boxes[category] = annotation
            else:
                filtered_single_boxes[category] = annotation

    filtered_annotations = list(filtered_single_boxes.values())
    filtered_annotations += [
        annotation for annotation in annotations
        if annotation["value"]["rectanglelabels"][0] in multiple_boxes
    ]

    return filtered_annotations

def parse_page_xml_to_json(xml_path, mods_path, output_dir, logits):
    '''Převede PAGE XML soubor na JSON pro Label Studio a přidá anotace na základě MODS dat'''
    tree = etree.parse(xml_path)
    page = tree.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Page")
    image_filename = page.attrib["imageFilename"]
    image_width = int(page.attrib["imageWidth"])
    image_height = int(page.attrib["imageHeight"])

    # Načíst data z MODS
    data = parse_mods(mods_path)
    annotations = []
    nezarovnane_annotace.clear()
    global radek
    radek = 1

    # Uložení všech OCR řádků
    pl = layout.PageLayout()
    pl.from_pagexml(xml_path)
    pl.load_logits(logits)
    ocr_lines = [(line, line.transcription) for line in pl.lines_iterator()]

    # Pro každý MODS údaj hledáme shody v OCR
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                handle_text(item, annotations, ocr_lines, pl, image_width, image_height, key)
        else:
            handle_text(value, annotations, ocr_lines, pl, image_width, image_height, key)

    # výstupní JSON
    result = {
        "data": {
            "image": "/data/local-files/?d=digilinka_knihy/images/" + image_filename + '.jpg'
        },
        "annotations": [
            {
                "result": annotations  # Anotace (kategorie != "unknown")
            }
        ],
        "predictions": [ ]
    }

    # Zpracování a uložení
    process_and_save_json(result, output_dir, xml_path)

def main():
    parser = argparse.ArgumentParser(description="Convert PAGE XML files to Label Studio JSON format.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the directory containing PAGE XML files.")
    parser.add_argument("-m", "--mods", type=str, required=True, help="Path to the directory containing MODS files.")
    parser.add_argument("-l", "--logits", type=str, required=True, help="Path to the directory containing LOGITS files.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output directory for JSON files.")
    args = parser.parse_args()

    xml_dir = args.input
    mods_dir = args.mods
    logits_dir = args.logits
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_basename = os.path.splitext(xml_file)[0]
            mods_file = os.path.join(mods_dir, f"{xml_basename}.mods")
            logits_file = os.path.join(logits_dir, f"{xml_basename}.logits")

            if not os.path.isfile(mods_file):
                print(f"MODS file for {xml_file} not found, skipping.")
                continue
            if not os.path.isfile(logits_file):
                print(f"LOGITS file for {xml_file} not found, skipping.")
                continue
            
            parse_page_xml_to_json(
                os.path.join(xml_dir, xml_file),
                mods_file,
                output_dir,
                logits_file
            )

if __name__ == "__main__":
    main()