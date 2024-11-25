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

sys.path.append('/home/maja/Plocha/BP-git/MetaKat/pero-ocr/pero_ocr/core')
from force_alignment import align_text_to_image
from pero_ocr.core import layout

# Funkce pro opravu textu
def fix_text(text):
    return ftfy.fix_text(text) if text else text

# Funkce pro normalizaci textu
def normalize_text(text):
    # Odstranit typografické uvozovky, speciální znaky a přebytečné mezery
    text = re.sub(r'[„“"\'.,;:!?(){}[\]]', '', text)  # Odstraní typografické uvozovky a interpunkci
    # Normalizace unicode pro odstranění diakritiky (háčky, čárky)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text.strip().lower()  # Uprav text na malá písmena a ořeže přebytečné mezery

# Funkce pro načítání údajů z MODS
def parse_mods(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ET.parse(file)
    root = tree.getroot()

    ns = {'mods': 'http://www.loc.gov/mods/v3'}
    data = {}

    def add_if_not_none(key, value):
        """Přidá klíč-hodnotu do slovníku pouze tehdy, pokud hodnota není None."""
        if value is not None and value != "N/A":
            data[key] = value

    # Název
    title = root.find(".//mods:titleInfo/mods:title", ns)
    add_if_not_none('title', fix_text(title.text) if title is not None else None)

    # Autoři
    authors = []
    for author in root.findall(".//mods:name[@type='personal']", ns):
        name = author.find("mods:namePart", ns)
        if name is not None:
            authors.append(fix_text(name.text))
    add_if_not_none('authors', authors if authors else None)

    # Překladatelé
    translators = []
    for translator in root.findall(".//mods:name[@role]/mods:role/mods:roleTerm[.='translator']", ns):
        name = translator.find("../mods:namePart", ns)
        if name is not None:
            translators.append(fix_text(name.text))
    add_if_not_none('translator', translators if translators else None)

    # Ilustrátoři
    illustrators = []
    for illustrator in root.findall(".//mods:name[@role]/mods:role/mods:roleTerm[.='illustrator']", ns):
        name = illustrator.find("../mods:namePart", ns)
        if name is not None:
            illustrators.append(fix_text(name.text))
    add_if_not_none('illustrator', illustrators if illustrators else None)

    # Datum vydání
    date_issued = root.find(".//mods:originInfo/mods:dateIssued", ns)
    add_if_not_none('date_issued', fix_text(date_issued.text) if date_issued is not None else None)

    # Místo vydání
    place = root.find(".//mods:originInfo/mods:place/mods:placeTerm[@type='text']", ns)
    add_if_not_none('place_of_publication', fix_text(place.text) if place is not None else None)

    # Nakladatel
    publisher = root.find(".//mods:originInfo/mods:publisher", ns)
    add_if_not_none('publisher', fix_text(publisher.text) if publisher is not None else None)

    # Rok vydání
    add_if_not_none('year_of_publication', 
        fix_text(date_issued.text.split("-")[0]) if date_issued is not None else None)

    # Podtitul
    subtitle = root.find(".//mods:titleInfo/mods:subTitle", ns)
    add_if_not_none('subtitle', fix_text(subtitle.text) if subtitle is not None else None)

    # Vydání
    edition = root.find(".//mods:originInfo/mods:edition", ns)
    add_if_not_none('edition', fix_text(edition.text) if edition is not None else None)

    # Díl
    volume = root.find(".//mods:part/mods:detail[@type='volume']/mods:number", ns)
    add_if_not_none('volume', fix_text(volume.text) if volume is not None else None)

    # Název dílu
    title_of_part = root.find(".//mods:relatedItem[@type='series']/mods:titleInfo/mods:title", ns)
    add_if_not_none('title_of_part', fix_text(title_of_part.text) if title_of_part is not None else None)

    # Editoři
    editors = []
    for editor in root.findall(".//mods:name[@role]/mods:role/mods:roleTerm[.='editor']", ns):
        name = editor.find("../mods:namePart", ns)
        if name is not None:
            editors.append(fix_text(name.text))
    add_if_not_none('editor', editors if editors else None)

    # Tiskaři
    printers = []
    for printer in root.findall(".//mods:name[@role]/mods:role/mods:roleTerm[.='printer']", ns):
        name = printer.find("../mods:namePart", ns)
        if name is not None:
            printers.append(fix_text(name.text))
    add_if_not_none('printer', printers if printers else None)

    # Místo tisku
    place_of_print = root.find(".//mods:originInfo/mods:place/mods:placeTerm[@type='print']", ns)
    add_if_not_none('place_of_print', fix_text(place_of_print.text) if place_of_print is not None else None)

    return data
    
# Funkce pro přiřazení kategorií textům
def assign_category(text, data):
    categories = {
        'titulek': ['title'],
        'misto vydani': ['place_of_publication'],
        'nakladatel': ['publisher'],
        'podtitulek': ['subtitle'],
        'vydavatel': ['publisher'],
        'autor': ['authors', 'author'],
        'prekladatel': ['translator'],
        'ilustrator': ['illustrator'],
        'vydani': ['edition'],
        'dil': ['volume', 'part'],
        'nazev dilu': ['title_of_part'],
        'datum vydani': ['date_issued', 'year_of_publication'],
        'editor': ['editor'],
        'tiskar': ['printer'],
        'misto tisku': ['place_of_print']
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
            x_start = x_char_alignment[start_idx] - 15
            x_end = x_char_alignment[end_idx - 1]

            # Y souřadnice se vypočítají z baseline a výšek textu
            y_center = (line.baseline[0][1] + line.baseline[-1][1]) / 2
            y_top = y_center - line.heights[0]
            y_bottom = y_center + line.heights[1]

            # Výpočet výšky a šířky pro podřetězec
            width = x_end - x_start + 15 # padding
            height = y_bottom - y_top

            return {
                "x": (x_start / image_width) * 100,
                "y": (y_top / image_height) * 100,
                "width": (width / image_width) * 100,
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
    for line, ocr_text in ocr_lines:
        normalized_ocr_text = normalize_text(ocr_text)
        normalized_search_text = normalize_text(search_text)

        if (normalized_ocr_text == normalized_search_text):
            #print(f"Shoda 100% pro '{normalized_search_text}' v OCR '{normalized_ocr_text}'")
            best_coords = get_line_coordinates(line, image_width, image_height)

            category = assign_category(search_text, {key: search_text})
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
                }
            })
        else:
            score = fuzz.partial_ratio(normalized_search_text, normalized_ocr_text) # substringy v radku
            score_changed_words = fuzz.token_sort_ratio(normalized_search_text, normalized_ocr_text) # prohozena slova
            if score_changed_words > threshold:
                words = normalized_search_text.split()
                normalized_search_text = " ".join(reversed(words))
            if score > threshold or score_changed_words > threshold:
                print(f"Nalezen podretezec pro '{normalized_search_text}' a '{normalized_ocr_text}'")
                category = assign_category(normalized_search_text, {key: normalized_ocr_text})

                if category == 'titulek' or category == 'autor':
                    best_coords = get_line_coordinates(line, image_width, image_height)
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
                        }
                    })
                else:
                    coords = check_logits_and_get_coords(pl, search_text, image_width, image_height)
                    if coords:
                        annotations.append({
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": coords["x"],
                                "y": coords["y"],
                                "width": coords["width"],
                                "height": coords["height"],
                                "rectanglelabels": [category]
                            }
                        })
        best_coords = None

# Funkce pro převod PAGE XML na JSON pro Label Studio
def parse_page_xml_to_json(xml_path, mods_path, output_dir, logits):
    tree = etree.parse(xml_path)
    page = tree.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Page")
    image_filename = page.attrib["imageFilename"]
    image_width = int(page.attrib["imageWidth"])
    image_height = int(page.attrib["imageHeight"])

    # Načíst data z MODS
    data = parse_mods(mods_path)
    annotations = []

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

    # Vytvořit výstupní JSON
    result = {
        "data": {
            "image": image_filename + '.jpg'
        },
        "annotations": [
            {
                "result": annotations  # Anotace (kategorie != "unknown")
            }
        ]
    }

    # Uložení JSON
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_path))[0] + ".json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

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
            print("PARSING FILE")
            parse_page_xml_to_json(
                os.path.join(xml_dir, xml_file),
                mods_file,
                output_dir,
                logits_file
            )

if __name__ == "__main__":
    main()