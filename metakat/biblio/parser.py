import os
import json
import argparse
import glob
from lxml import etree
import xml.etree.ElementTree as ET
import ftfy
import re
from rapidfuzz import fuzz

# Funkce pro opravu textu
def fix_text(text):
    return ftfy.fix_text(text) if text else text

# Funkce pro normalizaci textu
def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# Funkce pro načítání údajů z MODS
def parse_mods(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ET.parse(file)
    root = tree.getroot()

    ns = {'mods': 'http://www.loc.gov/mods/v3'}
    data = {}

    # Název
    title = root.find(".//mods:titleInfo/mods:title", ns)
    data['title'] = fix_text(title.text) if title is not None else "N/A"

    # Autoři
    authors = []
    for author in root.findall(".//mods:name[@type='personal']", ns):
        name = author.find("mods:namePart", ns)
        if name is not None:
            authors.append(fix_text(name.text))
    data['authors'] = authors

    # Datum vydání
    date_issued = root.find(".//mods:originInfo/mods:dateIssued", ns)
    data['date_issued'] = fix_text(date_issued.text) if date_issued is not None else "N/A"

    # Místo vydání
    place = root.find(".//mods:originInfo/mods:place/mods:placeTerm", ns)
    data['place_of_publication'] = fix_text(place.text) if place is not None else None

    # Nakladatel
    publisher = root.find(".//mods:originInfo/mods:publisher", ns)
    data['publisher'] = fix_text(publisher.text) if publisher is not None else None

    # Rok vydání
    date_issued = root.find(".//mods:originInfo/mods:dateIssued", ns)
    data['year_of_publication'] = fix_text(date_issued.text) if date_issued is not None else None

    # Jazyk
    language = root.find(".//mods:language/mods:languageTerm[@type='code']", ns)
    data['language'] = fix_text(language.text) if language is not None else None
    
    # Fyzický popis
    physical_description = []
    for extent in root.findall(".//mods:physicalDescription/mods:extent", ns):
        physical_description.append(fix_text(extent.text))
    data['physical_description'] = physical_description

    # Žánr
    genre = root.find(".//mods:genre", ns)
    data['genre'] = fix_text(genre.text) if genre is not None else None

    # Identifikátory
    identifiers = {}
    for identifier in root.findall(".//mods:identifier", ns):
        id_type = identifier.get('type')
        id_value = fix_text(identifier.text)
        identifiers[id_type] = id_value
    data['identifiers'] = identifiers

    return data


# Funkce pro přiřazení kategorií textům
def assign_category(text, data):
    categories = {
        'titulek': ['title'],
        'rocnik': ['ročník', 'number'],
        'misto vydani': ['place_of_publication'],
        'nakladatel': ['publisher'],
        'podtitulek': ['subtitle'],
        'vydavatel': ['publisher'],
        'autor': ['authors'],
        'prekladatel': ['translator'],
        'vydani': ['edition'],
        'dil': ['volume', 'part'],
        'nazev dilu': ['title of part'],
        'datum vydani': ['date_issued'],
        'serie': ['serie', 'series'],
        'editor': ['editor'],
        'tiskar': ['printer'],
        'misto tisku': ['place of print']
    }
    
    best_match = None
    highest_score = 0
    threshold=70
    
    for category, keys in categories.items():
        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):  # Pokud je hodnota seznam
                    value = " ".join(value)  # Spojí hodnoty seznamu do jednoho řetězce

                # Vypočítej podobnost mezi textem a hodnotou
                score = fuzz.partial_ratio(text.lower(), value.lower())
                
                # Pokud skóre překročí práh a je nejlepší, ulož kategorii
                if score > threshold and score > highest_score:
                    highest_score = score
                    best_match = category

    return best_match if best_match else "unknown"

# Funkce pro převod PAGE XML na JSON pro Label Studio
def parse_page_xml_to_json(xml_path, mods_path, output_dir):
    tree = etree.parse(xml_path)
    page = tree.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Page")
    image_filename = page.attrib["imageFilename"]
    image_width = int(page.attrib["imageWidth"])
    image_height = int(page.attrib["imageHeight"])

    data = parse_mods(mods_path)

    print(data)

    annotations = []

    for text_region in page.findall(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextRegion"):
        coords = text_region.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Coords").attrib["points"]
        points = [list(map(int, point.split(','))) for point in coords.split()]

        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)

        x = (x_min / image_width) * 100
        y = (y_min / image_height) * 100
        width = ((x_max - x_min) / image_width) * 100
        height = ((y_max - y_min) / image_height) * 100

        text_equiv = text_region.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Unicode")
        text = text_equiv.text if text_equiv is not None else ""

        category = assign_category(text, data)

        # pokud kategorie není 'unknown'
        if category != "unknown":
            annotations.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [category]
                }
            })

    # Výstupní JSON
    result = {
        "data": {
            "image": image_filename+'.jpg'
        },
        "annotations": [
            {
                "result": annotations  # kategorie != "unknown"
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
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output directory for JSON files.")
    args = parser.parse_args()

    xml_dir = args.input
    mods_dir = args.mods
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_basename = os.path.splitext(xml_file)[0]
            mods_file = os.path.join(mods_dir, f"{xml_basename}.mods")

            if not os.path.isfile(mods_file):
                print(f"MODS file for {xml_file} not found, skipping.")
                continue

            parse_page_xml_to_json(
                os.path.join(xml_dir, xml_file),
                mods_file,
                output_dir
            )

if __name__ == "__main__":
    main()
