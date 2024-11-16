import os
import json
import argparse
import glob
from lxml import etree
import xml.etree.ElementTree as ET
import ftfy
import re

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

    return data


# Funkce pro přiřazení kategorií textům
def assign_category(text):
    categories = {
        'titulek': ['uhlonošne útvary', 'tasmánii', 'název', 'headline', 'title'],
        'cislo': ['číslo', 'ii', 'iii', 'iv', 'issue'],
        'rocnik': ['ročník', 'number'],
        'datum cisla': ['datum', 'issue date', 'date issued'],
        'misto vydani': ['vydání', 'place of publication', 'place of print'],
        'nakladatel': ['nakladatelství', 'publisher'],
        'podtitulek': ['podtitulek', 'subtitle'],
        'redaktor': ['redaktor', 'editor', 'writing', 'compiler'],
        'vydavatel': ['vydavatel', 'publisher'],
        'datum rocniku': ['datum ročníku', 'year of edition'],
        'autor': ['autor', 'feistmantel', 'author', 'written by'],
        'prekladatel': ['překladatel', 'translator'],
        'ilustrator': ['ilustrátor', 'illustrator'],
        'vydani': ['vydání', 'edition'],
        'dil': ['díl', 'volume', 'part'],
        'nazev dilu': ['název dílu', 'title of part'],
        'cislo serie': ['číslo série', 'series number'],
        'datum vydani': ['datum vydání', 'publication date'],
        'serie': ['serie', 'series'],
        'editor': ['editor'],
        'tiskar': ['tiskař', 'printer'],
        'misto tisku': ['místo tisku', 'place of print']
    }
    
    # Text pro porovnání
    normalized_text = normalize_text(text)
    
    # Procházení kategorií
    for category, keywords in categories.items():
        if any(keyword in normalized_text for keyword in keywords):
            return category
    
    return 'unknown'

# Funkce pro převod PAGE XML na JSON pro Label Studio
def parse_page_xml_to_json(xml_path, mods_path, output_dir):
    tree = etree.parse(xml_path)
    page = tree.find(".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Page")
    image_filename = page.attrib["imageFilename"]
    image_width = int(page.attrib["imageWidth"])
    image_height = int(page.attrib["imageHeight"])

    data = parse_mods(mods_path)
    title = data.get('title', 'N/A')
    author = data.get('authors', ['N/A'])[0]  # Pokud více autorů, vezmeme první
    date_issued = data.get('date_issued', 'N/A')

    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Date Issued: {date_issued}")

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

        category = assign_category(text)

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
            "image": image_filename,
            "title": title,
            "author": author,
            "date_issued": date_issued
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
