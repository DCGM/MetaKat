import xml.etree.ElementTree as ET
import argparse
import ftfy
import os
import glob

def fix_text(text):
    return ftfy.fix_text(text) if text else text

def parse_mods(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ET.parse(file)
    root = tree.getroot()

    # namespace
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

    # ISBN
    isbn = root.find(".//mods:identifier[@type='isbn']", ns)
    data['isbn'] = fix_text(isbn.text) if isbn is not None else "N/A"

    # Datum vydání
    date_issued = root.find(".//mods:originInfo/mods:dateIssued", ns)
    data['date_issued'] = fix_text(date_issued.text) if date_issued is not None else "N/A"

    # Vydavatelé
    publishers = []
    for publisher in root.findall(".//mods:originInfo/mods:publisher", ns):
        publishers.append(fix_text(publisher.text))
    data['publishers'] = publishers

    # Místo vydání
    place = root.find(".//mods:originInfo/mods:place/mods:placeTerm[@type='text']", ns)
    data['place'] = fix_text(place.text) if place is not None else "N/A"

    return data

def process_folder(folder_path):
    mods_files = glob.glob(os.path.join(folder_path, "*.mods"))

    if not mods_files:
        print("Ve složce nejsou žádné soubory s příponou .mods")
        return

    for mods_file in mods_files:
        print(f"\nZpracovávám soubor: {mods_file}")
        metadata = parse_mods(mods_file)

        print("Název:", metadata['title'])
        print("Autoři:", ", ".join(metadata['authors']))
        print("ISBN:", metadata['isbn'])
        print("Datum vydání:", metadata['date_issued'])
        print("Vydavatelé:", ", ".join(metadata['publishers']))
        print("Místo vydání:", metadata['place'])

def process_file(file_path):
    metadata = parse_mods(file_path)

    print(f"\nZpracovávání souboru: {file_path}")
    print("Název:", metadata['title'])
    print("Autoři:", ", ".join(metadata['authors']))
    print("ISBN:", metadata['isbn'])
    print("Datum vydání:", metadata['date_issued'])
    print("Vydavatelé:", ", ".join(metadata['publishers']))
    print("Místo vydání:", metadata['place'])

def main():
    parser = argparse.ArgumentParser(description='Zpracování MODS souboru nebo složky s MODS soubory a extrakce bibliografických údajů.')
    parser.add_argument('path', help='Cesta k MODS souboru nebo složce s MODS soubory.')

    args = parser.parse_args()

    if os.path.isfile(args.path):
        # Pokud je to soubor
        process_file(args.path)
    elif os.path.isdir(args.path):
        # Pokud je to složka
        process_folder(args.path)
    else:
        print("Zadaná cesta není platný soubor ani složka.")

if __name__ == "__main__":
    main()
