"""
File: create_descriptions_from_ocr.py
Author: [Matej Smida]
Date: 2025-05-12
Description: Creates text prompts for CLIP model with Classification Head
             for [for training purposes].
"""

import argparse
import xml.etree.ElementTree as ET

from rapidfuzz import fuzz
import os, logging, time, sys

logger = logging.getLogger(__name__)

#manually collected keywords from images in dataset
keywords_by_class = {
    "Abstract": ["Abstract", "Abstrakt"],
    "Appendix": ["Dodatek", "Dodatky", "Apppendix", "Appendices", "Doplňky"],
    "BackCover": ["ISBN"],
    "Bibliography": ["prameny", "bibliografie", "bibliography", "literatury", "literatura"],
    "Dedication": ["dedication", "dedicatio", "dedicatoria", "dedicated to"],
    "Errata": ["opravy", "opravy chyb", "corrections", "errata"],
    "Imprimatur": ["Imprimatur"],
    "Index": ["index", "rejstřík"],
    "ListOfIllustrations": ["List of illustrations", "seznam vyobrazení", "seznam děl", "seznam ilustrací", "ilustrace", "seznam reprodukcí", "seznam obrázků", "seznam fotografií", "seznam obrazů"],
    "ListOfMaps": ["seznam map", "mapová příloha", "přehled map", "přehled mapek", "list of maps", "mapy"],
    "ListOfTables": ["seznam tabulek", "tabulky", "tables", "list of tables", "tabulková příloha", "přehled tabulek", "seznam tabulí"],
    "Map": ["mapa", "mapa města"],
    "Preface": ["preface", "předmluva"],
    "Table": ["tabulka", "tab.", "table"],
    "TableOfContents": ["Obsah", "contents", "table of contents"]
}

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ocr-xml-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--control-pages-file", default=None)
    parser.add_argument("--threshold", default=90)
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    return parser.parse_args()

def setup_logger(out_file, log_level = logging.INFO):
    log_formatter = logging.Formatter("'%(asctime)s - CREATE TEXT INPUT FROM OCR - %(levelname)s - %(message)s'")
    log_formatter.converter = time.gmtime

    stream = logging.StreamHandler()
    stream.setFormatter(log_formatter)

    file = logging.FileHandler(os.path.join(out_file, "create_desciptions_from_ocr.log"))
    file.setFormatter(log_formatter)

    my_logger = logging.getLogger()
    my_logger.handlers = [stream, file]
    my_logger.setLevel(log_level)

    return my_logger

def extract_text_with_coords(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespace = {'ns': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}

    page = root.find(".//ns:Page", namespace)
    image_height = int(page.attrib['imageHeight'])

    text_regions = page.findall(".//ns:TextRegion", namespace)

    texts_with_positions = []

    for region in text_regions:
        lines = region.findall(".//ns:TextLine", namespace)

        for line in lines:
            text_elem = line.find(".//ns:Unicode", namespace)

            if text_elem is not None and text_elem.text is not None:
                text = text_elem.text.strip()

                base_line = line.find(".//ns:Baseline", namespace)

                if base_line is not None:
                    points = base_line.attrib['points']
                    y_coords = [int(pt.split(',')[1]) for pt in points.split(" ")]
                    y_avg = sum(y_coords) / len(y_coords)

                    texts_with_positions.append((text, y_avg))

    return texts_with_positions, image_height

def get_possible_type(xml_path, threshold):
    lines_with_positions, image_height = extract_text_with_coords(xml_path)

    best_class = None
    best_score = 0

    for class_name, keywords in keywords_by_class.items():
        for keyword in keywords:
            for text, y_avg in lines_with_positions:
                kw_words = len(keyword.split())
                text_words = len(text.split())

                # skips if sentence is shorted than keyword
                if kw_words >= 2 and text_words < kw_words:
                    continue

                score = fuzz.ratio(text.lower(), keyword.lower())

                word_count = len(text.split())
                #punishes long sentences
                if word_count <= 3:
                    score *= 1.1
                elif word_count >= 8:
                    score *= 0.8

                #punishes images in lower 3/4 of page
                if y_avg < image_height * 0.25:
                    score *= 1.1
                else:
                    score *= 0.8

                if score > best_score:
                    best_score = score
                    best_class = class_name

    if best_score >= threshold:
        return best_class
    else:
        return None


def main():
    args = arg_parse()

    os.makedirs(args.output_file, exist_ok=True)

    my_logger = setup_logger(args.output_file, args.log_level)
    my_logger.info(" ".join(sys.argv))

    out_descriptions = os.path.join(args.output_file, "page_descriptions")
    os.makedirs(out_descriptions, exist_ok=True)

    xmls = args.ocr_xml_file

    correct = 0
    wrong = 0
    none = 0

    for i, xml in enumerate(os.listdir(xmls)):
        if ".jp" in xml:
            img_id = xml.strip().split(".jp")[0]
            img_path = os.path.basename(img_id)
        else:
            img_id = xml.strip().split(".xml")[0]
            img_path = os.path.abspath(img_id)

        xml_path = os.path.join(xmls, xml)

        possible_type = get_possible_type(xml_path, args.threshold)

        out_file = os.path.join(out_descriptions, img_id + ".txt")
        with open(out_file, "w") as f:
            if possible_type is not None:
                f.write(f"Possibly a page of type {possible_type}.")
            else:
                f.write("A page of unknown type.")

        if args.control_pages_file is not None:
            if possible_type is not None:
                for page in open(args.control_pages_file, "r"):
                    p_id = page.strip().split(".jp")[0]
                    p_type = page.strip().split(" ")[1]

                    if img_id == p_id:
                        if possible_type == p_type:
                            correct += 1
                        else:
                            wrong += 1
                        break
            else:
                none += 1


        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(os.listdir(xmls))} images")

    logger.info("\n")
    logger.info(f"Correct guessed types: {correct}")
    logger.info(f"Wrong guessed types: {wrong}")
    logger.info(f"Not guessed {none}")

    succcess_rate = correct / (correct + wrong) * 100
    logger.info(f"Success rate: {succcess_rate:.2f}%")

if __name__ == "__main__":
    main()