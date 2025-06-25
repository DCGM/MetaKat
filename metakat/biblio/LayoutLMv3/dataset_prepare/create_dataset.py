# Skript pro přípravu formátu datasetu pro trénink LayoutLMv3 modelu část první
# Autor: Marie Pařilová
# Datum: 23. dubna 2025

import os
import json
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon
from PIL import Image
from lxml import etree

def calculate_iou(box_1, box_2):
    """Spočítá IoU jako poměr průniku k menší ploše."""
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area, poly_2.area)
    return iou / min_area

def parse_alto_file(alto_path):
    """Zpracuje ALTO XML soubor a vrátí seznam slov, jejich bboxů a confidence skóre."""
    tree = etree.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': root.tag.split('}')[0].strip('{')}
    strings = root.findall(".//alto:String", namespaces=ns)
    words = []
    bboxes = []
    confidences = []

    for string in strings:
        text = string.attrib.get("CONTENT")
        if text:
            words.append(text)
            hpos = int(string.get('HPOS'))
            vpos = int(string.get('VPOS'))
            width = int(string.get('WIDTH'))
            height = int(string.get('HEIGHT'))
            bboxes.append([hpos, vpos, hpos + width, vpos + height])
            conf = float(string.get('WC', 0.5))
            confidences.append(conf)

    return words, bboxes, confidences

def parse_yolo_file(yolo_path, img_width, img_height):
    """Načte YOLO soubor a převede bboxy na absolutní souřadnice."""
    bboxes = []
    labels = []
    with open(yolo_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            label = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            bboxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            labels.append(label)
    return bboxes, labels

def process_dataset(image_folder, label_folder, alto_folder):
    """Zpracuje obrázky, ALTO soubory a YOLO anotace, a vrátí dataset ve formátu LayoutLMv3."""
    dataset = []
    file_list = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for filename in tqdm(file_list, desc=f"Processing {image_folder}"):
        print(f"\n🔍 Zpracovávám soubor: {filename}")  

        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_folder, filename)
        yolo_path = os.path.join(label_folder, base_name + '.txt')
        alto_file = os.path.join(alto_folder, base_name + '.xml')

        if not os.path.exists(alto_file):
            print(f"⚠️ ALTO soubor nenalezen pro {base_name}, přeskočeno.")
            continue

        img = Image.open(image_path)
        img_width, img_height = img.size

        yolo_bboxes, yolo_labels = parse_yolo_file(yolo_path, img_width, img_height) if os.path.exists(yolo_path) else ([], [])
        alto_words, alto_bboxes, alto_conf = parse_alto_file(alto_file)

        entry = {
            "id": len(dataset) + 1,
            "file_name": filename,
            "tokens": [],
            "bboxes": [],
            "ner_tags": [],
            "yolo": [],
            "yolo_labels": []
        }

        for i in range(len(alto_words)):
            word = alto_words[i]
            word_bbox = alto_bboxes[i]
            assigned_labels = set()
            assigned_yolo_boxes = []

            for j, yolo_bbox in enumerate(yolo_bboxes):
                iou = calculate_iou(yolo_bbox, [[word_bbox[0], word_bbox[1]], [word_bbox[2], word_bbox[1]],
                                                [word_bbox[2], word_bbox[3]], [word_bbox[0], word_bbox[3]]])
                if iou >= 0.9:
                    assigned_labels.add(yolo_labels[j])
                    assigned_yolo_boxes.append(yolo_bbox)

            if not assigned_labels:
                assigned_labels.add(16)  # Neoznačený token
                assigned_yolo_boxes.append((0, 0, 0, 0))

            for label, yolo in zip(assigned_labels, assigned_yolo_boxes):
                entry["tokens"].append(word)
                entry["bboxes"].append(word_bbox)
                entry["ner_tags"].append(label)
                entry['yolo'].append(yolo)

        entry['yolo_labels'].append(yolo_labels)
        dataset.append(entry)

    return dataset

def main():
    train_dataset = process_dataset('biblio_dataset_final/images/train',
                                    'biblio_dataset_final/yolo/train',
                                    'biblio_dataset_final/alto.raw')

    val_dataset = process_dataset('biblio_dataset_final/images/eval',
                                  'biblio_dataset_final/yolo/val',
                                  'biblio_dataset_final/alto.raw')

    with open('train.json', 'w', encoding='utf-8') as f:
        for entry in train_dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    with open('test.json', 'w', encoding='utf-8') as f:
        for entry in val_dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print("✅ Dataset uložen.")

if __name__ == "__main__":
    main()
