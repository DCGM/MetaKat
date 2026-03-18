import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pero_ocr.document_ocr.page_parser import PageParser
from pero_ocr.core.layout import PageLayout, RegionLayout
import configparser


class PeroAltoExtractor:
    def __init__(self, config_path):
        self.parser = None
        if config_path and os.path.exists(config_path):
            try:
                self.PageLayout = PageLayout

                config = configparser.ConfigParser()
                config.read(config_path)

                # Parser inicialization
                self.parser = PageParser(
                    config, config_path=os.path.dirname(config_path))
                print("Loading PERO-ALTO ...")
            except Exception as e:
                print(f"[ERROR]: Failed to init PERO: {e}")
        else:
            print(f"[ERROR]: PERO config not found: {config_path}")

    def extract(self, image, layout_items, **kwargs):
        file_id = kwargs.get('file_id')
        output_dir = kwargs.get('output_dir')

        if not file_id or not output_dir or not self.parser:
            return layout_items

        # Creating Layout
        layout = PageLayout(id=file_id, page_size=image.shape[:2])

        # Working all page
        self.parser.process_page(image, layout)

        # Saving results
        self.save_results(image, layout, file_id, output_dir)

        # Adding YOLO boxes
        return self.merge(layout_items, layout)

    def save_results(self, image, layout, file_id, output_dir):
        if not output_dir:
            return

        img_path = os.path.join(output_dir, f"{file_id}_alto.jpg")
        xml_path = os.path.join(output_dir, f"{file_id}_alto.xml")

        try:
            # Saving image
            cv2.imwrite(img_path, layout.render_to_image(image))
            # Saving XML
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(layout.to_altoxml_string())
            print(f"ALTO image and XML saved")
        except Exception as e:
            print(f"[ERROR]: Could not save ALTO results: {e}")

    def merge(self, layout_items, layout):
        alto_words = self.parse_alto_xml(layout)
        if not alto_words:
            return layout_items

        for item in layout_items:
            bbox = item.bbox  # [x1, y1, x2, y2]

            # Serching for words inside bbox
            matched_words = []
            for word in alto_words:
                if self.is_inside(bbox, word):
                    matched_words.append(word)

            # Getting text
            full_text = " ".join([w['content'] for w in matched_words]).strip()
            item.text = full_text

        return layout_items

    def parse_alto_xml(self, layout):
        try:
            xml_string = layout.to_altoxml_string()
            root = ET.fromstring(xml_string)

            ns = {'alto': root.tag.split('}')[0].strip(
                '{')} if '}' in root.tag else {}

            words = []
            findall_query = ".//alto:String" if ns else ".//String"

            for string_elem in root.findall(findall_query, ns):
                if 'CONTENT' not in string_elem.attrib:
                    continue

                words.append({
                    'content': string_elem.attrib['CONTENT'],
                    'x': float(string_elem.attrib['HPOS']),
                    'y': float(string_elem.attrib['VPOS']),
                    'w': float(string_elem.attrib['WIDTH']),
                    'h': float(string_elem.attrib['HEIGHT'])
                })

            return words

        except Exception as e:
            print(f"[ERROR] Error parsing ALTO layout: {e}")
            return []

    def is_inside(self, yolo_bbox, word, iou_thresh=0.5):
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
        word_x1, word_y1 = word['x'], word['y']
        word_x2, word_y2 = word['x'] + word['w'], word['y'] + word['h']

        x1 = max(yolo_x1, word_x1)         # left
        y1 = max(yolo_y1, word_y1)         # up
        x2 = min(yolo_x2, word_x2)         # right
        y2 = min(yolo_y2, word_y2)         # down

        # No intersection
        if x2 <= x1 or y2 <= y1:
            return False

        inter_Area = (x2 - x1) * (y2 - y1)
        word_Area = word['w'] * word['h']

        return (inter_Area / word_Area) >= iou_thresh


class PeroOCRExtractor:
    def __init__(self, config_path):
        self.parser = None
        if config_path and os.path.exists(config_path):
            try:
                self.PageLayout = PageLayout
                self.RegionLayout = RegionLayout

                config = configparser.ConfigParser()
                config.read(config_path)

                # Setting model to not DETECT, we will give regions from YOLO
                if 'LAYOUT_PARSER_1' in config:
                    print("OCR setting changed")
                    config['LAYOUT_PARSER_1']['DETECT_REGIONS'] = 'no'

                # Parser inicialization
                self.parser = PageParser(
                    config, config_path=os.path.dirname(config_path))
            except Exception as e:
                print(f"[ERROR]: Failed to init PERO: {e}")
        else:
            print(f"[ERROR]: PERO config not found: {config_path}")

    def extract(self, image, layout_items, **kwargs):
        file_id = kwargs.get('file_id', 'temp')
        output_dir = kwargs.get('output_dir')

        if not file_id or not output_dir or not self.parser:
            return layout_items

        # Creating Layout from YOLO
        layout = self.PageLayout(id=file_id, page_size=image.shape[:2])
        item_map = {}

        for i, item in enumerate(layout_items):
            coords = item.bbox  # [x1, y1, x2, y2]

            polygon = np.array([
                [coords[0], coords[1]], [coords[2], coords[1]],
                [coords[2], coords[3]], [coords[0], coords[3]]
            ], dtype=int)

            # Creating regions
            reg_id = f"region_{i}_{item.category}"
            region = self.RegionLayout(id=reg_id, polygon=polygon)
            # Adding region to Layout
            layout.regions.append(region)

            # Creating pair
            item_map[reg_id] = item

        # PERO
        self.parser.process_page(image, layout)

        # Saving results
        self.save_results(image, layout, file_id, output_dir)

        # Getting text
        for region in layout.regions:
            lines_text = [
                l.transcription for l in region.lines if l.transcription is not None]
            text = " ".join(lines_text).strip()

            # Adding to item
            if region.id in item_map:
                item_map[region.id].text = text

        return layout_items

    def save_results(self, image, layout, file_id, output_dir):
        if not output_dir:
            return

        img_path = os.path.join(output_dir, f"{file_id}_pero.jpg")
        xml_path = os.path.join(output_dir, f"{file_id}_pero.xml")

        try:
            # Saving image
            cv2.imwrite(img_path, layout.render_to_image(image))
            # Saving XML
            layout.to_pagexml(xml_path)
            print(f"PERO image and XML saved")
        except Exception as e:
            print(f"[ERROR]: Could not save PERO results: {e}")
