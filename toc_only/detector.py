import cv2
import numpy as np
from ultralytics import YOLO
from .data_types import PageItem

YOLO_MAP = {
    "kapitola": "chapter_L1",
    "jiny nadpis": "chapter_L2",
    "nadpis v textu": "info_block",
    "podnadpis": "info_block",
    "cislo strany": "page_number",
    "jine cislo": "chapter_number"
}

COLOR_MAP = {
    "chapter_L1": (255, 0, 0),
    "chapter_L2": (0, 255, 0),
    "page_number": (0, 0, 255),
    "info_block": (200, 50, 50),
    "chapter_number": (0, 165, 255)
}


class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        print(f"Loading YOLO ...")
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.names = self.model.names

    def detect(self, image, output_path=None) -> list[PageItem]:
        results = self.model.predict(image, conf=self.conf, verbose=False)
        items = []

        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_name = self.names[int(box.cls[0])]
            unified_cat = YOLO_MAP.get(cls_name, "info_block")

            items.append(PageItem(
                bbox=coords,
                category=unified_cat,
                conf=float(box.conf[0])
            ))

        items = self.filter_boxes(items)
        if output_path:
            self.save_visualization(image, items, output_path)
        return items

    def filter_boxes(self, items, iou_thresh=0.5):
        if not items:
            return []
        items.sort(key=lambda x: x.conf, reverse=True)
        keep = []
        for item in items:
            should_keep = True
            for frame in keep:
                # Left up corner
                xA, yA = max(item.bbox[0], frame.bbox[0]), max(
                    item.bbox[1], frame.bbox[1])
                # Right down corner
                xB, yB = min(item.bbox[2], frame.bbox[2]), min(
                    item.bbox[3], frame.bbox[3])

                # Calculate the intersection area
                inter_Area = max(0, xB - xA) * max(0, yB - yA)

                actual_item_Area = (item.bbox[2]-item.bbox[0]) * \
                    (item.bbox[3]-item.bbox[1])
                frame_Area = (frame.bbox[2]-frame.bbox[0]) * \
                    (frame.bbox[3]-frame.bbox[1])

                # Sum of 2 areas
                union = actual_item_Area + frame_Area - inter_Area

                coef = inter_Area / union if union > 0 else 0
                if coef > iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                keep.append(item)
        return keep

    def save_visualization(self, image, items, output_path):
        canvas = image.copy()
        for item in items:
            x1, y1, x2, y2 = item.bbox
            color = COLOR_MAP.get(item.category, (128, 128, 128))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.imwrite(output_path, canvas)
