import os
import cv2
import json
import argparse
from .detector import YoloDetector
from .extractors import PeroAltoExtractor, PeroOCRExtractor
from .structure import HierarchyBuilder
from .llm_extractor import LlmGPTExtractor, LlmGeminiExtractor

# --- Settings ---
YOLO_PATH = "runs_only_toc/bp_experiment/weights/best.pt"
PERO_CONFIG = "peromodel/config_cpu.ini"


def main():
    parser = argparse.ArgumentParser(description="ToC Processing Tool")
    parser.add_argument(
        "--mode", choices=["alto", "pero", "llm"], required=True, help="'alto' | 'pero' | 'llm'"
    )
    parser.add_argument(
        "--model", choices=["gpt", "gemini"], default="gemini", help="Choose LLM (only used if --mode llm)"
    )
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", default="final_results",
                        help="Output results folder")
    args = parser.parse_args()

    print(f"--- STARTING WORK: {args.mode.upper()} METHOD ---")

    # Inicialization
    if args.mode in ["alto", "pero"]:
        detector = YoloDetector(YOLO_PATH)
        builder = HierarchyBuilder()
        if args.mode == "alto":
            extractor = PeroAltoExtractor(PERO_CONFIG)
        else:
            extractor = PeroOCRExtractor(PERO_CONFIG)

    elif args.mode == "llm":
        print(f"--- USING LLM : {args.model.upper()} ---")
        if args.model == "gpt":
            api_key = "KEY"
            extractor = LlmGPTExtractor(api_key)
        elif args.model == "gemini":
            api_key = "KEY"
            extractor = LlmGeminiExtractor(api_key)

    # Images
    extensions = ('.jpg', '.png', '.jpeg')
    image_files = sorted([f for f in os.listdir(
        args.input) if f.lower().endswith(extensions)])

    for img_name in image_files:
        file_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(args.input, img_name)
        print(f"\nProcessing: {file_id}")

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Creating folders
        img_out_dir = os.path.join(args.output, file_id)

        if args.mode in ["alto", "pero"]:
            dir_yolo = os.path.join(img_out_dir, "yolo")
            dir_ocr = os.path.join(img_out_dir, "ocr")
            dir_interni = os.path.join(img_out_dir, "interni")
            dir_final = os.path.join(img_out_dir, "final")

            for d in [dir_yolo, dir_ocr, dir_interni, dir_final]:
                os.makedirs(d, exist_ok=True)

            # YOLO
            items = detector.detect(
                image, os.path.join(dir_yolo, f"{file_id}.jpg"))

            # OCR
            items = extractor.extract(
                image, items, file_id=file_id, output_dir=dir_ocr)

            # INTERNI FORMAT
            flat_list = [item.to_dict() for item in items]
            interni_path = os.path.join(dir_interni, f"{file_id}_interni.json")
            with open(interni_path, 'w', encoding='utf-8') as f:
                json.dump(flat_list, f, ensure_ascii=False, indent=4)

            # FINAL STRUCTURE
            hierarchy = builder.build(items, page_id=file_id)
            final_path = os.path.join(dir_final, f"{file_id}_structure.json")
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(hierarchy, f, ensure_ascii=False, indent=4)

        else:  # LLM
            dir_final = os.path.join(img_out_dir, "final")
            os.makedirs(dir_final, exist_ok=True)

            hierarchy = extractor.extract(image, file_id)

            final_path = os.path.join(
                dir_final, f"{file_id}_structure.json")
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(hierarchy, f, ensure_ascii=False, indent=4)

    print("\n--- FINISHED ---")


if __name__ == "__main__":
    main()
