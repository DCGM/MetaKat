import os
import argparse
import gc
import json
import shutil
import torch
import tempfile

from toc_only.data_types import BookData
from book_to_images import convert_pdf_to_images
from sort_images import run_yolo_sorting
from process_book import extract_text_from_pdf
from create_structure import calculate_final_structure
from interactive import make_pdf


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="PDF Processing Tool")
    parser.add_argument("input_path", type=str,
                        help="path to the book in pdf format or path to folder with images")
    parser.add_argument("--llm", type=str, choices=["gpt", "gemini"], default=None,
                        help="Choose the LLM model. If the --llm flag is not set -> PERO mode")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' is not found!")
        return

    is_pdf = os.path.isfile(input_path) and input_path.lower().endswith(".pdf")
    is_folder = os.path.isdir(input_path)

    if not is_pdf and not is_folder:
        print(f"Error: '{input_path}' is neither a PDF file nor a folder!")
        return

    toc_method = args.llm if args.llm else "pero"

    GPT_API_KEY = "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"
    GEMINI_API_KEY = "KEY"

    current_api_key = None
    if toc_method == "gpt":
        current_api_key = GPT_API_KEY
    elif toc_method == "gemini":
        current_api_key = GEMINI_API_KEY

    if is_pdf:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
    else:
        base_name = os.path.basename(os.path.normpath(input_path))

    output_interactive_pdf = f"{base_name}_interactive.pdf"
    output_json_path = f"{base_name}_structure.json"

    # Creating folders for results

    # images_folder = f"{base_name}_images"
    # output_dir = f"{base_name}_output"
    # os.makedirs(images_folder, exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)

    # Configs
    YOLO_SORTING_MODEL = 'runs/my_model/weights/best.pt'
    YOLO_TOC_MODEL = 'runs_only_toc/bp_experiment/weights/best.pt'
    PERO_CONFIG = 'peromodel/config_cpu.ini'

    print(f"\nStarting working with: {input_path}")

    book = BookData(input_path if is_pdf else None)

    with tempfile.TemporaryDirectory() as images_folder, tempfile.TemporaryDirectory() as output_dir:
        # 1. PDF to images
        if is_pdf:
            print(f"\n{'='*40}\nSTEP 1: PDF -> images\n{'='*40}")
            convert_pdf_to_images(book, images_folder)
        else:
            print(f"\n{'='*40}\nSTEP 1: Working with images\n{'='*40}")
            image_files = sorted(os.listdir(input_path))
            for file_name in image_files:
                src = os.path.join(input_path, file_name)
                dst = os.path.join(images_folder, file_name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            # 2. YOLO
        print(f"\n{'='*40}\nSTEP 2: SORT PDF\n{'='*40}")
        run_yolo_sorting(images_folder, book, YOLO_SORTING_MODEL, PERO_CONFIG)
        free_memory()

        # 3. Getting text
        print(f"\n{'='*40}\nSTEP 3: Getting text from images\n{'='*40}")
        extract_text_from_pdf(
            images_folder=images_folder,
            output_dir=output_dir,
            book=book,
            ocr_config_path=PERO_CONFIG,
            yolo_model_path=YOLO_TOC_MODEL,
            toc_method=toc_method,
            api_key=current_api_key
        )
        free_memory()

        # 4. FINAL
        print(f"\n{'='*40}\nSTEP 4: Final calculating of physical pages\n{'='*40}")
        calculate_final_structure(book)
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(book.final_structure, json_file,
                      ensure_ascii=False, indent=4)

            # 5. Generating PDF
        if is_pdf:
            print(f"\n{'='*40}\nSTEP 5: Creating interactive PDF\n{'='*40}")
            make_pdf(book, output_interactive_pdf)

    print(f"\n{'='*60}")
    if is_pdf:
        print(f"Final document: {output_interactive_pdf}")
    else:
        print(f"Final JSON: {output_json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
