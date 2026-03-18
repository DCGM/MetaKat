import os
import argparse
import time
import gc
import tempfile
import json
import torch


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
    parser.add_argument("pdf_path", type=str,
                        help="path to the book in pdf format")
    parser.add_argument("--llm", type=str, choices=["gpt", "gemini"], default=None,
                        help="Choose the LLM model. If the --llm flag is not set -> PERO mode")
    args = parser.parse_args()

    input_pdf = args.pdf_path
    if not os.path.exists(input_pdf):
        print(f"Error: File '{input_pdf}' is not found!")
        return

    toc_method = args.llm if args.llm else "pero"

    GPT_API_KEY = "OPENAI"
    GEMINI_API_KEY = "GEMINI"

    current_api_key = None
    if toc_method == "gpt":
        current_api_key = GPT_API_KEY
    elif toc_method == "gemini":
        current_api_key = GEMINI_API_KEY

    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    output_interactive_pdf = f"{base_name}_interactive.pdf"
    output_json_path = f"{base_name}_structure.json"

    YOLO_SORTING_MODEL = 'runs/my_model/weights/best.pt'
    YOLO_TOC_MODEL = 'runs_only_toc/bp_experiment/weights/best.pt'
    PERO_CONFIG = 'peromodel/config_cpu.ini'

    print(f"\nStarting working with: {input_pdf}")

    book = BookData(input_pdf)

    # Temporary folders
    with tempfile.TemporaryDirectory() as temp_images_folder, \
            tempfile.TemporaryDirectory() as temp_output_dir:

        # 1. PDF to images
        print(f"\n{'='*40}\nSTEP 1: PDF -> images\n{'='*40}")
        convert_pdf_to_images(book, temp_images_folder)

        # 2. YOLO
        print(f"\n{'='*40}\nSTEP 2: SORT PDF\n{'='*40}")
        run_yolo_sorting(temp_images_folder, book, YOLO_SORTING_MODEL)
        free_memory()

        # 3. Getting text
        print(f"\n{'='*40}\nSTEP 3: Getting text from images\n{'='*40}")
        extract_text_from_pdf(
            images_folder=temp_images_folder,
            output_dir=temp_output_dir,
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
        print(f"\n{'='*40}\nSTEP 5: Creating interactive PDF\n{'='*40}")
        make_pdf(book, output_interactive_pdf)

    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"Final document: {output_interactive_pdf}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
