import os
import cv2
from pero_ocr.document_ocr.page_parser import PageParser
from pero_ocr.core.layout import PageLayout
import configparser

from toc_only.detector import YoloDetector
from toc_only.extractors import PeroAltoExtractor, PeroOCRExtractor
from toc_only.structure import HierarchyBuilder
from toc_only.data_types import PageItem, BookData
from toc_only.llm_extractor import LlmGPTExtractor, LlmGeminiExtractor
from refine_llm_bboxes import refine_bboxes


# OCR on each ToC page for LLM
def generate_alto_for_toc(images_folder: str,
                          book: BookData,
                          ocr_config_path: str,
                          output_dir: str) -> list:

    # Config OCR
    config = configparser.ConfigParser()
    config.read(ocr_config_path)
    parser = PageParser(config, config_path=os.path.dirname(ocr_config_path))

    alto_paths = []

    # Every ToC page
    for toc_page in book.toc_pages:
        img_path = os.path.join(images_folder, toc_page["filename"])
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Cannot read ToC image: {img_path}")
            continue

        file_id = os.path.splitext(toc_page["filename"])[0]
        xml_path = os.path.join(output_dir, f"{file_id}_alto.xml")

        # skip if already generated
        if os.path.exists(xml_path):
            alto_paths.append(xml_path)
            continue

        layout = PageLayout(id=file_id, page_size=image.shape[:2])
        try:
            parser.process_page(image, layout)
        except Exception as e:
            print(f"[WARNING] ALTO generation failed for {file_id}: {e}")
            continue

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(layout.to_altoxml_string())

        # Save XML
        alto_paths.append(xml_path)

    return alto_paths   # list of paths


# ToC working + PERO working on pages
def extract_text_from_pdf(
    images_folder: str,
    output_dir: str,
    book: BookData,
    ocr_config_path: str,
    yolo_model_path: str,
    toc_method: str = "pero",
    api_key: str = None
):

    os.makedirs(output_dir, exist_ok=True)

    if toc_method == "pero":
        print(f"Loading YOLO for ToC ...")
        yolo = YoloDetector(yolo_model_path)

        print(f"Loading PERO ALTO for ToC ...")
        alto_extractor = PeroAltoExtractor(ocr_config_path)
        builder = HierarchyBuilder()

    elif toc_method == "gpt":
        llm_extractor = LlmGPTExtractor(api_key=api_key)
    elif toc_method == "gemini":
        llm_extractor = LlmGeminiExtractor(api_key=api_key)
    else:
        print("[ERROR]: Unknown method!")
        return

    print(f"Loading PERO OCR for chapters ...")
    ocr_extractor = PeroOCRExtractor(ocr_config_path)

    print(f"\n--- ToC working (Method: {toc_method.upper()}) ---")

    # PERO
    if toc_method == "pero":
        for toc_page in book.toc_pages:
            img_path = os.path.join(images_folder, toc_page["filename"])
            image = cv2.imread(img_path)
            file_id = os.path.splitext(toc_page["filename"])[0]

            print(f"Reading ToC: {file_id} ...")

            items = yolo.detect(image)

            # PERO-ALTO (getting info from YOLO boxes)
            items_with_text = alto_extractor.extract(
                image, items, file_id=file_id, output_dir=output_dir
            )

            # Tree
            page_tree = builder.build(items_with_text, page_id=file_id)
            book.theoretical_toc.extend(page_tree)

    # LLM
    elif toc_method in ["gpt", "gemini"]:
        llm_images = []
        llm_file_ids = []

        # Collecting all ToC images
        for toc_page in book.toc_pages:
            img_path = os.path.join(images_folder, toc_page["filename"])
            image = cv2.imread(img_path)
            file_id = os.path.splitext(toc_page["filename"])[0]

            llm_images.append(image)
            llm_file_ids.append(file_id)

        if llm_images:
            # get chapter structure from LLM
            print(f"[LLM] Sending {len(llm_images)} page(s) to LLM ...")
            chapters = llm_extractor.extract_multiple(llm_images, llm_file_ids)

            # generate ALTO XML for ToC pages
            print(f"\n[LLM] Generating ALTO XML for bbox ...")
            alto_paths = generate_alto_for_toc(
                images_folder=images_folder,
                book=book,
                ocr_config_path=ocr_config_path,
                output_dir=output_dir,
            )

            # calculate bboxes using Levenshtein matching
            if alto_paths:
                chapters = refine_bboxes(
                    llm_chapters=chapters,
                    alto_xml_paths=alto_paths,
                )
            else:
                print("[WARNING] No ALTO files for ToC bbox\n")

            book.theoretical_toc.extend(chapters)

    print("\n--- Chapters working ---")
    for chapter_page in book.chapter_pages:
        img_path = os.path.join(images_folder, chapter_page["filename"])
        image = cv2.imread(img_path)
        file_id = os.path.splitext(chapter_page["filename"])[0]
        physical_page_num = chapter_page["page_num"]

        yolo_items = []

        # Chapters
        for title_info in chapter_page.get("titles_to_crop", []):
            yolo_items.append(PageItem(
                bbox=title_info["coords"],
                category="chapter_L1",
                conf=title_info["conf"]
            ))

        # Number of page
        for number_info in chapter_page.get("page_numbers_to_crop", []):
            yolo_items.append(PageItem(
                bbox=number_info["coords"],
                category="page_number",
                conf=number_info["conf"]
            ))

        # PERO
        try:
            extracted_items = ocr_extractor.extract(
                image, yolo_items, file_id=file_id, output_dir=output_dir
            )
        except Exception as e:
            print(f"[WARNING] OCR failed on {file_id}: {e}")
            continue

        # Results
        extracted_titles = []
        extracted_numbers = []

        for item in extracted_items:
            if item.category == "chapter_L1" and item.text:
                extracted_titles.append(item.text)
            elif item.category == "page_number" and item.text:
                extracted_numbers.append(item.text)

        found_page_number = extracted_numbers[0] if extracted_numbers else None

        # Saving to the memory
        for title in extracted_titles:
            book.actual_chapters.append({
                "physical_page":        physical_page_num,
                "extracted_text":       title,
                "extracted_page_number": found_page_number
            })

    print("-" * 40)
    print("[INFO] Collecting info from the book - Done!")
