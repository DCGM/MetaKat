import os
import cv2
from toc_only.detector import YoloDetector
from toc_only.extractors import PeroAltoExtractor, PeroOCRExtractor
from toc_only.structure import HierarchyBuilder
from toc_only.data_types import PageItem, BookData
from toc_only.llm_extractor import LlmGPTExtractor, LlmGeminiExtractor


def extract_text_from_pdf(
    images_folder: str,
    output_dir: str,
    book: BookData,
    ocr_config_path: str,
    yolo_model_path: str,
    toc_method: str = "pero",
    api_key: str = None
):
    """
    ToC working + PERO working on pages 
    """
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

            # YOLO
            # yolo_path = os.path.join(output_dir, f"{file_id}_yolo.jpg")
            # items = yolo.detect(image, output_path=yolo_path)
            items = yolo.detect(image)

            # PERO-ALTO
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

        # Sending all by one request
        if llm_images:
            chapters = llm_extractor.extract_multiple(llm_images, llm_file_ids)
            book.theoretical_toc.extend(chapters)

    print("\n--- Chapters working ---")
    for chapter_page in book.chapter_pages:
        img_path = os.path.join(images_folder, chapter_page["filename"])
        image = cv2.imread(img_path)
        file_id = os.path.splitext(chapter_page["filename"])[0]
        physical_page_num = chapter_page["page_num"]

        print(f"Reading: {file_id} ...")

        yolo_items = []

        # Looking for chapters in book and page numbers of their pages

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
        extracted_items = ocr_extractor.extract(
            image, yolo_items, file_id=file_id, output_dir=output_dir
        )

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
                "physical_page": physical_page_num,
                "extracted_text": title,
                "extracted_page_number": found_page_number
            })

    print("-" * 40)
    print("[INFO] Collecting info from the book - Done!")
