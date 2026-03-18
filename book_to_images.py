import os
from pdf2image import convert_from_path, pdfinfo_from_path
from toc_only.data_types import BookData

DPI = 450
FORMAT = "jpg"


def convert_pdf_to_images(book: BookData, output_folder: str) -> str:
    """
    PDF -> images
    """
    input_pdf_path = book.pdf_path

    if not os.path.exists(input_pdf_path):
        print(f"[ERROR]: File doesnt exist: '{input_pdf_path}'")
        return ""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        print(f"Reading PDF: {input_pdf_path} ...")
        info = pdfinfo_from_path(input_pdf_path)
        total_pages = info["Pages"]
        book.total_pages = total_pages

        print(f"Got {total_pages} pages. Saving in JPG ...")

        for page_num in range(1, total_pages + 1):
            filename = f"page_{page_num:03d}.{FORMAT}"
            save_path = os.path.join(output_folder, filename)

            if not os.path.exists(save_path):
                pages = convert_from_path(
                    input_pdf_path, dpi=DPI, fmt=FORMAT,
                    grayscale=True, first_page=page_num, last_page=page_num
                )
                if pages:
                    pages[0].save(save_path, optimize=True)

        print("-" * 40)
        print("[INFO] PDF to Images - Done!")
        return output_folder

    except Exception as e:
        print(f"[ERROR]: Converting PDF was ended with {e}\n")
        return ""
