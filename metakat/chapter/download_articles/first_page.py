from __future__ import annotations

import io

import pymupdf
from PIL import Image

from metakat.chapter.download_articles.models import FirstPageImage

# A single image covering at least this share of the page is treated as the page scan.
FULL_PAGE_COVERAGE = 0.9
# Pages without any scan (born-digital PDFs) have no resolution of their own.
DEFAULT_RENDER_DPI = 300

_EMBEDDED_EXTENSIONS = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "jpx": "jp2", "jp2": "jp2", "tiff": "tif"}


def extract_first_page(pdf_bytes: bytes, default_dpi: float = DEFAULT_RENDER_DPI) -> tuple[bytes, str, FirstPageImage]:
    """Return the first page of a PDF as image bytes, their file extension and how they were obtained.

    A scanned page is kept at the resolution it was scanned at: a single upright full-page image is
    stored byte for byte, any other scanned page is rendered at the highest resolution of its images.
    A page without images is rendered at ``default_dpi``.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count == 0:
            raise ValueError("PDF has no pages")
        page = document[0]
        page_area = page.rect.width * page.rect.height

        scans = []
        for image in page.get_images(full=True):
            xref, width = image[0], image[2]
            for bbox in page.get_image_rects(xref):
                if bbox.is_empty:
                    continue
                scans.append((xref, width, image[3], bbox))

        if len(scans) == 1 and page.rotation == 0:
            xref, width, height, bbox = scans[0]
            covers_page = bbox.width * bbox.height >= FULL_PAGE_COVERAGE * page_area
            upright = (width >= height) == (bbox.width >= bbox.height)
            extracted = document.extract_image(xref)
            extension = _EMBEDDED_EXTENSIONS.get((extracted or {}).get("ext", ""))
            if covers_page and upright and extension and not extracted.get("smask"):
                data = extracted["image"]
                return data, extension, FirstPageImage(
                    file="", width=extracted["width"], height=extracted["height"], method="embedded",
                    dpi=round(extracted["width"] / (bbox.width / 72), 1),
                )

        if scans:
            dpi = max(width / (bbox.width / 72) for _, width, _, bbox in scans)
        else:
            dpi = default_dpi
        pixmap = page.get_pixmap(dpi=round(dpi), colorspace=pymupdf.csRGB, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue(), "jpg", FirstPageImage(
            file="", width=pixmap.width, height=pixmap.height, method="rendered", dpi=round(dpi, 1),
        )
