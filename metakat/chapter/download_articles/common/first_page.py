from __future__ import annotations

import io

import pymupdf
from PIL import Image

from metakat.chapter.download_articles.common.models import FirstPageImage

# A single image covering at least this share of the page is treated as the page scan; scans are
# often placed inside page margins, so it is well below the whole page.
SCAN_COVERAGE = 0.5
# Pages without any scan (born-digital PDFs) have no resolution of their own.
DEFAULT_RENDER_DPI = 300
# A small sharp photo or logo on a born-digital page would ask for 1000-2000 dpi.
MAX_RENDER_DPI = 600

# Formats stored as extracted; PyMuPDF already converts fax, JBIG2 and raw images to lossless PNG.
_EMBEDDED_EXTENSIONS = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "jpx": "jp2", "jp2": "jp2", "tiff": "tif", "tif": "tif"}


def extract_first_page(pdf_bytes: bytes, page_index: int = 0,
                       default_dpi: float = DEFAULT_RENDER_DPI) -> tuple[bytes, str, FirstPageImage]:
    """Return a page of a PDF as image bytes, their file extension and how they were obtained.

    A scanned page is kept at the resolution it was scanned at: a single upright scan covering most
    of the page is stored as extracted, without the page margins around it; any other page is
    rendered at the highest resolution of its images, but at least ``default_dpi`` (born-digital
    pages, perhaps with a coarse figure) and at most ``MAX_RENDER_DPI`` (a sharp photo or logo).
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if page_index >= document.page_count:
            raise ValueError(f"PDF has {document.page_count} pages, page {page_index + 1} requested")
        page = document[page_index]
        page_area = page.rect.width * page.rect.height

        scans = []
        for image in page.get_images(full=True):
            xref, width, height = image[0], image[2], image[3]
            for bbox in page.get_image_rects(xref):
                if not bbox.is_empty:
                    scans.append((xref, width, height, bbox))

        if len(scans) == 1 and page.rotation == 0:
            xref, width, height, bbox = scans[0]
            covers_page = bbox.width * bbox.height >= SCAN_COVERAGE * page_area
            upright = (width >= height) == (bbox.width >= bbox.height)
            extracted = document.extract_image(xref)
            if covers_page and upright and extracted and not extracted.get("smask"):
                extension = _EMBEDDED_EXTENSIONS.get(extracted["ext"])
                data = extracted["image"]
                if extension is None:
                    data, extension = pymupdf.Pixmap(document, xref).tobytes("png"), "png"
                return data, extension, FirstPageImage(
                    file="", width=extracted["width"], height=extracted["height"], method="embedded",
                    dpi=round(extracted["width"] / (bbox.width / 72), 1),
                )

        # The sharpest image is the scan's resolution (scans may be cropped to the text block or cut in
        # strips). On a born-digital page it may be a figure or a logo of any resolution, so the page
        # gets at least ``default_dpi`` and, whatever the figure, at most ``MAX_RENDER_DPI``.
        dpi = max([default_dpi] + [width / (bbox.width / 72) for _, width, _, bbox in scans])
        dpi = min(dpi, MAX_RENDER_DPI)
        pixmap = page.get_pixmap(dpi=round(dpi), colorspace=pymupdf.csRGB, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue(), "jpg", FirstPageImage(
            file="", width=pixmap.width, height=pixmap.height, method="rendered", dpi=round(dpi, 1),
        )
