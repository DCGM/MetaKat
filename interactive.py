import fitz
from toc_only.data_types import BookData


def make_pdf(book: BookData, output_pdf_path: str):
    doc = fitz.open(book.pdf_path)
    pdf_chapters = []   # List of the chapters

    def process_chapter(chapter_node, level=1):
        chapter = chapter_node.get("chapter_name", "Untitled")
        chapter_page = chapter_node.get("physical_page_num")

        if chapter_page is not None:
            pdf_chapters.append([level, chapter, chapter_page])

            source_page_id = chapter_node.get("toc_source_page")
            polygon = chapter_node.get("toc_polygon")

            if source_page_id and polygon and len(polygon) >= 3:

                source_pdf_page_id = int(source_page_id.split('_')[1]) - 1
                source_pdf_page = doc[source_pdf_page_id]

                # DPI = 450 -> scale = 72/450(1 Point = 1/72 inch)
                scale = 72 / 450

                # Convert coords
                x1 = polygon[0][0] * scale
                y1 = polygon[0][1] * scale
                x2 = polygon[2][0] * scale
                y2 = polygon[2][1] * scale

                # Hypertext square
                rect = fitz.Rect(x1, y1, x2, y2)

                # Type of action, from, where
                link_data = {
                    "kind": fitz.LINK_GOTO,
                    "from": rect,
                    "page": chapter_page - 1
                }
                source_pdf_page.insert_link(link_data)

                # source_pdf_page.draw_rect(rect, color=(1, 0, 0), width=0.5)

        # works recursively with subchapters
        for sub in chapter_node.get("subchapters", []):
            process_chapter(sub, level + 1)

    for item in book.final_structure:
        process_chapter(item, level=1)

    doc.set_toc(pdf_chapters)
    doc.save(output_pdf_path)
    doc.close()
    print(
        f"[INFO] Interactive PDF - Done! Saved as: {output_pdf_path}\n")
