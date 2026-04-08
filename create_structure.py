from collections import Counter
from toc_only.data_types import BookData
from helpers import normalize_text, flatten_tree


# Looking for a physical pages in book
def calculate_final_structure(book: BookData):

    toc_data = book.theoretical_toc
    actual_data = book.actual_chapters

    if not toc_data or not actual_data:
        print("[ERROR]: No data!")
        return

    # flatten all chapters and subchapters for offset voting
    toc_items_for_offset = flatten_tree(toc_data)

    # Offset
    offset_votes = []
    used_for_offset = set()  # dont calculate the same chapters

    # Taking all chapters in ToC and trying to find it beetween "real" chapters in the book
    for toc_item in toc_items_for_offset:
        logical_page = toc_item.get('start_page')
        if not logical_page:
            continue

        toc_title = normalize_text(toc_item.get('chapter_name', ''))

        for idx, actual_item in enumerate(actual_data):
            if idx in used_for_offset:
                continue

            actual_title = normalize_text(
                actual_item.get('extracted_text', ''))
            physical_page = actual_item.get('physical_page')

            if len(actual_title) > 3 and (actual_title in toc_title or toc_title in actual_title):
                current_offset = physical_page - logical_page
                offset_votes.append(current_offset)
                used_for_offset.add(idx)
                break

    if not offset_votes:
        print("[ERROR]: No one match in offset!")
        return

    most_common_offset = Counter(offset_votes).most_common(1)[0][0]
    print(f"[INFO] Final Offset: {most_common_offset}")

    used_actual_indices = set()

    def process_node(node):
        logical_page = node.get('start_page')
        toc_title = normalize_text(node.get('chapter_name', ''))

        physical_page = None
        extracted_page_number = None

        # Looking for the chapter in book(its real page)
        for idx, actual_item in enumerate(actual_data):
            if idx in used_actual_indices:
                continue
            actual_title = normalize_text(
                actual_item.get('extracted_text', ''))

            if len(actual_title) > 3 and actual_title in toc_title:
                physical_page = actual_item.get('physical_page')
                raw_extracted_num = actual_item.get('extracted_page_number')
                if raw_extracted_num and raw_extracted_num.isdigit():
                    extracted_page_number = int(raw_extracted_num)

                used_actual_indices.add(idx)
                break

        # Real page
        if physical_page is not None:
            if logical_page is None:
                if extracted_page_number is not None:
                    logical_page = extracted_page_number
                else:
                    logical_page = physical_page - most_common_offset
        else:
            if logical_page is not None:
                physical_page = logical_page + most_common_offset

        # Working with subchapters
        processed_subchapters = []
        for sub in node.get("subchapters", []):
            processed_subs = process_node(sub)
            if processed_subs:
                processed_subchapters.extend(processed_subs)

        # Returning node
        if physical_page is not None:
            return [{
                "chapter_name": node.get("chapter_name"),
                "logical_page_printed": logical_page,
                "physical_page_num": physical_page,
                "toc_source_page": node.get("page_name"),
                "toc_polygon": node.get("polygon"),
                "subchapters": processed_subchapters
            }]
        else:
            return processed_subchapters

    # For all chapters
    for toc_item in toc_data:
        processed_items = process_node(toc_item)
        if processed_items:
            book.final_structure.extend(processed_items)

    print("-" * 40)
    print("[INFO] Final structure - Done!")
