import os
from ultralytics import YOLO
from toc_only.data_types import BookData


def run_yolo_sorting(images_folder: str, book: BookData, yolo_model_path: str):
    """
    Looking for different types of pages
    """
    if not os.path.exists(images_folder):
        print(f"[ERROR]: Folder '{images_folder}' is not found!")
        return

    print(f"Loading YOLO model ...")
    model = YOLO(yolo_model_path)

    # Sorting images of the book
    image_files = sorted([f for f in os.listdir(images_folder)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])

    for page_num, filename in enumerate(image_files, start=1):
        img_path = os.path.join(images_folder, filename)

        # YOLO prediction
        results = model(img_path, verbose=False, conf=0.25)

        # Count classes on the page
        class_counts = {
            "cislo strany": 0,
            "kapitola": 0,
            "jiny nadpis": 0,
            "nadpis v textu": 0
        }

        page_chapter_titles = []
        page_number_boxes = []

        for box in results[0].boxes:
            conf = box.conf[0].item()
            if conf > 0.25:
                cls_name = model.names[int(box.cls[0].item())]

                if cls_name in class_counts:
                    class_counts[cls_name] += 1

                # Saving coords of the chapter name
                if cls_name == "nadpis v textu":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    page_chapter_titles.append({
                        "coords": [x1, y1, x2, y2],
                        "conf": round(conf, 2)
                    })
                if cls_name == "cislo strany":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    page_number_boxes.append({
                        "coords": [x1, y1, x2, y2],
                        "conf": round(conf, 2)
                    })

        # Rules for ToC page
        is_toc_candidate = ((class_counts["kapitola"] >= 3) and (class_counts["cislo strany"] >= 3)) or \
            (((class_counts["kapitola"] + class_counts["jiny nadpis"])
             >= 3) and (class_counts["cislo strany"] >= 2))

        is_chapter_start = class_counts["nadpis v textu"] > 0

        # ToC can be in first 15% or in last 10% of the book
        is_toc = False
        if is_toc_candidate:
            total_pages = len(image_files)
            if (page_num < total_pages * 0.15) or (page_num > total_pages * 0.90):
                is_toc = True

        # Adding info in Book class
        if is_toc:
            book.toc_pages.append({
                "page_num": page_num,
                "filename": filename,
                "score": class_counts["kapitola"] + class_counts["cislo strany"] + class_counts["jiny nadpis"]
            })

        elif is_chapter_start:
            book.chapter_pages.append({
                "page_num": page_num,
                "filename": filename,
                "titles_to_crop": page_chapter_titles,
                "page_numbers_to_crop": page_number_boxes
            })

        else:
            book.ignored_pages_count += 1

    if book.toc_pages:
        groups = []
        current_group = [book.toc_pages[0]]

        for i in range(1, len(book.toc_pages)):
            prev_page = book.toc_pages[i-1]["page_num"]
            curr_page = book.toc_pages[i]["page_num"]

            if curr_page == prev_page + 1:
                current_group.append(book.toc_pages[i])
            else:
                groups.append(current_group)
                current_group = [book.toc_pages[i]]
        groups.append(current_group)

        if len(groups) > 1:

            # Calculating the 'power' of every ToC
            def get_group_score(group):
                return sum(page["score"] for page in group)

            # Choosing the best one
            best_group = max(groups, key=get_group_score)

            # Saving to memory only real one
            book.toc_pages = best_group

    print(f"[INFO] Table of Contents pages: {len(book.toc_pages)}")
    print("-" * 40)
    print("[INFO] PDF Sorting - Done!\n")
