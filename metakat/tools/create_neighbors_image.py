"""
File: create_neighbors_image.py
Author: [Matej Smida]
Date: 2025-05-12
Description: Processes a text file that contains image and its neighbors and puts them in one image.
            file must be in format main: img_id
                                   prev: img_id
                                   next: img_id
                                   ...
             for [for creating dataset].
"""

from PIL import Image
import os, argparse

#resizes images so all images in final image will have the same height
def resize_keep_aspect(img, target_height):
    if img.width == 0 or img.height == 0 or target_height == 0:
        raise ValueError(f"Invalid image dimensions: ({img.width}, {img.height}), target height: {target_height}")

    aspect_ratio = img.width / img.height
    new_width = max(1, int(target_height * aspect_ratio))
    return img.resize((new_width, target_height))


def create_image_with_neighbors(top, left, right, padding):
    heights = [img.height for img in [left, right] if img]
    common_height = min(heights) if heights else top.height

    if left:
        left = resize_keep_aspect(left, common_height)
    if right:
        right = resize_keep_aspect(right, common_height)

    #creates empty black space if left or right image is not present
    if left and not right:
        right = Image.new("RGB", (left.width, common_height), color="black")
    if right and not left:
        left = Image.new("RGB", (right.width, common_height), color="black")
    if not left and not right:
        default_width = top.width
        left = Image.new("RGB", (default_width, common_height), color="black")
        right = Image.new("RGB", (default_width, common_height), color="black")

    bottom_width = left.width + padding + right.width

    top = resize_keep_aspect(top, common_height)

    canvas_width = max(bottom_width, top.width)
    canvas_height = common_height * 2 + padding

    result = Image.new("RGB", (canvas_width, canvas_height), color="black")

    top_x = (canvas_width - top.width) // 2
    result.paste(top, (top_x, 0))

    y_offset = common_height + padding
    result.paste(left, (0, y_offset))
    result.paste(right, (left.width + padding, y_offset))

    return result

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pages", required=True, help="text file with image ids")
    parser.add_argument("--images", required=True, help="folder with images")
    parser.add_argument("--neighbors_images", required=True, help="folder with neighbors images")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--padding", default=20, help="padding around images")

    return parser.parse_args()

def main():
    args = arg_parse()

    os.makedirs(args.out, exist_ok=True)
    padding = args.padding

    main_id = None
    prev_id = None
    next_id = None

    has_prev = False
    has_next = False


    for i, line in enumerate(open(args.pages, "r")):
        if (i + 1) % 1000 == 0:
            print(f"processed {i+1} images")

        if line.startswith("main"):
            main_id = line.strip().replace("main: ", "")

        if line.startswith("prev"):
            has_prev = True
            prev_id = line.strip().replace("prev: ", "")

            if prev_id == "None" or "ERR" in line:
                prev_id = None

        if line.startswith("next"):
            has_next = True
            next_id = line.strip().replace("next: ", "")

            if next_id == "None" or "ERR" in line:
                next_id = None

        if (i + 1) % 3 == 0:
            if has_prev is True and has_next is True:
                main_img = Image.open(os.path.join(args.images, main_id))

                if prev_id is not None:
                    prev_img = Image.open(os.path.join(args.neighbors_images, prev_id))
                else:
                    prev_img = None

                if next_id is not None:
                    next_img = Image.open(os.path.join(args.neighbors_images, next_id))
                else:
                    next_img = None

                final_img = create_image_with_neighbors(main_img, prev_img, next_img, padding)

                save_path = os.path.join(args.out, main_id)

                try:
                    final_img.save(save_path)
                except OSError as e:
                    print(f"[WARNING] Failed to save '{save_path}' due to: {e}")

            has_prev = False
            has_next = False

            main_id = None
            prev_id = None
            next_id = None

            #cleanes RAM space
            main_img.close()
            if prev_img: prev_img.close()
            if next_img: next_img.close()
            final_img.close()

if __name__ == "__main__":
    main()