import os
import glob
import cv2
import configparser
from pero_ocr.document_ocr.page_parser import PageParser
from pero_ocr.core.layout import PageLayout

# --- Settings  ---
PERO_CONFIG_PATH = './peromodel/config_cpu.ini'
INPUT_FOLDER = 'images/images_2'
BASE_OUTPUT_FOLDER = 'results_pero'

# I dont have gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def find_original_image(file_id):
    extensions = ['.jpg', '.jpeg', '.png']
    for ext in extensions:
        path = os.path.join(INPUT_FOLDER, file_id + ext)
        if os.path.exists(path):
            return path
    return None


def main():
    print("Init Pero OCR...")
    if not os.path.exists(PERO_CONFIG_PATH):
        return print("Config not found")

    config = configparser.ConfigParser()
    config.read(PERO_CONFIG_PATH)
    try:
        page_parser = PageParser(
            config, config_path=os.path.dirname(PERO_CONFIG_PATH))
    except Exception as e:
        return print(f"Error Pero Init: {e}")

    # Searching folders with yolo results in "results"
    result_dirs = sorted(glob.glob(os.path.join(BASE_OUTPUT_FOLDER, '*')))

    for idx, main_output_dir in enumerate(result_dirs):
        if not os.path.isdir(main_output_dir):
            continue

        file_id = os.path.basename(main_output_dir)

        input_xml_path = os.path.join(
            main_output_dir, 'yolo', f"{file_id}_regions.xml")

        # Creating peroocr folder
        pero_output_dir = os.path.join(main_output_dir, 'peroocr')

        if not os.path.exists(input_xml_path):
            continue

        if not os.path.exists(pero_output_dir):
            os.makedirs(pero_output_dir)

        print(f"[{idx+1}/{len(result_dirs)}] Pero Processing: {file_id}")

        image_path = find_original_image(file_id)
        if not image_path:
            print(f"Original image not found for {file_id}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        # Getting XML from YOLO
        layout_for_pero = PageLayout()
        try:
            layout_for_pero.from_pagexml(input_xml_path)
        except Exception as e:
            print(f"Error reading XML: {e}")
            continue

        # OCR working
        try:
            page_parser.process_page(image, layout_for_pero)
        except Exception as e:
            print(f"Error Pero Processing: {e}")
            continue

        # Saving render and final xml
        cv2.imwrite(os.path.join(pero_output_dir, f"{file_id}_render.jpg"),
                    layout_for_pero.render_to_image(image))

        final_xml_path = os.path.join(pero_output_dir, f"{file_id}_final.xml")
        layout_for_pero.to_pagexml(final_xml_path)


if __name__ == "__main__":
    main()
