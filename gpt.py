import os
import glob
import json
import base64
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal
from PIL import Image

# --- Settings ---
OPENAI_API_KEY = "KEY"
YOLO_MODEL_PATH = '../runs/bp_experiment/weights/best.pt'
INPUT_FOLDER = 'images/images_2'
BASE_OUTPUT_FOLDER = 'results_llm'
AI_MODEL = "gpt-4o"


client = OpenAI(api_key=OPENAI_API_KEY)


class StructureItem(BaseModel):
    text: str
    category: Literal[
        "chapter_L1",
        "chapter_L2",
        "info_block",
        "page_number",
        "chapter_number"
    ]


class StructureResponse(BaseModel):
    items: List[StructureItem]


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def normalize_json_response(json_data):

    if "items" in json_data:
        return json_data

    for key, value in json_data.items():
        if isinstance(value, list):
            return {"items": value}

    return {"items": []}


def create_system_prompt():
    return """
    You are an expert Document Layout Analyzer.
    
    TASK:
    1. Analyze the image of the Table of Contents.
    2. Extract all distinct text regions, don't extract the titiles such as "Obsah", "Contents" etc.
    3. Categorize them.
 
    CATEGORIES:
    - "chapter_L1": Chapters of the first level.
    - "chapter_L2": Sub-chapters(chapters of the second level).
    - "info_block": Descriptions/Authors or any additional infromation.
    - "page_number": Page numbers.
    - "chapter_number": Standalone numbers (such as "1.", "II.") that refers to the chapter.

    OUTPUT FORMAT:
    JSON object with a list.
    Example:
    {
        "items": [
        {
          "category": "chapter_L1",
          "text": "Introduction",
        }
      ]
    }
    """


def process_single_image(image_path, file_id, output_dir):
    print(f"--- Processing: {file_id} ---")

    # LLM Request
    base64_img = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": create_system_prompt()},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract structure."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ]
        )
        json_res = json.loads(response.choices[0].message.content)

        json_res = normalize_json_response(json_res)

        llm_data = StructureResponse(**json_res)

    except Exception as e:
        print(f"LLM Error: {e}")
        return

    final_data = []

    for i, item in enumerate(llm_data.items):
        final_data.append({
            "id": i,
            "category": item.category,
            "text": item.text
        })

    output_json_path = os.path.join(
        output_dir, f"{file_id}_output_gpt.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)


def main():
    if not os.path.exists(BASE_OUTPUT_FOLDER):
        os.makedirs(BASE_OUTPUT_FOLDER)
    image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.jpg'))) + \
        sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.png')))

    print(f"Found {len(image_files)} images.")

    for idx, image_path in enumerate(image_files):
        file_name = os.path.basename(image_path)
        file_id = os.path.splitext(file_name)[0]

        main_output_dir = os.path.join(BASE_OUTPUT_FOLDER, file_id)
        json_output_dir = os.path.join(main_output_dir, 'interni_format')
        os.makedirs(json_output_dir, exist_ok=True)

        process_single_image(image_path, file_id, json_output_dir)


if __name__ == "__main__":
    main()
