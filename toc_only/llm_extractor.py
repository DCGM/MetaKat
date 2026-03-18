import cv2
import json
import base64
from openai import OpenAI
import google.genai as genai


class BaseLlmExtractor:
    """Basic class for all LLMs"""

    @staticmethod
    def get_system_prompt(file_ids: list) -> str:
        files_str = ", ".join(file_ids)
        # One prompt for all models
        return f"""
        You are an expert Document Structure Analyzer.
        You are provided with sequential page images of a Table of Contents (it can be one page or multiple).
        The file names of these images, in order, are: [{files_str}].
        Your task is to analyze the provided page images and extract its hierarchical table of contents or logical structure.
        
        INSTRUCTIONS FOR POLYGONS:
        You must provide bounding box coordinates in a normalized 0-1000 scale.
        [0, 0] is the top-left corner of the image, and [1000, 1000] is the bottom-right corner.
        For each element, estimate its bounding box as [x1, y1, x2, y2] in this 0-1000 scale.
        
        You MUST return a JSON object with a single root key "chapters", containing a list of hierarchical chapter objects.
        Each chapter object MUST strictly follow this JSON schema:
        {{
            "chapter_name": str, "Title of the chapter/section as written in the text",
            "bbox_1000": [x1, y1, x2, y2], " The estimated bounding box in 0-1000 scale, or null if absent"
            "page_name": str, "page name, similar for one page"
            "chapter_number": int, "Number or string (e.g., '1', '1.1', 'IV'), or null if absent"
            "start_page": int, "the page number the chapter points to, or null if absent"
            "subchapters": [ ... list of nested chapter objects following this exact same schema ... ]
        }}
        
        There can be some symbols in the names of chapters that suggest that the corresponding part of the text should be taken from 
        the previous chapter name, replace these symbols with the actual text.
        """

    @staticmethod
    def convert_coordinates(chapters: list, img_w: int, img_h: int):
        """Calculating the 0-1000 coords to real one"""
        for chap in chapters:
            bbox = chap.get("bbox_1000")

            if bbox and isinstance(bbox, list) and len(bbox) == 4:
                x1 = int((bbox[0] / 1000.0) * img_w)
                y1 = int((bbox[1] / 1000.0) * img_h)
                x2 = int((bbox[2] / 1000.0) * img_w)
                y2 = int((bbox[3] / 1000.0) * img_h)

                chap["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                chap["polygon"] = []

            if "bbox_1000" in chap:
                del chap["bbox_1000"]

            if "subchapters" in chap and chap["subchapters"]:
                BaseLlmExtractor.convert_coordinates(
                    chap["subchapters"], img_w, img_h)


class LlmGPTExtractor(BaseLlmExtractor):
    def __init__(self, api_key=None, model="gpt-4o"):
        self.model = model
        try:
            self.client = OpenAI(api_key=api_key)
            print(f"OpenAI Client initialized ({model}).")
        except Exception as e:
            self.client = None
            print(f"[ERROR]: Failed to init OpenAI: {e}")

    def extract(self, image, file_id: str):
        return self.extract_multiple([image], [file_id])

    def extract_multiple(self, images: list, file_ids: list):
        if not self.client or not images:
            return []

        img_h, img_w = images[0].shape[:2]

        content_payload = [
            {"type": "text", "text": "Extract the document structure"}]

        for img in images:
            # Convert image to format for AI
            success, buffer = cv2.imencode('.jpg', img)
            if success:
                base64_img = base64.b64encode(buffer).decode("utf-8")
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })

        try:
            # print(
            #     f"Analyzing {len(images)} page(s) using {self.model} ...")
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0,
                messages=[
                    {"role": "system",
                        "content": self.get_system_prompt(file_ids)},
                    {"role": "user", "content": content_payload}
                ]
            )

            result_json = json.loads(response.choices[0].message.content)
            chapters = result_json.get("chapters", [])
            self.convert_coordinates(chapters, img_w, img_h)
            return chapters

        except Exception as e:
            print(f"[ERROR]: GPT Request failed: {e}")
            return []


class LlmGeminiExtractor(BaseLlmExtractor):
    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        self.model = model
        try:
            self.client = genai.Client(api_key=api_key)
            print(f"Gemini Client initialized ({model}).")
        except Exception as e:
            self.client = None
            print(f"[ERROR]: Failed to init Gemini: {e}")

    def extract(self, image, file_id: str):
        return self.extract_multiple([image], [file_id])

    def extract_multiple(self, images: list, file_ids: list):
        if not self.client or not images:
            return []

        img_h, img_w = images[0].shape[:2]

        # Creating parts for the request
        parts = [{"text": "Extract the document structure"}]

        for img in images:
            success, buffer = cv2.imencode('.jpg', img)
            if success:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buffer).decode("utf-8")
                    }
                })

        try:
            # Request
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": parts}],
                config={
                    "system_instruction": self.get_system_prompt(file_ids),
                    "response_mime_type": "application/json",
                    "temperature": 0.0
                }
            )

            result_json = json.loads(response.text)
            chapters = result_json.get("chapters", [])
            self.convert_coordinates(chapters, img_w, img_h)
            return chapters

        except Exception as e:
            print(f"[ERROR]: Gemini Request failed: {e}")
            return []
