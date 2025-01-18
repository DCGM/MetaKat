import os
import json
from PIL import Image, ImageDraw

# Funkce pro vykreslení bounding boxů z JSON na obrázek
def draw_bounding_boxes_from_folder(json_folder, image_folder, output_folder):
    # Iterace přes všechny JSON soubory ve složce
    for json_filename in os.listdir(json_folder):
        if json_filename.endswith(".json"):
            json_path = os.path.join(json_folder, json_filename)
            
            # Načtení JSON dat
            with open(json_path, 'r') as file:
                data = json.load(file)
            
            # Cesta k obrázku podle JSON
            image_basename = os.path.basename(data["data"]["image"])
            image_path = os.path.join(image_folder, image_basename)
            if not os.path.exists(image_path):
                print(f"Obrázek {image_path} nebyl nalezen pro JSON {json_filename}.")
                continue
            
            # Otevření obrázku
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # Načtení výsledků predikcí
            predictions = data.get("predictions", [])
            if not predictions:
                print(f"Chybí predikce v souboru {json_filename}.")
                continue
            predictions = predictions[0]["result"]
            
            # Rozměry obrázku pro převod relativních souřadnic na pixely
            img_width, img_height = image.size
            
            # Vykreslení každého bounding boxu
            for item in predictions:
                bbox = item["value"]
                x = bbox["x"] * img_width / 100
                y = bbox["y"] * img_height / 100
                width = bbox["width"] * img_width / 100
                height = bbox["height"] * img_height / 100
                
                # Vypočtení rohových bodů bounding boxu
                top_left = (x, y)
                bottom_right = (x + width, y + height)
                
                # Barva a popisek boxu
                label = bbox["rectanglelabels"][0]
                color = "red"
                draw.rectangle([top_left, bottom_right], outline=color, width=5)
                draw.text((x, y - 10), label, fill=color)
            
            # Uložení výsledného obrázku
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, image_basename)
            image.save(output_path)
            print(f"Uložený obrázek: {output_path}")

# Příklad použití
json_dir = "/home/maja/Plocha/digilinka_knihy2/tasks"          # Cesta ke složce s JSON soubory
image_dir = "/home/maja/Plocha/digilinka_knihy2/images"       # Cesta ke složce s obrázky
output_dir = "/home/maja/Plocha/digilinka_knihy2/out_check"      # Cesta ke složce pro výstupy
draw_bounding_boxes_from_folder(json_dir, image_dir, output_dir)
