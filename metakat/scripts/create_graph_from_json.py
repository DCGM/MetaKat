import json

# json_file = "/home/matko/Desktop/weighted_recall.json"
# json_file = "/home/matko/Desktop/weighted_fscore.json"
json_file = "/home/matko/Desktop/weighted_precision.json"
target_name = "tst.neighbors"
out = "/home/matko/Desktop/clip_neighbors_weighted_precision.dat"

try:
    with open(json_file, "r", encoding='utf-8') as f_in:
        data = json.load(f_in)

        target_data = None
        for item in data:
            if isinstance(item, dict) and item.get('name') == target_name:
                target_data = item
                break


        if target_data:
            x_values = target_data.get('x')
            y_values = target_data.get('y')

            if x_values and y_values:
                with open(out, "w", encoding='utf-8') as f_out:
                    for x, y in zip(x_values, y_values):
                        f_out.write(f"{x} {y}\n")
            else:
                print("Error in json file")

        else:
            print("Error: No target data found")

except FileNotFoundError:
    print(f"Error: {json_file} not found")
except json.JSONDecodeError:
    print(f"Chyba: Soubor '{json_file}' neobsahuje validní JSON.")
except Exception as e:
    print(f"Nastala neočekávaná chyba: {e}")