import os
import sys

# Funkce pro vytvoření seznamu názvů souborů z train složky
def save_train_ids(train_folder):
    train_ids_path = os.path.join(train_folder, "train_ids.txt")
    
    # Iterace přes všechny soubory v train složce
    train_ids = []
    for filename in os.listdir(train_folder):
        if filename.endswith('.txt'):
            file_id = os.path.splitext(filename)[0]
            train_ids.append(file_id)
    
    # Uložení do train_ids.txt
    with open(train_ids_path, "w") as train_ids_file:
        for train_id in sorted(train_ids):  # Volitelně seřadíme názvy
            train_ids_file.write(f"{train_id}\n")
    
    print(f"ID trénovací sady uložena do: {train_ids_path}")

# Hlavní část programu
if len(sys.argv) < 2:
    print("Použití: python skript.py <cesta_ke_train_složce>")
    sys.exit(1)

train_folder = sys.argv[1]

if not os.path.exists(train_folder):
    print(f"Chyba: Složka {train_folder} neexistuje!")
    sys.exit(1)

# Zavolání funkce pro uložení názvů souborů ze složky train
save_train_ids(train_folder)
