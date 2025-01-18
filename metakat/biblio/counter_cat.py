import os
from collections import Counter
import sys

def count_categories_in_folder(folder_path):
    category_counts = Counter()
    file_categories = {}  # Slovník na uložení kategorií pro každý soubor

    # Procházení všech souborů ve složce
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # Čtení souboru a extrakce kategorií
            file_category_counts = Counter()
            with open(file_path, 'r') as file:
                for line in file:
                    # Extrakce prvního čísla z řádku (kategorie)
                    category = line.split()[0]
                    category_counts[category] += 1
                    file_category_counts[category] += 1

            # Uložení kategorií pro aktuální soubor
            file_categories[filename] = dict(file_category_counts)

    return category_counts, file_categories

if len(sys.argv) != 2:
    print("Použití: python count_categories.py <cesta_ke_složce_txt>")
    sys.exit(1)

folder_path = sys.argv[1]

if not os.path.exists(folder_path):
    print(f"Chyba: Složka {folder_path} neexistuje!")
    sys.exit(1)

# Spočítání kategorií
category_counts, file_categories = count_categories_in_folder(folder_path)

# Výpis výsledků seřazených vzestupně podle číselných hodnot kategorií
print("Počet záznamů v jednotlivých kategoriích (seřazeno vzestupně):")
for category, count in sorted(category_counts.items(), key=lambda x: int(x[0])):
    print(f"Kategorie {category}: {count}")

# Výpis názvů souborů a kategorií, které obsahují
print("\nSoubory a jejich kategorie:")
for filename, categories in file_categories.items():
    print(f"{filename}:")
    for category, count in sorted(categories.items(), key=lambda x: int(x[0])):
        print(f"  Kategorie {category}: {count}")
