# Projekt: Multilabel rozpoznávání pojmenovaných entit pomocí LayoutLMv3

## Přehled

Tato složka obsahuje skripty a konfigurace pro trénování, vyhodnocení a predikci úloh rozpoznávání pojmenovaných entit (NER) ve formátu multilabel na základě dokumentových obrázků s využitím modelu **LayoutLMv3** od Microsoftu. Je potřeba výkonnou GPU a obrázky k datasetu. Obsahuje:

- **`trainer.py`** – Hlavní kompletní trénovací pipeline pro multilabel tokenovou klasifikaci. Zajišťuje výpočet vah tříd, trénování s vlastní ztrátovou funkcí, vyhodnocování metrik a ukládání výstupů jako JSON.
- **`trainer_*.py`** – Experimentální trénovací skripty
- **`layoutlmv3.py`** – Vlastní builder pro knihovnu HuggingFace Datasets (`BiblioExtraction`) umožňující načítání, předzpracování a poskytování datasetu s bibliografickými informacemi (včetně tokenizace, načtení obrázků a normalizace souřadnic).

## Struktura složky `LayoutLMv3/`

```bash
LayoutLMv3/
├── READMEllmv3.md # Tento popisný soubor
├── dataset_prepare/ # Složka pro přípravu dat
├── train/ # Složka pro trénování modelu
│ ├── layoutlmv3/ # Datový adresář s anotacemi a obrázky
│ │ ├── class_list.txt # Seznam tříd (entit)
│ │ ├── trainM.txt # Trénovací data (každý řádek = 1 dokument)
│ │ ├── testM.txt # Testovací data
│ │ └── <obrázky> # Obrázky dokumentů (nejsou součástí odevzdání)
│ ├── layoutlmv3.py # Dataset builder pro HuggingFace
│ └── trainer.py # Skript pro trénování, evaluaci a predikci
│ └── trainer_geo.py # Skript pro trénování, evaluaci a predikci pro model s geometrií
│ └── trainer_geo_text.py # Skript pro trénování, evaluaci a predikci pro model s geometrií a textem
│ └── trainer_large.py # Skript pro trénování, evaluaci a predikci pro large model
```


## Závislosti

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- torchvision
- scikit-learn
- pandas
- Pillow

## Použití

### 1. Příprava dat

Data musí být strukturována dle výše uvedeného příkladu. Skript `layoutlmv3.py` definuje builder `BiblioExtraction`, který:

- Načte třídy z `class_list.txt`
- Zpracuje `trainM.txt` a `testM.txt` (jedna položka na řádek ve formátu JSON)
- Načítá obrázky a převádí souřadnice do měřítka 0–1000

### 2. Trénování / Vyhodnocení / Predikce

Pro spuštění tréninku, evaluace nebo predikce skript `trainer.py`.

**Trénink:**

```bash
python trainer.py --train \
  --cache_dir ./cache \
  --output_dir ./outputs \
  --batch_size 2 \
  --epochs 20 \
  --pos_weight_path pos_weight.pt
```

**Evaluace**:

```bash
python trainer.py --eval \
  --cache_dir ./cache \
  --model_dir ./outputs \
  --output_dir ./eval_logs \
  --threshold 0.5
```

**Predict**:

```bash
python trainer.py --predict \
  --cache_dir ./cache \
  --model_dir ./outputs \
  --output_dir ./predictions \
  --threshold 0.5
```

Predikce budou uloženy jako JSON soubory ve složce `output_dir/prediction_outputs`.

## Konfigurační možnosti

* `--cache_dir`: Cesta pro uložení cache datasetů (HuggingFace)
* `--output_dir`: Složka pro ukládání modelu a logů
* `--batch_size`: Velikost batch na zařízení
* `--epochs`: Počet trénovacích epoch
* `--pos_weight_path`: Cesta pro uložení vah tříd
* `--model_dir`: Cesta k modelu
* `--threshold`: Prahová hodnota pro binarizaci vícetřídních výstupů

## Licence

Tento projekt využívá licenci MIT, stejně jako původní kód LayoutLMv3. Úpravy provedené Marií Pařilovou (2025) jsou poskytovány rovněž pod licencí MIT.

## Citace

Pokud tento kód použijete, prosím uveďte citaci:

```
@misc{parilova2025multilabel,
  author = {Pařilová, Marie},
  title = {Multi-Label NER with LayoutLMv3},
  year = {2025},
  note = {Načítání datasetu upraveno z projektu shivarama23/LayoutLMV3}
}

```
