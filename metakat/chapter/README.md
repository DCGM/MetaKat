# Before anything
Ensure python virtual environment is activated
./venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Usage and arguments
## Usage
```bash
python3 process_book.py
```
## Arguments
- `--source` (path, default=`data/images`):  
  Path to the input directory containing images to be classified.

- `--output` (path):  
  Directory where resulting JSON files will be saved. If not specified, defaults to the `--source` directory.

- `--imgOut` (path):  
  Directory where annotated images (with drawn bounding boxes) will be saved. Defaults to `image-results` inside the source directory.

- `--labelsOut` (path):  
  Directory where raw bounding box data will be saved. Defaults to `box-results` inside the source directory.

- `--weights` (path, default=`models/YOLO/best.pt`):  
  Path to the YOLO model weights.

- `--iou` (float, default=`0.15`):  
  Intersection-over-Union threshold for filtering predictions.

- `--conf` (float, default=`0.35`):  
  Confidence threshold for filtering predictions.

- `--dontSave` (flag):  
  If set, suppresses saving of images, labels and annotations. Json files will still be saved.

- `--debug` (flag):  
  Enables debug mode for debug output.

- `--twoModels` (flag):  
  Use two YOLO specialised models for classification instead of one universal model.

# Models
Used models can be found in the `models` directory.
## YOLO models
- `gen.pt` - General YOLO model trained on the entire dataset.
- `text_sep.pt` - Specialised YOLO model trained on the text dataset.
- `toc_sep.pt` - Specialised YOLO model trained on the table of contents dataset.
## GNN models
- `chapter_classifier_gnn.pth` - GNN model for chapter hierarchy classification.
## RFC models
- `rfc_model.pkl` - Random Forest Classifier model for page number to chapter mapping.
## XGBoost models
- `latest.ubj` - XGBoost model for chapter hierarchy classification.

# Files
## `process_book.py`
Full pipeline for processing a book.
## `train_gnn.py`
Training script for the GNN model.
## `train_rfc.py`
Training script for the Random Forest Classifier model.
## `train_xgboost.py`
Training script for the XGBoost model.
## `train_yolo.py`
Training script for the YOLO model.
## `testing` folder
Contains testing scripts for the GNN and RFC models.
### `gnn_test.py`
Testing script for the GNN model.
### `rfc_test.py`
Testing script for the Random Forest Classifier model.
### `xgb_test.py`
Testing script for the XGBoost model.
### `yolo_test.py`
Testing script for the YOLO model.
## `src` folder
### `models` folder
Contains model definitions and training scripts.
### `utils` folder
Contains utility functions for data processing, model training, and evaluation.

