# XGBoost training script
# Author: Richard Blažo
# File name: train_xgb.py
# Description: Script used to train an XGBoost model for predicting the hierarchy of chapters
# and subchapters.

import json

from src.utils.build_training_data import build_training_data_xgb
from src.models.xgbClassifier import train_chapter_classifier

with open("data/XGB/digilinka.json", "r") as f:
    raw_data = json.load(f)

training_data = build_training_data_xgb(raw_data)
model = train_chapter_classifier(training_data)
model.save_model("models/XGB/new_new.ubj")
