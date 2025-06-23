# RFC training script
# Author: Richard Blažo
# File name: train_rfc.py
# Description: Script used to train a Random Forest Classifier model for assigning
# page numbers to chapters on a TOC page.

import pickle

from src.models.random_forest_module import prepare_forest_classifier

model = prepare_forest_classifier("data/RFC/tasks_updated.json")
print("Model prepared and trained successfully.")
print("Saving to models/RFC/rfc_new.pkl")
with open("models/RFC/rfc_new.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")
