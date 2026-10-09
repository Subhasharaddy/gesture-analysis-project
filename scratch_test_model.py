import sys
import os
import joblib
sys.path.append(os.getcwd())
import config

try:
    model = joblib.load(config.MODEL_PKL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    print("Model classes type:", type(model.classes_[0]))
    print("Model classes sample:", model.classes_[:5])
    print("Label encoder classes sample:", label_encoder.classes_[:5])
except Exception as e:
    print("Error:", e)
