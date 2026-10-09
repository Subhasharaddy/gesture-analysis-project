import sys
import os
import joblib
import numpy as np
sys.path.append(os.getcwd())
import config
from utils.gesture_database import class_id_to_gesture_name

try:
    model = joblib.load(config.MODEL_PKL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    
    # Create dummy vector
    dummy = np.zeros(73)
    probs = model.predict_proba([dummy])[0]
    best_idx = int(np.argmax(probs))
    raw_cls = model.classes_[best_idx]
    
    print("best_idx (argmax of proba):", best_idx)
    print("raw_cls (model.classes_[best_idx]):", raw_cls)
    
    idx_val = int(raw_cls)
    print("label_encoder.classes_[idx_val]:", label_encoder.classes_[idx_val])
    print("label_encoder.inverse_transform([idx_val]):", label_encoder.inverse_transform([idx_val])[0])
    print("class_id_to_gesture_name(idx_val):", class_id_to_gesture_name(idx_val))
except Exception as e:
    print("Error:", e)
