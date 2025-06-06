from tensorflow.keras.models import load_model
import json
import os

def load_model_and_classes(model_path, class_indices_path):
    print(f"Loading model from: {model_path}") 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class indices file not found: {class_indices_path}")

    model = load_model(model_path)  # Must return the model object, NOT a path
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    return model, class_indices
