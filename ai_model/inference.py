import os
import numpy as np
from tensorflow.keras.models import load_model
from .image_preprocessing import load_and_preprocess_image
import logging

logging.basicConfig(level=logging.INFO)

class PlantSpeciesIdentifier:
    def __init__(self, model_path, class_indices):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        if not class_indices:
            raise ValueError("class_indices cannot be empty")

        self.model = load_model(model_path)
        self.class_indices = class_indices
        self.class_names = {v: k for k, v in class_indices.items()}

    def predict(self, image_path):
        img = load_and_preprocess_image(image_path)
        if img is None:
            raise ValueError(f"Could not process image: {image_path}")
        img = np.expand_dims(img, axis=0)
        preds = self.model.predict(img, verbose=0)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        species = self.class_names[class_id]
        return species, confidence