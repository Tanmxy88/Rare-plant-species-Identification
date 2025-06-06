import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image from disk, resize it, and normalize pixel values.
    """
    try:
        if not Path(image_path).exists():
            logging.error(f"Image file not found: {image_path}")
            return None
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = img_to_array(image) / 255.0  # Normalize to [0,1]
        return image
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def preprocess_batch(image_paths, target_size=(224, 224)):
    """
    Preprocess a batch of images.
    """
    images = [load_and_preprocess_image(path, target_size) for path in image_paths]
    images = [img for img in images if img is not None]
    return np.array(images)
