import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path, data_dir, img_size=(224, 224), batch_size=32):
    """
    Evaluate the trained model on test data and print/save classification metrics.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Test directory {data_dir} invalid")

    model = load_model(model_path)
    print(f"Loaded model summary:")
    model.summary()

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    preds = model.predict(test_generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    print("Confusion Matrix:\n")
    print(cm)

    # Save metrics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'evaluation_{timestamp}.json', 'w') as f:
        json.dump({'report': report, 'confusion_matrix': cm.tolist()}, f)
    print(f"Saved evaluation metrics to evaluation_{timestamp}.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model on test data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test dataset directory')
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_dir)
#     evaluate_model(args.model_path, args.test_dir)