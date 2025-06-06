import unittest
from ai_model.inference import PlantSpeciesIdentifier
import os
import json

class TestInference(unittest.TestCase):

    def test_prediction_on_sample_image(self):
        # Load sample test image
        image_path = "datasets/test/species_1/img0001.jpg"
        self.assertTrue(os.path.exists(image_path), "Test image does not exist.")

        # Load class indices
        with open("backend/class_indices.json", "r") as f:
            class_indices = json.load(f)

        # Run prediction
        model_path = "ai_model/plant_model.h5"
        identifier = PlantSpeciesIdentifier(model_path, class_indices)
        species, confidence = identifier.predict(image_path)

        self.assertIsInstance(species, str)
        self.assertIsInstance(confidence, float)
        print(f"Predicted: {species} with {confidence*100:.2f}% confidence")

if __name__ == '__main__':
    unittest.main()
