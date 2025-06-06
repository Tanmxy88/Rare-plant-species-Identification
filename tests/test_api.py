import unittest
import requests

class TestAPI(unittest.TestCase):

    def test_root_endpoint(self):
        response = requests.get("http://127.0.0.1:5000/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_predict_endpoint(self):
        image_path = "datasets/test/species_1/img0001.jpg"
        with open(image_path, "rb") as img:
            files = {"image": ("img0001.jpg", img, "image/jpeg")}
            response = requests.post("http://127.0.0.1:5000/api/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("species", data)
        self.assertIn("confidence", data)
        print(f"Prediction: {data['species']} ({data['confidence']*100:.2f}%)")

if __name__ == '__main__':
    unittest.main()
