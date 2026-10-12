import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("Iris.csv"))

    def test_model_exists(self):
        self.assertTrue(os.path.exists("iris_model.pkl"))

    def test_metrics_exist(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_expected_dataset_split(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertEqual(metrics["total_records"], 150)
        self.assertEqual(metrics["training_records"], 120)
        self.assertEqual(metrics["testing_records"], 30)

    def test_model_prediction(self):
        model = joblib.load("iris_model.pkl")

        sample = pd.DataFrame([{
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2,
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            prediction,
            ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        )

    def test_model_output_classes(self):
        model = joblib.load("iris_model.pkl")

        classes = list(
            model.named_steps["classifier"].classes_
        )

        self.assertEqual(len(classes), 3)


if __name__ == "__main__":
    unittest.main()
