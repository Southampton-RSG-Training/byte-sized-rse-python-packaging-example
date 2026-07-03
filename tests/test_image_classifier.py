from pathlib import Path
from unittest import skipUnless, TestCase

import numpy as np
import torch
import torchvision.models
from PIL import Image

from image_classifiers.image_classifier import ImageClassifier

DATA_DIR = Path(__file__).parent / "data"


class TestImageClassifier(TestCase):

    def setUp(self):
        self.weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.labels = np.array(self.weights.meta["categories"], dtype="U")

    def load_images(self):
        images = [Image.open(path) for path in sorted(DATA_DIR.glob("*.jpg"))]
        return images

    def assert_labels_equal(self, labels, expected_labels):
        self.assertEqual(len(labels), len(expected_labels))

        for label_set, expected_label_set in zip(labels, expected_labels):
            self.assertEqual(len(label_set), len(expected_label_set))

            for label, expected_label in zip(
                label_set, expected_label_set, strict=True
            ):
                self.assertEqual(label[0], expected_label[0])
                self.assertAlmostEqual(label[1], expected_label[1], places=4)

    def test_create(self):
        classifier = ImageClassifier(
            model=self.model,
            transforms=self.weights.transforms(),
            device="cpu",
            labels=self.labels,
        )

        self.assertIs(classifier.model, self.model)
        self.assertEqual(
            next(classifier.model.parameters()).device, torch.device("cpu")
        )

    def test_classify(self):
        classifier = ImageClassifier(
            model=self.model,
            transforms=self.weights.transforms(),
            device="cpu",
            labels=self.labels,
        )

        images = self.load_images()

        # test preprocessing
        tensor = classifier.preprocess(images)

        self.assertEqual(tensor.shape, (3, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

        # test classification
        probabilities = classifier.predict(tensor)

        self.assertEqual(probabilities.shape, (3, 1000))
        self.assertEqual(probabilities.dtype, torch.float32)

        # test likely labels
        with self.subTest(threshold="default"):
            labels = classifier.likely_labels(probabilities.numpy())

            expected_labels = [
                [
                    ("ballpoint", 0.31526673),
                    ("fountain pen", 0.12406595),
                    ("letter opener", 0.055778615),
                ],
                [("platypus", 0.2944924)],
                [
                    ("red fox", 0.31532815),
                    ("Arctic fox", 0.038479563),
                    ("grey fox", 0.037279706),
                    ("kit fox", 0.025280198),
                ],
            ]

            self.assert_labels_equal(labels, expected_labels)

        with self.subTest(threshold=0.1):
            labels = classifier.likely_labels(probabilities.numpy(), 0.1)

            expected_labels = [
                [("ballpoint", 0.31526673), ("fountain pen", 0.12406595)],
                [("platypus", 0.2944924)],
                [("red fox", 0.31532815)],
            ]

            self.assert_labels_equal(labels, expected_labels)

    @skipUnless(torch.mps.is_available(), "MPS not available")
    def test_classify_mps(self):
        classifier = ImageClassifier(
            model=self.model,
            transforms=self.weights.transforms(),
            device="cpu",
            labels=self.labels,
        )

        images = self.load_images()

        # test preprocessing
        tensor = classifier.preprocess(images)

        self.assertEqual(tensor.shape, (3, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

        # test classification
        probabilities = classifier.predict(tensor)

        self.assertEqual(probabilities.shape, (3, 1000))
        self.assertEqual(probabilities.dtype, torch.float32)

        # test likely labels
        with self.subTest(threshold="default"):
            labels = classifier.likely_labels(probabilities.numpy())

            expected_labels = [
                [
                    ("ballpoint", 0.31526673),
                    ("fountain pen", 0.12406595),
                    ("letter opener", 0.055778615),
                ],
                [("platypus", 0.2944924)],
                [
                    ("red fox", 0.31532815),
                    ("Arctic fox", 0.038479563),
                    ("grey fox", 0.037279706),
                    ("kit fox", 0.025280198),
                ],
            ]

            self.assert_labels_equal(labels, expected_labels)

        with self.subTest(threshold=0.1):
            labels = classifier.likely_labels(probabilities.numpy(), 0.1)

            expected_labels = [
                [("ballpoint", 0.31526673), ("fountain pen", 0.12406595)],
                [("platypus", 0.2944924)],
                [("red fox", 0.31532815)],
            ]

            self.assert_labels_equal(labels, expected_labels)

    @skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_classify_cuda(self):
        classifier = ImageClassifier(
            model=self.model,
            transforms=self.weights.transforms(),
            device="cuda",
            labels=self.labels,
        )

        images = self.load_images()

        # test preprocessing
        tensor = classifier.preprocess(images)

        self.assertEqual(tensor.shape, (3, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

        # test classification
        probabilities = classifier.predict(tensor)

        self.assertEqual(probabilities.shape, (3, 1000))
        self.assertEqual(probabilities.dtype, torch.float32)

        # test likely labels
        with self.subTest(threshold="default"):
            labels = classifier.likely_labels(probabilities.numpy())

            expected_labels = [
                [
                    ("ballpoint", 0.31526673),
                    ("fountain pen", 0.12406595),
                    ("letter opener", 0.055778615),
                ],
                [("platypus", 0.2944924)],
                [
                    ("red fox", 0.31532815),
                    ("Arctic fox", 0.038479563),
                    ("grey fox", 0.037279706),
                    ("kit fox", 0.025280198),
                ],
            ]

            self.assert_labels_equal(labels, expected_labels)

        with self.subTest(threshold=0.1):
            labels = classifier.likely_labels(probabilities.numpy(), 0.1)

            expected_labels = [
                [("ballpoint", 0.31526673), ("fountain pen", 0.12406595)],
                [("platypus", 0.2944924)],
                [("red fox", 0.31532815)],
            ]

            self.assert_labels_equal(labels, expected_labels)

    def test_predict_empty(self):
        classifier = ImageClassifier(
            model=self.model,
            transforms=self.weights.transforms(),
            device="cpu",
            labels=self.labels,
        )

        # test classification
        probabilities = classifier.predict(torch.zeros((0, 3, 224, 244)))

        self.assertEqual(probabilities.shape, (0, 1000))
        self.assertEqual(probabilities.dtype, torch.float32)

    def test_likely_labels_empty(self):
        classifier = ImageClassifier(
            model=self.model,
            transforms=self.weights.transforms(),
            device="cpu",
            labels=self.labels,
        )

        # test likely labels
        labels = classifier.likely_labels(np.zeros((0, 1000), dtype="float32"))

        self.assertEqual(labels, [])
