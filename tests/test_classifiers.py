from unittest import skipUnless, TestCase

import numpy as np
import torch
import torchvision.models

from image_classifiers.classifiers import (
    ClassifierFactory,
    default_registry,
    get_classifier,
)
from image_classifiers.image_classifier import ImageClassifier


class TestClassifiers(TestCase):

    def setUp(self):
        self.weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.labels = np.array(self.weights.meta["categories"], dtype="U")

    def test_get_classifier(self):
        classifier = get_classifier("resnet50")

        self.assertIsInstance(classifier, ImageClassifier)
        self.assertEqual(type(classifier.model), type(self.model))
        self.assertEqual(classifier.device, "cpu")
        np.testing.assert_array_equal(classifier.labels, self.labels)
        self.assertEqual(next(classifier.model.parameters()).device.type, "cpu")

    @skipUnless(torch.mps.is_available(), "MPS not available")
    def test_get_classifier_mps(self):
        classifier = get_classifier("resnet50", device="mps")

        self.assertIsInstance(classifier, ImageClassifier)
        self.assertEqual(type(classifier.model), type(self.model))
        self.assertEqual(classifier.device, "mps")
        np.testing.assert_array_equal(classifier.labels, self.labels)
        self.assertEqual(next(classifier.model.parameters()).device.type, "mps")

    @skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_get_classifier_cuda(self):
        classifier = get_classifier("resnet50", device="cuda")

        self.assertIsInstance(classifier, ImageClassifier)
        self.assertEqual(type(classifier.model), type(self.model))
        self.assertEqual(classifier.device, "cuda")
        np.testing.assert_array_equal(classifier.labels, self.labels)
        self.assertEqual(next(classifier.model.parameters()).device.type, "cuda")

    def test_get_classifier_error(self):
        with self.assertRaises(ValueError):
            get_classifier("NO SUCH CLASSIFIER")

    def test_default_registry(self):
        self.assertIsInstance(default_registry, dict)
        self.assertNotEqual(default_registry, {})
        self.assertTrue(all(isinstance(key, str) for key in default_registry))
        self.assertTrue(
            all(
                isinstance(value, ClassifierFactory)
                for value in default_registry.values()
            )
        )
