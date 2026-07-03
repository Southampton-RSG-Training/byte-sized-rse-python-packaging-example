from unittest import skipUnless, TestCase

import torch

from image_classifiers.util import get_devices


class TestUtil(TestCase):

    def test_get_devices(self):
        devices = get_devices()

        self.assertIn("cpu", devices)

    @skipUnless(torch.mps.is_available(), "MPS not available")
    def test_get_devices_mps(self):
        devices = get_devices()

        self.assertIn("mps", devices)

    @skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_get_devices_cuda(self):
        devices = get_devices()

        self.assertIn("cuda:0", devices)
