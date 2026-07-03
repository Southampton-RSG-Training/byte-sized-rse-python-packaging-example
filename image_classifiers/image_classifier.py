"""The main ImageClassifier class"""

import bisect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ImageClassifier:
    """Model and labels for image classification."""

    #: A Torch model instance
    model: torch.nn.Module

    #: A transform that converts an image to a form suitable for the model
    transforms: Callable[[Any], torch.Tensor]

    #: An array of human-readable labels
    labels: np.typing.NDArray[np.str_]

    #: The device to use when performing computations
    device: str

    #: If the classifier returns logits then softmax is needed to get probabilities.
    needs_softmax: bool = True

    def __setattr__(self, name, value):
        # when setting the device, update the model device
        if name == "device" and hasattr(self, "model"):
            self.model.to(value)
        if name == "model" and hasattr(self, "device"):
            value.to(self.device)
        super().__setattr__(name, value)

    def preprocess(self, images: Iterable[Any]) -> torch.Tensor:
        """Preprocess a collection of images to be suitable for the model.

        Parameters
        ----------
        images : iterable of images
            The images to preprocess, usually as Pillow Image instances.

        Returns
        -------
        image_tensor : torch.Tensor
            An (N, 3, W, H) Torch tensor containing the processed image data.
        """
        images = [self.transforms(image) for image in images]
        tensor = torch.empty(
            (len(images), *images[0].shape),
            device=self.device,
        )
        for i, image in enumerate(images):
            tensor[i] = image
        return tensor

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predict the label probabilities for a collection of images.

        Parameters
        ----------
        images : torch.Tensor
            An (N, 3, W, H) Torch tensor containing the processed image data.

        Returns
        -------
        probabilities : torch.Tensor
            The of label probabilities.
        """
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            predictions = self.model.forward(images)
            if self.needs_softmax:
                predictions = torch.softmax(predictions, -1)
        return predictions

    def likely_labels(
        self,
        prediction: np.typing.NDArray[np.float32],
        threshold: float = 0.02,
    ) -> list[list[tuple[str, float]]]:
        """Find likely labels in a human readable form.

        Parameters
        ----------
        prediction : float array of shape N x M
            The predicted probability distribution for N images and M classes.
        threshold : float
            The minimum probability for which to report predictions.

        Returns
        -------
        likely_labels : list of list of (probability, label)
            The most likely labels for each image, provided as a list of tuples
            of probability and human-readable value.
        """
        likely_labels: list[list[tuple[str, float]]] = [
            [] for _ in range(len(prediction))
        ]
        for img_index, label_index in zip(*np.where(prediction > threshold)):
            image_data = likely_labels[img_index]
            bisect.insort(
                image_data,
                (self.labels[label_index], prediction[img_index, label_index]),
                key=lambda x: -x[1],
            )
        return likely_labels
