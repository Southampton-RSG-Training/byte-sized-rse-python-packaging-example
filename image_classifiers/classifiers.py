"""Routines for accessing Torchvision classifiers."""

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torchvision.models

from .image_classifier import ImageClassifier


@runtime_checkable
class ClassifierFactory(Protocol):
    """Protocol for  classifier factories.

    Can be implemented as a function or callable class.

    Parameters
    ----------
    device : str
        The Torch device to set on the ImageClassifier instance. Defaults to
        "cpu".

    Returns
    -------
    image_classifier : ImageClassifier
        The ImageClassifier instance created by the factory.
    """

    #: The qualified name (for both functions and class instances.)
    __qualname__: str

    def __call__(self, device: str = "cpu") -> ImageClassifier:
        raise NotImplementedError()


default_registry: dict[str, ClassifierFactory] = {}


@dataclass
class TorchvisionClassifierFactory(ClassifierFactory):
    """Factory class for Torchvision image classifiers.

    Parameters
    ----------
    device : str
        The Torch device to set on the ImageClassifier instance. Defaults to
        "cpu".

    Returns
    -------
    image_classifier : ImageClassifier
        An ImageClassifier built using the Torchvision classifier given by the
        model_name with its default weights and labels from the metadata. The
        model instance is cached so subsequent calls return the same object.
    """

    #: The name of the Torchvision model factory function
    model_name: str

    #: The image classifier instance, or None if it hasn't been created yet.
    classifier: ImageClassifier | None = None

    def __call__(self, device: str = "cpu"):
        if self.classifier is None:
            weights = torchvision.models.get_model_weights(self.model_name)[
                "DEFAULT"
            ]
            model = torchvision.models.get_model(
                self.model_name,
                weights=weights,
            )
            labels = np.array(weights.meta["categories"], dtype="U")
            self.classifier = ImageClassifier(
                model=model,
                transforms=weights.transforms(),
                device=device,
                labels=labels,
            )
        else:
            self.classifier.device = device
        return self.classifier

    @classmethod
    def register_all(
        cls,
        registry: MutableMapping[str, ClassifierFactory] = default_registry,
    ):
        model_names = torchvision.models.list_models(module=torchvision.models)
        for model_name in model_names:
            registry[model_name] = cls(model_name=model_name)


def get_classifier(
    classifier_name: str,
    device: str = "cpu",
    registry: Mapping[str, ClassifierFactory] = default_registry,
) -> ImageClassifier:
    """Get a classifier instance by name.

    Parameters
    ----------
    classifier_name : str
        The name of the classifier factory in te registry.
    device : str
        The Torch device to set on the ImageClassifier instance. Defaults to
        "cpu".
    registry : mapping of str to ClassifierFactory objects
        The registry to use when looking up the names. Uses the module default
        registry if not specified.

    Returns
    -------
    image_classifier : ImageClassifier
        The ImageClassifier instance created by the factory.
    """
    if classifier_name in registry:
        return registry[classifier_name](device)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name!r}")


def classifier_factory(
    classifier: str | ClassifierFactory,
    registry: MutableMapping[str, ClassifierFactory] = default_registry,
) -> Callable[[ClassifierFactory], ClassifierFactory] | ClassifierFactory:
    """Decorator for registering a callable as a classifier factory.

    This can be used either as::

        @classifier_factory("classifier-name", my_registry)
        def my_factory(device: str = "cpu"):
            ...

    or to derive the name from the function name and use the default registry::

        @classifier_factory
        def my_factory(device: str = "cpu"):
            ...
    """

    if isinstance(classifier, str):

        def classifier_decorator(classifier_factory: ClassifierFactory):
            registry[classifier] = classifier_factory
            return classifier_factory

        return classifier_decorator

    else:
        registry[classifier.__qualname__] = classifier
        return classifier


# Add all the Torchvision classifiers to the default registry
TorchvisionClassifierFactory.register_all()
