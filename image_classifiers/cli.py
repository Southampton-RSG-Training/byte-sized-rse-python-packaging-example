"""Command-line interface for image classifiers"""

import pathlib

import click
from PIL import Image
from rich.console import Console

from .classifiers import get_classifier, default_registry
from .text_io import image_table
from .util import get_devices

CLASSIFIERS = sorted(default_registry)
CLASSIFIER_TEXT = ", ".join(CLASSIFIERS)
DEVICES = sorted(get_devices())


@click.command(epilog="""Available classifiers:

""" + CLASSIFIER_TEXT)
@click.option(
    "--classifier-name",
    type=str,
    default="vit_l_32",
    help="The name of the classifier to use.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.02,
    help="The minimum label probability to report.",
)
@click.option(
    "--device",
    type=click.Choice(DEVICES),
    default="cpu",
    help="The Torch device to use.",
)
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def predict(
    classifier_name: str,
    threshold: float,
    device: str,
    files: tuple[pathlib.Path, ...],
):
    """Predict labels for FILES."""
    # Handle edge-cases of parameters
    if not files:
        # nothing to do
        return
    try:
        classifier = get_classifier(classifier_name, device=device)
    except ValueError:
        raise click.BadParameter(
            message=f"{classifier_name!r} is not one of {CLASSIFIER_TEXT}",
            param_hint="--classifier-name",
        )

    # Get predictions
    images = [Image.open(path) for path in files]
    image_tensor = classifier.preprocess(images)
    predictions = classifier.predict(image_tensor).cpu().numpy()
    image_labels = classifier.likely_labels(
        predictions,
        threshold=threshold,
    )

    # Output images and labels
    console = Console()
    grid = image_table(files, images, image_labels)
    console.print(grid)


if __name__ == "__main__":
    predict()
