"""Routines for rich-text display."""

from collections.abc import Iterable
from pathlib import Path

from PIL.Image import Image
from rich.console import Console
from rich.table import Table
from rich_pixels import Pixels


def image_thumbnail(image: Image, size: tuple[int, int] = (24, 24)) -> Pixels:
    """Create a low-resolution rich text thumbnail of an image.

    This is intended to give enough context to identify the image.

    Parameters
    ----------
    image : a Pillow Image
        The image to get the thumbnail of.
    size : (int, int) tuple
        The size of the thumbnail in characters.

    Returns
    -------
    pixels : Pixels
        A rich_pixels Pixels object containing the image thumbnail.
    """
    image = image.copy()
    image.thumbnail(size)
    return Pixels.from_image(image)


def likely_labels_table(
    likely_labels: Iterable[tuple[str, float]],
    image_id: str | None = None,
) -> Table:
    """Create a Rich Table for the most likely labels for an image.

    Parameters
    ----------
    likely_labels : list of (float, str)
        The list of probabilities and corresponding labels, in order.
    image_id : str or None
        An id for the image or None if no id.
    """
    table = Table(title=image_id)
    table.add_column("Probability", justify="right")
    table.add_column("Label", justify="left")

    for label, probability in likely_labels:
        table.add_row(f"{probability:.1%}", label)

    return table


def image_table(
    files: Iterable[Path],
    images: Iterable[Image],
    image_labels: Iterable[list[tuple[str, float]]],
) -> Table:
    """Create a vertical grid of images and labels.

    Parameters
    ----------
    files : iterable of Path instances
        The paths to the files in the table.
    images : iterable of Image instances
        The Pillow Image objects from the files.
    image_labels : iterable of lists of (str, float) tuples
        The list of labels and probabilities for each image.

    Returns
    -------
    grid : Table
        A rich Table with the images and labels.
    """
    grid = Table.grid(padding=1)
    grid.add_column(justify="center", max_width=24)
    grid.add_column(justify="left")
    for file, image, image_label_set in zip(files, images, image_labels):
        table = likely_labels_table(image_label_set, file.name)
        pixels = image_thumbnail(image)
        grid.add_row(pixels, table)
    return grid
