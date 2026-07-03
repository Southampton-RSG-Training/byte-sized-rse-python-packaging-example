"""GUI interface for image classifiers"""

from concurrent.futures import ThreadPoolExecutor

import toga
from toga.constants import COLUMN
from toga.sources import ListSource
from PIL.Image import Image

from .classifiers import get_classifier, default_registry
from .util import get_devices

IMAGE_FILE_TYPES = ["jpg", "jpeg", "png"]


class ImageClassificationApp(toga.App):

    def startup(self):
        self.executor = ThreadPoolExecutor()
        self.raw_image = None
        self.photo = toga.ImageView(image=toga.Image("resources/default.png"))

        self.classifier = get_classifier("resnet50")
        self.labels_source = ListSource(
            accessors=["label", "probability"],
            data=[("camera", "100%")],
        )

        open_image_cmd = toga.Command.standard(
            self,
            toga.Command.OPEN,
            action=self.open_image,
        )
        self.commands.add(open_image_cmd)

        camera_box = toga.Box(
            children=[
                toga.Box(
                    children=[
                        toga.Box(flex=1),
                        self.photo,
                        toga.Box(flex=1),
                    ]
                ),
                toga.Box(
                    children=[
                        # Take a fresh photo
                        toga.Button(
                            "Take Photo",
                            on_press=self.take_photo,
                            flex=1,
                            margin=5,
                        ),
                    ],
                ),
            ],
            direction=COLUMN,
            margin_bottom=20,
        )

        label_box = toga.Box(
            children=[
                toga.Box(
                    children=[
                        toga.Label("Classifier:"),
                        toga.Selection(
                            items=sorted(default_registry),
                            value="resnet50",
                            on_change=self.set_classifier,
                            flex=1,
                        ),
                    ],
                ),
                toga.Box(
                    children=[
                        toga.Label("Device:"),
                        toga.Selection(
                            items=sorted(get_devices()),
                            value="cpu",
                            on_change=self.set_device,
                            flex=1,
                        ),
                    ],
                ),
                toga.Table(
                    columns=["Label", "Probability"],
                    data=self.labels_source,
                    flex=1,
                ),
            ],
            direction=COLUMN,
            flex=1,
        )

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = toga.Box(
            children=[camera_box, label_box],
            flex=1,
        )
        self.main_window.show()

    async def take_photo(self, widget, **kwargs):
        """Use the camera hardware to take a photo."""
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()

            image = await self.camera.take_photo()
            if image is not None:
                await self.set_image(image)
        except NotImplementedError:
            await self.main_window.dialog(
                toga.InfoDialog(
                    "Oh no!",
                    "The Camera API is not implemented on this platform",
                )
            )
        except PermissionError:
            await self.main_window.dialog(
                toga.InfoDialog(
                    "Oh no!",
                    "You have not granted permission to take photos",
                )
            )

    async def open_image(self, command, **kwargs):
        """Open an image file from disk."""
        open_dialog = toga.OpenFileDialog(
            title="Open Image File",
            file_types=IMAGE_FILE_TYPES,
        )
        path = await self.main_window.dialog(open_dialog)
        if path is not None:
            image = toga.Image(path)
            await self.set_image(image)

    async def set_classifier(self, widget, **kwargs):
        """Set the classifier to use and update classifications."""
        classifier_name = widget.value
        # Do get classifier in background thread (may download weights)
        self.classifier = await self.loop.run_in_executor(
            self.executor,
            get_classifier,
            classifier_name,
            self.classifier.device,
        )
        if self.raw_image is not None:
            await self.classify()

    async def set_device(self, widget, **kwargs):
        """Set the Torch device to use for computation."""
        device = widget.value
        self.classifier.device = device

    async def set_image(self, image: toga.Image):
        """Update the image and classify it."""
        self.raw_image = image.as_format(Image)
        thumbnail = self.raw_image.copy()
        thumbnail.thumbnail((640, 480))
        self.photo.image = toga.Image(thumbnail)
        await self.classify()

    async def classify(self):
        """Classify the current image, updating label list."""
        # Do analysis in background thread
        tensor = await self.loop.run_in_executor(
            self.executor, self.classifier.preprocess, [self.raw_image]
        )
        probabilities = await self.loop.run_in_executor(
            self.executor,
            self.classifier.predict,
            tensor,
        )

        probabilities = probabilities.cpu().numpy()
        likely_labels = self.classifier.likely_labels(probabilities)[0]
        self.labels_source.clear()
        for label, probability in likely_labels:
            self.labels_source.append((label, f"{probability:.1%}"))


def main():
    return ImageClassificationApp(
        "Image Classifier",
        "uk.ac.soton.srsg_training",
    )


if __name__ == "__main__":
    main().main_loop()
