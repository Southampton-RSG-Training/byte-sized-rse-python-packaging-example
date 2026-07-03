Image Classifier Example
========================

This is an example project designed to support activities in the Byte-Sized RSE course on Python Packaging.

This code provides three things:

- a library for performing inference on images using the Torchvision models
- a command-line utility that reports most likely classes for image files
- a GUI application which classifies images

Installation
------------

To run the code as it currently stands, you will need to create a Python virtual environment and install the basic requirements::

    $ python3 -m venv venv
    $ source ./venv/bin/activity
    (venv) $ pip install -U pip
    (venv) $ pip install numpy pillow torch torchvision

If you want to run the command-line tool you should also install::

    (venv) $ pip install click rich rich-pixels

If you want to run the GUI you should also install::

    (venv) $ pip install toga

Usage
-----

To run the command-line tool within the activated virtual environment use::

    (venv) $ python -m image_classifiers.cli image_1.png image_2.png ...

To run the GUI app within the activated virtual environment use::

    (venv) $ python -m image_classifiers.gui

Activity
--------

The activity for this example is to package this project:

- write a ``pyproject.toml``
- add optional dependencies for the command-line tool, the GUI app, and for developers
- add script entry points for the command-line tool and the GUI app
- use ``pip -e .`` to create a development install in the virtual environment
- build a wheel and sdist for the project

Optionally:

- test uploading the package to ``test.pypi.org`` using Twine
- add a Github action to use ``cibuildwheel`` to automatically build wheels
- add briefcase configuration to the ``pyproject.toml`` for a CLI tool and the GUI app
- build native applications for the CLI tool and GUI app
