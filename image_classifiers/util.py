"""Utility routines."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_devices():
    """Utility method to get a list of available Torch devices."""
    devices = ["cpu"]
    try:
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    except Exception:
        # cuda not available
        logger.exception("Cuda not available")

    try:
        if torch.mps.is_available():
            devices.append("mps")
    except Exception:
        # cuda not available
        logger.exception("mps not available")

    return devices
