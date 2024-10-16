from __future__ import annotations
import logging

# Default logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Send output to the terminal
)

from .text_classifier import TextClassifier

__version__ = "0.2"

__all__ = ["TextClassifier"]