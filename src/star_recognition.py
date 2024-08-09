import cv2
import numpy as np
import tensorflow as tf


def recognize_stars(image):
    """
    Recognize stars in the provided image by detecting contours.

    Args:
        image (np.ndarray or tf.Tensor): Input image (could be RGB or grayscale).

    Returns:
        list: Detected contours in the image.
    """
    # Convert TensorFlow tensor to NumPy array if needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Check if the image has 3 channels (RGB)
    if image.ndim == 3 and image.shape[-1] == 3:
        # Convert RGB to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[-1] == 1:
        # Remove the single channel dimension
        gray_image = np.squeeze(image, axis=-1)
    elif image.ndim == 2:
        # The image is already grayscale
        gray_image = image
    else:
        raise ValueError("Unsupported number of channels in the input image.")

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def label_stars(image, contours):
    """
    Label detected stars on the provided image.

    Args:
        image (np.ndarray or tf.Tensor): Input image (should be grayscale for labeling).
        contours (list): List of detected contours.

    Returns:
        np.ndarray: Image with labeled stars.
    """
    # Convert TensorFlow tensor to NumPy array if needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Convert the image to BGR for labeling if it is grayscale
    if len(image.shape) == 2:
        labeled_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        labeled_image = image.copy()

    # Draw contours on the image
    for contour in contours:
        cv2.drawContours(labeled_image, [contour], -1, (0, 255, 0), 2)

    return labeled_image
