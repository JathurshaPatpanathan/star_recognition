import cv2
import numpy as np
import tensorflow as tf


def recognize_stars(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    if image.ndim == 3 and image.shape[-1] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[-1] == 1:
        gray_image = np.squeeze(image, axis=-1)
    elif image.ndim == 2:
        gray_image = image
    else:
        raise ValueError("Unsupported number of channels in the input image.")

    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def label_stars(image, contours):
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    if len(image.shape) == 2:
        labeled_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        labeled_image = image.copy()

    for contour in contours:
        cv2.drawContours(labeled_image, [contour], -1, (0, 255, 0), 2)

    return labeled_image
