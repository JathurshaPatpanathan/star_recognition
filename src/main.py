import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_dataset
from image_processing import preprocess_image
from star_recognition import recognize_stars, label_stars

def main():
    dataset, _ = load_dataset()
    image, _ = next(iter(dataset))

    # Convert TensorFlow tensor to NumPy array if needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    print(f"Image shape: {image.shape}")

    star_label = recognize_stars(image)
    labeled_image = label_stars(image, star_label)

    if len(labeled_image.shape) == 4:
        labeled_image = np.squeeze(labeled_image)

    plt.imshow(labeled_image, cmap='gray')
    plt.title('Labeled Stars')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()