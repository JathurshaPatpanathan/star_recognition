from star_recognition import recognize_stars, label_stars
from image_processing import preprocess_image
from data_loader import load_dataset
import sys
import os
import matplotlib.pyplot as plt

# Ensure the src directory is included in the module search path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))


def main():
    # Load the dataset and info
    dataset, _ = load_dataset()  # Dataset is the first element

    # Get an image from the dataset
    image, _ = next(iter(dataset))

    # Check image shape
    print(f"Image shape: {image.shape}")

    # Recognize stars in the image
    star_label = recognize_stars(image)  # Adjust as needed

    # Label the stars in the image
    labeled_image = label_stars(image, star_label)  # Adjust as needed

    # Check the shape of labeled_image and reshape if needed
    if len(labeled_image.shape) == 4:
        labeled_image = labeled_image.squeeze()

    # Display the image using matplotlib
    plt.imshow(labeled_image, cmap='gray')
    plt.title('Labeled Stars')
    plt.axis('off')  # Hide axes
    plt.show()


if __name__ == "__main__":
    main()
