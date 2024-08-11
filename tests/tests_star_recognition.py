import unittest
import numpy as np
import cv2
import sys
import os

# Ensure the src directory is included in the module search path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))

from star_recognition import recognize_stars, label_stars

class TestStarRecognition(unittest.TestCase):
    def setUp(self):
        # Create a simple black image with white dots to simulate stars
        self.image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.image, (30, 30), 3, 255, -1)
        cv2.circle(self.image, (70, 70), 3, 255, -1)

    def test_recognize_stars(self):
        contours = recognize_stars(self.image)
        self.assertEqual(len(contours), 2)

    def test_label_stars(self):
        contours = recognize_stars(self.image)
        labeled_image = label_stars(self.image, contours)
        self.assertEqual(labeled_image.shape, (100, 100, 3))  # Ensure the output image is in BGR format

if __name__ == '__main__':
    unittest.main()
