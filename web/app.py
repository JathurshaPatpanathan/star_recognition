import sys
import os
from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
from io import BytesIO

# Ensure the src directory is included in the module search path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))

from star_recognition import recognize_stars, label_stars  # Import after adding src to the path

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        if file:
            # Read the image file as a NumPy array
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Recognize stars in the image
            contours = recognize_stars(image)

            # Label the stars in the image
            labeled_image = label_stars(image, contours)

            # Convert the labeled image to bytes for sending as a response
            _, buffer = cv2.imencode('.png', labeled_image)
            byte_io = BytesIO(buffer)

            return send_file(byte_io, mimetype='image/png')

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
