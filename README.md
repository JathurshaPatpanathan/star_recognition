# star_recognition

# Star Recognition Using Machine Learning

## Project Overview
This project focuses on developing a machine learning pipeline to detect and label stars in astronomical images. The project progressed through three versions, each building on the last to improve accuracy, introduce new features, and integrate CI/CD for automated testing and deployment.

## Team Members
- Jathursha Patpanathan
- Suramanjari Kesavan
- Vilas Vetri
- Harshitha Krishna

## Key Technologies
- **TensorFlow**: Used for building and training the Convolutional Neural Network (CNN).
- **OpenCV**: Used for image preprocessing and star recognition.
- **Python**: Core programming language for developing the pipeline.
- **Flask**: Web framework for creating a user-friendly interface.
- **GitHub Actions**: CI/CD tool for automating testing and deployment.
- **Docker**: Containerization for deploying the application.

## Project Versions

### Version 1: Basic Implementation
- **Objective**: Initial testing with a simplified dataset (MNIST).
- **Features**:
  - Simple preprocessing using TensorFlow and OpenCV.
  - Basic star recognition through contour detection.
- **Challenges**:
  - Accuracy limited by the simplistic dataset.
  - Not representative of actual astronomical images.

### Version 2: Enhanced Model
- **Objective**: Transition to a more realistic dataset with improved accuracy.
- **Features**:
  - Introduced a real-world star dataset.
  - Developed a CNN model for more accurate star detection.
  - Enhanced preprocessing techniques for complex data.
- **Challenges**:
  - Increased complexity in data handling.
  - Longer training times due to the CNN architecture.

### Version 3: Optimization and CI/CD Integration
- **Objective**: Optimize the model, integrate CI/CD, and develop a web interface.
- **Features**:
  - Refined CNN model for better performance.
  - Implemented CI/CD pipeline using GitHub Actions.
  - Developed a Flask web interface for real-time star recognition.
  - Added monitoring for model accuracy and response time.
- **Achievements**:
  - Production-ready model with automated testing and deployment.
  - User-friendly web interface for image upload and analysis.
 
### Getting Started
**Prerequisites**
-Python 3.x
-Git

**Installation**
-**Clone the repository**
git clone https://github.com/JathurshaPatpanathan/star_recognition.git
cd star_recognition

-**Install dependecies**
pip install -r requirements.txt

-**Run the flask web app**
python web/app.py

-**Run test**
python -m unittest discover tests

### CI/CD Integration
GitHub Actions:
The project uses GitHub Actions for continuous integration and deployment.
Each commit triggers automated tests. If the tests pass, the model is deployed to production.
