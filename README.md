# Computer Vision Project: Face Recognition and Blink Detection

## Overview
This project demonstrates two key aspects of computer vision using Python and OpenCV:
1. **Face Data Collection**: Capturing facial images from a webcam, processing them, and storing the data for future use.
2. **Blink Detection**: Detecting blinks using Eye Aspect Ratio (EAR) calculated from facial landmarks.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Dlib (`dlib`)
- Numpy (`numpy`)
- Scipy (`scipy`)
- Streamlit (`streamlit`)
- Streamlit Autorefresh (`streamlit_autorefresh`)
- Pickle (`pickle`)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-repository.git
    cd your-repository
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python dlib numpy scipy streamlit streamlit-autorefresh
    ```

3. Ensure that the required models are downloaded and placed in the `models/` directory:
    - `haarcascade_frontalface_default.xml`
    - `shape_predictor_68_face_landmarks.dat`

## Project Structure

- `data/`: Directory to store the collected face data and names.
- `models/`: Directory containing the pre-trained models for face detection and facial landmarks.

## Face Data Collection

### Description
This script captures facial images from a webcam, processes them, and stores them for later use.

### Steps
1. The script prompts the user to enter their name.
2. The webcam captures frames, and the script detects faces using a Haar Cascade Classifier.
3. The detected faces are cropped, resized, and stored in the `data/` directory.
4. The script continues capturing images until it collects 100 face images.
5. The collected data is stored using the Pickle module for future use.

### Usage
Run the script:
```bash
python face_data_collection.py
