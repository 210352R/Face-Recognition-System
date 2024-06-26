import cv2
import dlib
import numpy as np
from scipy.spatial import distance


# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Constants for EAR threshold and consecutive frames
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 3

# Initialize counters
blink_count = 0
frame_counter = 0

# Load face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define indexes for left and right eye landmarks
LEFT_EYE_INDEXES = range(36, 42)
RIGHT_EYE_INDEXES = range(42, 48)


# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        left_eye = shape_np[LEFT_EYE_INDEXES]
        right_eye = shape_np[RIGHT_EYE_INDEXES]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= EAR_CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0

        # for x, y in left_eye:
        #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        # for x, y in right_eye:
        #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.putText(
        frame,
        "Blinks: {}".format(blink_count),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
