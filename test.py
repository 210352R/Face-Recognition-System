from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from scipy.spatial import distance
import dlib


from win32com.client import Dispatch

# -------------------------------------------------------------------------------------------------------

# Eye Blink Logic


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


# ------------------------------------------------------------------------------------------------------


def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

with open("data/names.pkl", "rb") as w:
    LABELS = pickle.load(w)
with open("data/faces_data.pkl", "rb") as f:
    FACES = pickle.load(f)

print("Shape of Faces matrix --> ", FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


COL_NAMES = ["NAME", "TIME"]

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = frame[y : y + h, x : x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(
            frame,
            str(output[0]),
            (x, y - 15),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            1,
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        attendance = [str(output[0]), str(timestamp)]
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
    k = cv2.waitKey(1)
    if blink_count == 5:
        speak("Attendance Taken..")
        time.sleep(3)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
        blink_count = 0
    if k == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
