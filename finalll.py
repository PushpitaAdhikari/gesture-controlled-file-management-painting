import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import pyttsx3  # For voice output
import os  # To open files
import pyautogui  # For mouse control

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = keras.models.load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 5)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

# Create a blank canvas for painting
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Presettings for drawing
drawColor = (0, 0, 255)
thickness = 20
tipIds = [4, 8, 12, 16, 20]
xp, yp = [0, 0]

# Define the file path to open when a gesture is detected
file_path = r"C:\Users\TGA\Music\plant\test\Apple___Black_rot\Apple___Black_rot (1).JPG" # Change this to your file

file_path1=r"C:\Users\TGA\Music\alzeimers\alzeimers\test\MildDemented\mildDem2.jpg"
# Main loop
with hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        className = ''
        img = frame.copy()  # Initialize img to be the frame

        if results.multi_hand_landmarks:
            landmarks = []
            for handslms in results.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * width)
                    lmy = int(lm.y * height)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

                # Speak the gesture name
                if className:
                    engine.say(className)
                    engine.runAndWait()

                # Perform actions based on gesture
                if className == 'peace':  # Example: Use 'peace' gesture to open a file
                    os.startfile(file_path)  # Opens a file using the system default program

                if className == 'call me':  # Example: Use 'call me' gesture to open a file
                    os.startfile(file_path1)  # Opens a file using the system default program
                    
                if className == 'fist':  # Example: Use 'fist' gesture to move the mouse
                    pyautogui.moveTo(lmx, lmy)  # Move mouse to the detected position
                
                if className == 'thumbs_up':  # Example: Use 'thumbs up' gesture to click
                    pyautogui.click()

            # Drawing functionality
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                x1, y1 = points[8]  # Index finger
                fingers = [(points[tipIds[0]][0] < points[tipIds[0] - 1][0])]  # Thumb
                for id in range(1, 5):
                    fingers.append(points[tipIds[id]][1] < points[tipIds[id] - 2][1])

                # Draw Mode
                if fingers[1] and not any(fingers[i] for i in range(2, 5)):
                    cv2.circle(frame, (x1, y1), int(thickness / 2), drawColor, -1)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                    xp, yp = x1, y1

                # Clear Canvas
                if all(fingers[i] == 0 for i in range(5)):
                    imgCanvas = np.zeros((height, width, 3), np.uint8)
                    xp, yp = [0, 0]

        # Show canvas on the frame
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(frame, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', img)

        if cv2.waitKey(1) == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
