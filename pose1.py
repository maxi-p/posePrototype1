import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Holisitc model
mp_holistic = mp.solutions.holistic
# Drawing utils
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
	return image, results

def draw_landmarks(image, results):
	# mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
	mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
	mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh,rh])
    
cap = cv2.VideoCapture("vid.mp4")
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
	while cap.isOpened():
		# Read feed
		ret,frame = cap.read()
		
		# Make detections
		image,results = mediapipe_detection(frame,holistic)
		draw_landmarks(image,results)
		
		print(results.left_hand_landmarks)
		print(results)
		
		# Show to screen
		cv2.imshow('OpenCV Feed asd', image)
		
		# Break
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()