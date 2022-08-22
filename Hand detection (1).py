#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2             # computer vision
import mediapipe as mp # machine learning
import numpy as np     # numerical python

def get_label(index, hand_landmark, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:

            # Process results
            label = classification.classification[0].label

            #Extract Coordinates
            coordinates = tuple(np.multiply(
                np.array((hand_landmark.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmark.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
           # print(coordinates)
            output = label ,coordinates

    return output


# In[2]:


mp_drawing = mp.solutions.drawing_utils   # call function for Drawing the hands
mp_hands = mp.solutions.hands             # call function for give the  solution

cam = cv2.VideoCapture(0)  # start vedio but not on screen
#stored as variable called hands
with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_hands=10) as hands:

    while cam.isOpened():
        success, image = cam.read()

        """if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue"""

        image = cv2.flip(image, 1)  # flip the image

        # hand Detection code
        results = hands.process(image)  # detect only hands without drawing

        # draw the hand and land marks (x:horizontall,y:verticall,z:depth)
        if results.multi_hand_landmarks:      #(if exist hands and landmarks)
            for num,hand_landmarks in enumerate(results.multi_hand_landmarks):   #(for loop in the hand that detected)
                mp_drawing.draw_landmarks(       #draw the dimensions and landmarks
                     image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                     mp_drawing.DrawingSpec(color=(0, 0, 250), thickness=5, circle_radius=3), #hand marks (bgr)
                     mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=2))   #lines      (bgr)

                if get_label(num,hand_landmarks,results):
                    label,coordinates = get_label(num,hand_landmarks,results)
                    cv2.putText(image,label,coordinates,cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,250),2,cv2.LINE_AA)

        cv2.imshow('Hand Detection', image)
        if cv2.waitKey(5) == 27:
            break

#print(results.multi_hand_landmarks)
#print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
#print(results.multi_hand_landmarks[1])
#print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
#print(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST])
#print(results.multi_handedness)
#print(results.multi_handedness[0].classification[0].index)

