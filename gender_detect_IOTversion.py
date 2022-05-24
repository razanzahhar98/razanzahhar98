# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:54:33 2021

@author: Desktop
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import paho.mqtt.client as mqtt
import time

mqttBroker = "broker.mqttdashboard.com"
client = mqtt.Client("Dispenser System")
client.connect(mqttBroker)
client.subscribe("iotmfi/assignment/groupB")

LocationID = "BLOCK_A"
CameraID = "B1_01"

                    
# load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man','woman']

# loop through frames
while webcam.isOpened():
    
    LogTimestamp = time.time()

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (255,0,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        color = (0, 255, 0) if classes[idx] == 'woman' else (0, 0, 255) 
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (startX,startY), (endX,endY), color, 2)

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        
        GenderID = 1 if classes[idx] == 'woman' else 0
        
        message = "{\"LogTimestamp\":" + str(int(LogTimestamp)) + ", \"GenderID\":" + str(int(GenderID)) + ", \"LocationID\":\"" + LocationID + "\", \"CameraID\":\"" + CameraID + "\"}"
        print(message)
        client.publish("iotmfi/miniproject/groupB", message)
        print("Just published the Data")
        time.sleep(3)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()