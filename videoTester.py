# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:12:45 2019

@author: harshit
"""

import os
import cv2
import numpy as np
import faceRecognition as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.createLBPHFaceRecognizer()
face_recognizer.load('/Users/harshit/Desktop/New folder/FaceRegspyder/trainingData.yml')
name = {0:"Pratik",1:"Unknown",2:"Harshit"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=4)
    
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.waitKey(10)

    fr.put_text(test_img,"Press Q To Exit ",70,70)    
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        #label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        label=face_recognizer.predict(roi_gray)
        confidence=face_recognizer.predict(roi_gray)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]

        if(label==0):#If confidence less than 37 then don't print predicted face text on screen
             fr.put_text(test_img,predicted_name,x,y)
        #continue
        elif(label==2):
             fr.put_text(test_img,predicted_name,x,y)
        else:
            fr.put_text(test_img,"Unknown",x,y)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Face Recognition",resized_img)
    if cv2.waitKey(10) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows