# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:17:09 2019

@author: harshit
"""

import cv2
import os
import numpy as np

#Given an image below function returns rectangle for face detected alongwith gray scale image
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('/Users/harshit/Desktop/New folder/FaceRegspyder/HaarCascade/haarcascade_frontalface_default.xml')#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles

    return faces,gray_img

#Given a directory below function returns part of gray_img which is face alongwith its label/ID
def labels_for_training_data(directory):
    faces=[]
    faceID=[]
  
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue

            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
               continue 
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]#crop images 
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
    #face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer=cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

#Below function draws bounding boxes around detected face in image
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

#Below function writes name of person for detected label
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x-20,y-20),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),1)
