

import cv2
import os
import numpy as np
import faceRecognition as fr


#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread('/Users/harshit/Desktop/New folder/FaceRegspyder/TestImages/IMG_20170828_214432_660.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

#Comment belows lines when running this program second time.Since it saves training.yml file in directory
#faces,faceID=fr.labels_for_training_data('/Users/harshit/Desktop/New folder/FaceRegspyder/trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')

#Uncomment below line for subsequent runs
face_recognizer=cv2.face.createLBPHFaceRecognizer()
face_recognizer.load('/Users/harshit/Desktop/New folder/FaceRegspyder/trainingData.yml')


name={0:"Pratik",1:"Prea",2:"Harshit"}
for face in faces_detected: 
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label=face_recognizer.predict(roi_gray)
    confidence=face_recognizer.predict(roi_gray)
    #predicting the label of given image
    #print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    #if(confidence>37):#If confidence less than 37 then don't print predicted face text on screen
         #fr.put_text(test_img,"Unknown",x,y)#continue
    #fr.put_text(test_img,predicted_name,x,y)
    if(label==0 or label==2):#If confidence less than 37 then don't print predicted face text on screen
        fr.put_text(test_img,predicted_name,x,y)#continue
    else:
        fr.put_text(test_img,"Unknown",x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("Face Recognition Tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows