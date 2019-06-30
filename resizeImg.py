# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:15:59 2019

@author: harshit
"""

import cv2
import os
import numpy as np

count=0

for path, subdirnames, filenames in os.walk("/Users/harshit/Desktop/New folder/FaceRegspyder/trainingImages"):

    for filename in filenames:
      if filename.startswith("."):
        print("Skipping File:",filename)#Skipping files that startwith .
        continue
      img_path=os.path.join(path, filename)#fetching image path
      print("img_path",img_path)
      id=os.path.basename(path)#fetching subdirectory names
      img = cv2.imread(img_path)
      if img is None:
        print("Image not loaded properly")
        continue
      resized_image = cv2.resize(img, (100, 100))
      new_path="/Users/harshit/Desktop/New folder/FaceRegspyder/ResizedTrainingImages"+"/"+str(id)
      print("desired path is",os.path.join(new_path, "frame%d.jpg" % count))#write all images to resizedTrainingImages/id directory
      cv2.imwrite(os.path.join(new_path, "frame%d.jpg" % count),resized_image)
      count += 1