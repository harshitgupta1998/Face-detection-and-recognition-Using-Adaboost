# face detection and recongnition
Using LBPH and haar-adaboost, upload your own training images.

A computer program that decides whether an image is a positive image (face image) or negative image (non-face image) is called a classifier. A classifier is trained on hundreds of thousands of face and non-face images to learn how to classify a new image correctly. OpenCV provides us with two pre-trained and ready to be used for face detection classifiers:
Haar Classifier
LBP Classifier
Both of these classifiers process images in gray scales, basically because we don't need color information to decide if a picture has a face or no. As these are pre-trained in OpenCV, their learned knowledge files also come bundled with OpenCV.
Each file starts with the name of the classifier it belongs to. For example, a Haar cascade classifier starts off as haarcascade_frontalface_alt.xml.

These are the two types of classifiers we will be using to analyze Casper.

HAAR Classifier
The Haar Classifier is a machine learning based approach, an algorithm created by Paul Viola and Michael Jones; which are trained from many positive images (with faces) and negatives images (without faces).

Each window is placed on the picture to calculate a single feature. This feature is a single value obtained by subtracting the sum of pixels under the white part of the window from the sum of the pixels under the black part of the window.
Now, all possible sizes of each window are placed on all possible locations of each image to calculate plenty of features.
For example we are extracting two features. The first one focuses on the property that the region of the eyes is often darker than the area of the nose and cheeks. The second feature relies on the property that the eyes are darker than the bridge of the nose.
But among all these features calculated, most of them are irrelevant. For example, when used on the cheek, the windows become irrelevant because none of these areas are darker or lighter than other regions on the cheeks, all sectors here are the same.
So we promptly discard irrelevant features and keep only those relevant with a fancy technique called Adaboost. AdaBoost is a training process for face detection, which selects only those features known to improve the classification (face/non-face) accuracy of our classifier.
In the end, the algorithm considers the fact that generally: most of the region in an image is a non-face region. Considering this, it’s a better idea to have a simple method to check if a window is a non-face region, and if it's not, discard it right away and don’t process it again. So we can focus mostly on the area where a face is.

LBP Cascade Classifier
As any other classifier, the Local Binary Patterns, or LBP in short, also needs to be trained on hundreds of images. LBP is a visual/texture descriptor, and thankfully, our faces are also composed of micro visual patterns.
So, LBP features are extracted to form a feature vector that classifies a face from a non-face.
For each block, LBP looks at 9 pixels (3×3 window) at a time, and with a particular interest in the pixel located in the center of the window.
Then, it compares the central pixel value with every neighbor's pixel value under the 3×3 window. For each neighbor pixel that is greater than or equal to the center pixel, it sets its value to 1, and for the others, it sets them to 0.
After that, it reads the updated pixel values (which can be either 0 or 1) in a clockwise order and forms a binary number. Next, it converts the binary number into a decimal number, and that decimal number is the new value of the center pixel. We do this for every pixel in a block.
LBP conversion to binary. 

Source: López & Ruiz; Local Binary Patterns applied to Face Detection and Recognition

Then, it converts each block values into a histogram, so now we have gotten one histogram for each block in an image.
Finally, it concatenates these block histograms to form a one feature vector for one image, which contains all the features we are interested. So, this is how we extract LBP features from a picture.

HAAR VS. LBP. which is best for Face Detection?

Each OpenCV face detection classifier has its pros and cons, but the major differences are in accuracy and speed.
So, in case more accurate detections are required, Haar classifier is the way to go. This bad boy is more suitable in technology such as security systems or high-end stalking.
But the LBP classifier is faster, therefore, should be used in mobile applications or embedded systems.
