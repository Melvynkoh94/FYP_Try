import numpy as np
import cv2, os
import sys
from PIL import Image

cascadePath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascadePath) #face classifier

recognizer = cv2.createBPHFaceRecognizer()

def get_images_and_labels(path):
# Append all the absolute image paths in a list image_paths  
# We will not read the image with the .sad extension in the training set 
# Rather, we will use them to test our accuracy of the training - See 
image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
# images will contains face images
images = []
# labels will contains the label that is assigned to the image
labels = []
for image_path in image_paths:
    
    # Read the image and convert to grayscale
    image_pil = Image.open(image_path).convert('L   # Convert the image format into numpy array
    image = np.array(image_pil, 'uint8')
    # Get the label of the image
    nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", "")) 
    # Detect the face in the image
    faces = faceCascade.detectMultiScale(image)                                           
    # If face is detected, append the face to images and the label to labels
    for (x, y, w, h) in faces:                                               
        images.append(image[y: y + h, x: x + w])
        labels.append(nbr)#sthash.wBCaaSGc.dpuf
    # return the images list and labels list
    return images, labels

cap = cv2.VideoCapture(0)
while(True):
                                               
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x, y, w, h) in faces:
                                               
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    face_file_name = "img.jpg"
    cv2.imwrite(face_file_name, roi_color)

src = cv2.LoadImage("img.jpg", cv2.CV_LOAD_IMAGE_COLOR)
src0 = cv2.LoadImage("img0.jpg", cv2.CV_LOAD_IMAGE_COLOR)

sc0= cv2.CompareHist(src, src0, cv2.CV_COMP_BHATTACHARYYA)
sc1= cv2.CompareHist(src, src1, cv2.CV_COMP_BHATTACHARYYA)                                               
                                              

if sc0==0.0:
                                               
                                               
            cv2.putText(image, 'Raghava', (x, y), cv2.FONT_ITALIC, 1, (200,255,155),2)
if sc1==0.0:
            cv2.putText(image, 'Abdul kalam Sir', (x, y), cv2.FONT_ITALIC, 1, (200,255,155),2)











                                               
