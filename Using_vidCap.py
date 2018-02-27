import numpy as np
import cv2

vidCap = cv2.VideoCapture(0) #number is to specify which camera(index)
path = "haarcascade_frontalface_default.xml"
path1 = "haarcascade_eye.xml"

while(True):
    #capture frame-by-frame
    ret, frame = vidCap.read()

    #operations on the frame starts here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Display the resulting frame
    #cv2.imshow('NORMAL', frame)
    #cv2.imshow('HSV', hsv)
    #cv2.imshow('GRAY', gray)
    
    face_cascade = cv2.CascadeClassifier(path)
    eye_cascade = cv2.CascadeClassifier(path1)
    faces_found = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30)        
        )
    for (x, y, w, h) in faces_found:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes_found = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor = 1.1)
        for (ex, ey, ew, eh) in eyes_found:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #press 'q' to quit
        break

#after everything is done, release the capture
vidCap.release()
cv2.destroyAllWindows()


