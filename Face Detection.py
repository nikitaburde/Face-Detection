import cv2

#load the pretrained face detection model
faceDetection=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

#Read the input
image=cv2.imread('C:/Users/DELL/Python/people.jpg')
image=cv2.resize(image,(800,600))
image_gray=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

#Detect faces
detection=faceDetection.detectMultiScale(image_gray,scaleFactor=1.039,minNeighbors=7)

#Draw boxes around detected faces
for (x,y,w,h) in detection:
    cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0),2)

#Show the output
cv2.imshow('Face Detection',image)
cv2.waitKey(0)
cv2.destroyAllWindows()