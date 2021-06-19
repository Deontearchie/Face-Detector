import cv2

#load some pre-trained data on face frontals from opencv (haar cascade algorithms)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose a img you want to detect
img = cv2.imread('Dwowface2.jpg')

#covert photo to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect face
face_coordinate = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangle around the faces
(x,y,w,h) = face_coordinate[0]
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)


cv2.imshow('D face detector',img)
cv2.waitKey()

print("code complete")