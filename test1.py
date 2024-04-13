import cv2
img = cv2.imread('Donald_Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# import face_recognition
face_cascade = cv2.CascadeClassifier('face_id.xml')
faces = face_cascade.detectMultiScale(gray)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()