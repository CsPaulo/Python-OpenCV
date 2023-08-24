import cv2 #OpenCV


#carregamento da imagem
image = cv2.imread("Images/people1.jpg")

#image = cv2.resize(image, (1600, 900))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Imagem em escala de cinza',image_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(image_gray.shape)

#detecção de faces
face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
detections_face = face_detector.detectMultiScale(image_gray, scaleFactor= 1.3, minSize= (30,30))

for x, y, w, h in detections_face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)

#detecção de olhos
eye_detector = cv2.CascadeClassifier("Cascades/haarcascade_eye.xml")
detections_eye = eye_detector.detectMultiScale(image_gray, scaleFactor=1.09, minNeighbors=10, maxSize= (50, 50))

for x, y, w, h in detections_eye:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,255), 2)

cv2.imshow("Faces e olhos detectados", image)
cv2.waitKey(0)