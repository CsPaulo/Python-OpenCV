import cv2 #OpenCV

#Detecção de Carros
image = cv2.imread("Images/car.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

car_detector = cv2.CascadeClassifier("Cascades/cars.xml")
detections_car = car_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=5)

for x, y, w, h in detections_car:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,255), 2)

#cv2.imshow("Carros detectados", image)
#cv2.waitKey(0)

#Detecção de relógios
image = cv2.imread("Images/clock.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clock_detector = cv2.CascadeClassifier("Cascades/clocks.xml")
detections_clock = clock_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=1)

for x, y, w, h in detections_clock:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,255), 2)

#cv2.imshow("Carros detectados", image)
#cv2.waitKey(0)

#Detecção de Corpo Inteiro
image = cv2.imread("Images/people3.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fullbody_detector = cv2.CascadeClassifier("Cascades/fullbody.xml")
detections_fullbody = fullbody_detector.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5, minSize = (50, 50))

for x, y, w, h in detections_fullbody:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,255), 2)

cv2.imshow("Pessoas Interias detectadas", image)
cv2.waitKey(0)