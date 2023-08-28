import cv2
from PIL import Image
import cv2

face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
face_recognition = cv2.face.LBPHFaceRecognizer_create()
# necessario treinar o lpbh para a face correspondente
face_recognition.read("lbph_classifier.yml")
height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
video_capture = cv2.VideoCapture(0)

while True:
    ok, frame = video_capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(150,150))
    for (x, y , w, h) in detections:
        image_face = cv2.resize(image_gray[y:y + w, x:x + h],(width, height)) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = face_recognition.predict(image_face)
        name = ""
        if id == 1:
            name = 'Gabriel'
        else:            
            name = 'Paulo' #o certo seria colocar o id correspondente a imagem do meu resto, por enquanto, vou usar esse pequeno truque

        cv2.putText(frame, name, (x, y + (w + 30)), font, 2, (0,0,255))
        cv2.putText(frame, str(confidence), (x, y + (h + 50)), font, 1, (0,0,255))


    
    cv2.imshow("vídeo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video_capture.release()
cv2.destroyAllWindows()
