import cv2

face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Captura frame por frame
    ok, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(150,150))

    # Desenha o retangulo
    for (x, y , w, h) in detections:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

    #mostra o resultado do vídeo
    cv2.imshow("vídeo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Libera memória no final
video_capture.release()
cv2.destroyAllWindows()
