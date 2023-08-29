import cv2

#tracker = cv2.TrackerKCF_create() #velocidade padrão mas menos preciso
tracker = cv2.TrackerCSRT_create() #câmra lenta e mais preciso

video = cv2.VideoCapture("Videos/street.mp4")
ok, frame = video.read()

bbox = cv2.selectROI(frame) # região de interesse

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok: break

    ok, bbox = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for  v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Error", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Rastreamento", frame)
    if cv2.waitKey(1) & 0XFF == 27: #Esc
        break




