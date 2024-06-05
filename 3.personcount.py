import cv2
from ultralytics import YOLO
model = YOLO("yolov9c.pt")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    dect=results[0]
    counter = 0
    for detection in dect:
        class_id = int(detection.boxes.cls[0])
        index_value=class_id.conjugate()
        if (index_value==0):
            counter+=1
    label1 = f"person count : {counter}"
    cv2.putText(frame,label1, (30,40), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0, 0, 0),2)
    cv2.imshow("Count people image", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()