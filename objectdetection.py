import cv2
from ultralytics import YOLO
model = YOLO("yolov9c.pt")
with open("coco.names", 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
def draw_bounding_boxes(frame, detections, class_labels):
    try:
        for detection in detections[0]:
            box = detection.boxes.xyxy[0].numpy().astype(int)
            class_id = int(detection.boxes.cls[0])
            confidence = detection.boxes.conf[0]
            label = f"{class_labels[class_id]} : {confidence:.2f}"
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 128), 1)
            cv2.putText(frame, label, (x1+50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255),2)
    except IndexError:
        print("there is no object take place")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()
while True:
    ret, frame = cap.read()
    results = model(frame)
    draw_bounding_boxes(frame, results, class_labels)
    cv2.imshow("YOLOv8 Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()