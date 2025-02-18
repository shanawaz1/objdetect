import cv2
import supervision as sv
from ultralytics import YOLOv10
import os
model=YOLOv10('C:/Users/shana/source/repos/face-attendance-system/best.pt')

bounding_box_annotator=sv.BoundingBoxAnnotator()
label_annotator=sv.LabelAnnotator()

cam=cv2.VideoCapture(0)

if not cam.isOpened():
    print('Unable to open the camera')
img_counter=0
while True:
    ret, frame = cam.read()
    if not ret:
        break
    Results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(Results)

    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    cv2.imshow('Webcam Feed', frame)
    k=cv2.waitKey(1)

    if k%256 == 27:
        print('Escape hit, closing...')
        break
cam.release()
cv2.destroyAllWindows()