from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('YOLOv8nNO.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

noteWidth = 0.3556
calibratingDistance = 1.1

d = 96.4869 # pixels

f = d * calibratingDistance/noteWidth

while True:
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)
    _, img = cap.read()
    img = img.conve
    # lower exposure
    
    # img = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
    try:
        results = model.predict(img, conf=0.5)

        for r in results:
            
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0] 
                c = box.cls
                annotator.box_label(b, model.names[int(c)], color=(0, 255, 0))
                width = box.xyxy[0][2] - box.xyxy[0][0]
                print(width, noteWidth)
                print( noteWidth * f / width)
            print("__________________________")
            
        img = annotator.result()  
        cv2.imshow('YOLO V8 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    except:
        print("hi")

cap.release()
cv2.destroyAllWindows()
