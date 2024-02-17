from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('YOLOv8nNO.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, img = cap.read()
# lower exposure of img
    try:
        results = model.predict(img, conf=0.5)

        for r in results:
            
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0] 
                c = box.cls
                annotator.box_label(b, model.names[int(c)], color=(0, 255, 0))
            
        img = annotator.result()  
        cv2.imshow('YOLO V8 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    except:
        print("hi")

cap.release()
cv2.destroyAllWindows()

# from ultralytics import YOLO

# # Load a model
# model = YOLO('YOLOv6nNO.pt')  # load an official model

# # Predict with the model
# results = model(0)  # predict using webcam