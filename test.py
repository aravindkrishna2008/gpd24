import cv2
import numpy as np
import tensorflow as tf

# Load SSD-MobileNetV2 model
model = tf.saved_model.load("/Users/archita/robtoics/gpd24/model.tflite")

# Load label names
with open("/Users/archita/robtoics/gpd24/labels.txt", "r") as file:
    labels = file.read().splitlines()

# Open camera (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
    frame_rgb = tf.expand_dims(frame_rgb, 0)

    # Inference
    detections = model(frame_rgb)

    # Parse detection results
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    # Set a confidence threshold for detections
    confidence_threshold = 0.5
    selected_indices = np.where(scores > confidence_threshold)

    # Draw bounding boxes on the frame
    for i in selected_indices[0]:
        box = boxes[i]
        class_name = labels[classes[i] - 1]  # Subtract 1 because COCO labels start from 1
        score = scores[i]

        y_min, x_min, y_max, x_max = box
        image_height, image_width, _ = frame.shape

        x_min = int(x_min * image_width)
        x_max = int(x_max * image_width)
        y_min = int(y_min * image_height)
        y_max = int(y_max * image_height)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name}: {score:.2f}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Display the result
    cv2.imshow("SSD-MobileNetV2 Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
