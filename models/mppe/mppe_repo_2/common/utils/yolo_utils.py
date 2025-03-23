from ultralytics import YOLO
import os


def get_bbox_list(image_path):
    # Get path to the YOLOv8 model file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    model_path = os.path.join(project_root, "demo/models/yolov8l.pt")
    print(f"Using model from: {model_path}")

    model = YOLO(model_path)

    results = model(source=image_path, conf=0.5)

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    bbox_list = []

    # Iterate through the results
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box

        # Detected class is "person" (class 0)
        if int(cls) == 0:
            bbox_list.append([x1, y1, x2 - x1, y2 - y1])

    # If no people detected, return a default bounding box covering the whole image
    if not bbox_list:
        import cv2

        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        bbox_list.append([0, 0, w, h])

    return bbox_list
