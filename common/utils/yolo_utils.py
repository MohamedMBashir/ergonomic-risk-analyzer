from ultralytics import YOLO

def get_bbox_list(image_path):
    model = YOLO("models/yolov8l.pt")

    results = model(source=image_path, show=True, conf=0.9)

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    bbox_list = []

    # Iterate through the results
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]
        bbox_list.append([x1, y1, x2-x1, y2-y1])

    return bbox_list

