import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

image_code = 'test2.jpg'

def predict_boxes():
    model = YOLO('runs/detect/train5/weights/best.pt')
    boxes = model(image_code)
    return boxes

def plot_image(results):
    yolov8_output_np = results.cpu().numpy()
    boxes = yolov8_output_np[:, :4]
    confidences = yolov8_output_np[:, 4]

    confidence_threshold = 0.5
    filtered_indices = confidences >= confidence_threshold
    filtered_boxes = boxes[filtered_indices]

    # Load the image
    image = cv2.imread(image_code)  # Replace with your image path
    image_height, image_width, _ = image.shape
    # Draw bounding boxes on the image
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # Green color for the bounding box (you can change this)
        thickness = 2
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # Display the image with bounding boxes using OpenCV
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


if __name__=="__main__":
    results = predict_boxes()
    print(results[0].boxes.data)

    plot_image(results[0].boxes.data)
    
    