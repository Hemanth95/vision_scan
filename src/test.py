import cv2
import torch
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

# image_code = 'sample6.jpg'

def predict_boxes(image_code):
    model = YOLO('runs/detect/train5/weights/best.pt')
    boxes = model(image_code)
    return boxes

def is_barcode_detected(results):
    yolov8_output_np = results.cpu().numpy()
    boxes = yolov8_output_np[:, :4]
    if len(boxes) > 0:
        return True, boxes
    return False, None

def read_barcode(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_text = decode(gray_image)
    if decoded_text:
        barcode_data = decoded_text[0].data.decode("utf-8")
        return barcode_data

def crop_image(image, box):
    x1, y1, x2, y2 = box
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image

def read_barcodes():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        results = predict_boxes(frame)
        # plot_image(results[0].boxes.data, frame)
        barcode_detected, boxes = is_barcode_detected(results[0].boxes.data)
        if barcode_detected:
            for box in boxes:
                cropped_image = crop_image(frame, box)
                decoded_text = read_barcode(cropped_image)
                if decoded_text:
                    print("Decoded text", decoded_text)
                    cv2.imshow('Image with Bounding Boxes', cropped_image)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    read_barcodes()