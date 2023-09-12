from ultralytics import YOLO



def train():
    model = YOLO('yolov8n.pt')
    # Train the model with 2 GPUs
    results = model.train(data='./train_conf.yaml', epochs=100, imgsz=640, device=[0])



if __name__=="__main__": 
    train()