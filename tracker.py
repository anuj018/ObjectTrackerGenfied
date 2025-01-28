from ultralytics import YOLOv8
from DeepSort import DeepSortAlgo

class YOLOv8_DeepSort:
    def __init__(self):
        model = YOLOv8
        DeepSortAlgo = DeepSortAlgo
    
    def process(video_path):
        model = model.eval()
        result = model(video_path)
        return result