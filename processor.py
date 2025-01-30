import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from yolov4_deepsort.deep_sort import preprocessing, nn_matching
from yolov4_deepsort.deep_sort.detection import Detection
from yolov4_deepsort.deep_sort.tracker import Tracker
from yolov4_deepsort.tools import generate_detections as gdet
from ultralytics import YOLO

class YOLOv8_DeepSort:
    def __init__(self, info_flag=True):
        # Initialize YOLOv8 model
        self.yolo_model = YOLO("yolo11l.pt")
        
        # Initialize DeepSORT
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 0.8
        self.model_filename = 'yolov4_deepsort/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)
        
        # Flag for detailed information
        self.info_flag = info_flag

    def process(self, video_path: str):
        """
        Process the video frame by frame and return detections for each frame.
        """
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise ValueError("Unable to open video file")

        frame_num = 0
        all_frames_data = []

        while True:
            ret, frame = vid.read()
            if not ret:
                break

            frame_num += 1
            print(f"\nProcessing Frame #{frame_num}")

            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Define the inner box dimensions
            frame_height, frame_width, _ = frame.shape
            shrink_percentage = 0.10
            x1 = int(frame_width * shrink_percentage)
            y1 = int(frame_height * shrink_percentage)
            x2 = int(frame_width * (1 - shrink_percentage))
            y2 = int(frame_height * (1 - shrink_percentage))

            # Draw the inner box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the frame to the inner box for processing
            cropped_frame = frame_rgb[y1:y2, x1:x2]

            # Run YOLO inference on the cropped frame
            yolo_results = self.yolo_model(cropped_frame, stream=True)

            detections = []
            for result in yolo_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i, class_id in enumerate(class_ids):
                    class_name = self.yolo_model.names[class_id]
                    if class_name == "person" and scores[i] >= 0.55:
                        bbox = boxes[i]
                        # Adjust coordinates back to full frame space
                        tlwh_bbox = [bbox[0] + x1, bbox[1] + y1, bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        confidence = scores[i]
                        detections.append((tlwh_bbox, confidence, class_name))

            features = self.encoder(frame, [d[0] for d in detections])
            detections = [Detection(bbox, score, class_name, feature) for (bbox, score, class_name), feature in zip(detections, features)]

            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            self.tracker.predict()
            self.tracker.update(detections)

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            active_tracks = []
            entity_coordinates = []

            # Print frame header for detailed info
            if self.info_flag:
                print(f"\nFrame {frame_num} Tracking Details:")

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = (bbox[1] + bbox[3]) / 2

                if (x1 <= bbox_center_x <= x2 and y1 <= bbox_center_y <= y2):
                    # Print detailed tracking information
                    if self.info_flag:
                        print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                            str(track.track_id), track.get_class(), 
                            (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), 
                                (int(bbox[0])+(len(track.get_class())+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, f"{track.get_class()}-{track.track_id}",
                              (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

                    active_tracks.append(track.track_id)
                    entity_coordinates.append({
                        "person_id": str(track.track_id),
                        "x_coord": int(bbox_center_x),
                        "y_coord": int(bbox_center_y)
                    })

            # Print total number of people in the frame
            if self.info_flag:
                print(f"Total people in frame: {len(active_tracks)}")
                if active_tracks:
                    print(f"Active tracker IDs: {active_tracks}")

            frame_data = {
                "camera_id": 0,
                "image_url": "",
                "is_organised": True,
                "no_of_people": len(active_tracks),
                "current_time": datetime.utcnow().isoformat(),
                "ENTITY_COORDINATES": entity_coordinates
            }
            
            all_frames_data.append(frame_data)

            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Frame {frame_num} - People Count: {len(active_tracks)}")
            plt.show()
            plt.close()

        vid.release()
        self.tracker.save_global_database()

        output_file = os.path.join(os.getcwd(), "tracking_results.json")
        with open(output_file, "w") as f:
            json.dump(all_frames_data, f, indent=4)

        return all_frames_data


def process_video(file_path: str, info_flag=True):
    """
    Process the video and extract required information.
    """
    model = YOLOv8_DeepSort(info_flag=info_flag)
    results = model.process(file_path)
    return results


if __name__ == "__main__":
    video_path = "/home/azureuser/workspace/Genfied/input_videos/cropped_test.mp4"
    results = process_video(video_path, info_flag=True)
    print("Processing complete. Results saved in 'tracking_results.json'.")