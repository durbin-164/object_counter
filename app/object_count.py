from collections import defaultdict
from math import sqrt

import cv2
from ultralytics import YOLO

from app.config import config
from videos import VIDEO_BASE_PATH


#####################################################################################
# Open the video file
video_path = f"{VIDEO_BASE_PATH}/road1.mp4"
videocapture = cv2.VideoCapture(video_path)
frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

# Output setup
save_dir = f"{VIDEO_BASE_PATH}/road1_out.mp4"
video_writer = cv2.VideoWriter(save_dir, fourcc, fps, (frame_width, frame_height))
save_img = True
######################################################################################

# count object that is pass the count line.
in_object = 0
out_object = 0

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

track_history = defaultdict(list)

# Loop through the video frames
while videocapture.isOpened():
    # Read a frame from the video
    success, frame = videocapture.read()

    # count object that is in and out of circle line.
    in_circle = 0
    out_circle = 0

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                label = str(cls)
                xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)

                # for visualize and track object direction.
                track = track_history[track_id]

                # if object is right side of count line then 0 otherwise 1.
                sign = 0 if (config.count_line.start_point[0] - x) < 0 else 1

                if len(track) > 1:
                    _, last_sigh = track[-1]
                    if last_sigh != sign:
                        if last_sigh == 0:
                            # object was right side of reference or count line
                            in_object += 1
                        else:
                            # object was left side of reference line
                            out_object += 1

                track.append(((int(x), int(y)), sign))

                if len(track) > 30:
                    track.pop(0)

                # annotate an object's last 30 moving position.
                for center, sign in track:
                    annotated_frame = cv2.circle(annotated_frame, center, config.dir_visualize.radius,
                                                 config.dir_visualize.color, config.dir_visualize.thickness)

                #########################################################################################
                # calculated distance of a object from count circle radius.
                # if distance is less than radius then object is in circle.
                distance = sqrt((x - config.count_circle.center[0]) ** 2 + (y - config.count_circle.center[1]) ** 2)
                if distance <= config.count_circle.radius:
                    in_circle += 1
                else:
                    out_circle += 1

        # draw the reference of count line.
        annotated_frame = cv2.line(annotated_frame, config.count_line.start_point,
                                   config.count_line.end_point, config.count_line.color, config.count_line.thickness)

        # draw the reference count circle.
        annotated_frame = cv2.circle(annotated_frame, config.count_circle.center, config.count_circle.radius,
                                     config.count_circle.color, config.count_circle.thickness)

        # draw the result text.
        in_text = (f'In Object: {in_object} || Out Object: {out_object} || In Circle: {in_circle} ||'
                   f' Out Circle: {out_circle}')
        annotated_frame = cv2.putText(annotated_frame, in_text, config.text.org,
                                      config.text.font, config.text.fontScale, config.text.color,
                                      config.text.thickness, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if save_img:
            video_writer.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
videocapture.release()
video_writer.release()
cv2.destroyAllWindows()
