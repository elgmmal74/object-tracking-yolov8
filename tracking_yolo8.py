import cv2
from ultralytics import YOLO

# Load YOLOv8 model 
model = YOLO('yolov8m.pt')  

# Load video or webcam
cap = cv2.VideoCapture("test.mp4")  # Use 0 for webcam

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer for saving output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking (only class 2: car, and 3: motorcycle)
    results = model.track(source=frame, persist=True, classes=[2, 3], tracker="bytetrack.yaml")

    car_count = 0
    moto_count = 0

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())       # Class ID
            conf = float(box.conf[0].item())      # Confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            track_id = int(box.id[0]) if box.id is not None else -1  # Tracking ID

            if cls_id == 2:
                label = f"Car ID:{track_id}"
                car_count += 1
            elif cls_id == 3:
                label = f"Motorcycle ID:{track_id}"
                moto_count += 1
            else:
                continue

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Show vehicle count
    cv2.putText(frame, f"Cars: {car_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Motorcycles: {moto_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Save the frame to output video
    out.write(frame)

    # Show live preview
    cv2.imshow("Vehicle Tracking with YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
