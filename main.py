from opcua_interface import PhoneDetectionOPCUAServer
import numpy as np
import cv2
from ultralytics import YOLO
import time

# Initialize
opc = PhoneDetectionOPCUAServer(endpoint="opc.tcp://0.0.0.0:4840")
opc.start()

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # (Optional) low-light enhancement / preprocessing here

        # Detect
        results = model(frame)[0]
        phone_found = False
        conf = 0.0
        bbox = (0, 0, 0, 0)
        for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            label = model.names[int(cls)]
            if label.lower() in ("cell phone", "mobile phone", "phone"):
                x1, y1, x2, y2 = map(int, box.tolist())
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                phone_found = True
                conf = float(score)
                bbox = (x1, y1, w, h)
                break

        # Crop and simulate temperature if phone found
        temp_crop = None
        hotspot = 0.0
        average = 0.0
        if phone_found:
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # downsample to shape the OPC UA server expects
            temp_crop_gray = cv2.resize(gray, (opc.phone_temp_array.get_array_dimensions()[1],
                                               opc.phone_temp_array.get_array_dimensions()[0]),
                                        interpolation=cv2.INTER_AREA)
            # map to simulated temperature
            temp_crop = 20.0 + (temp_crop_gray.astype(np.float32)/255.0) * (40.0 - 20.0)
            hotspot = float(np.max(temp_crop))
            average = float(np.mean(temp_crop))

        # Compute FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        # Update OPC UA
        opc.update_detection(
            detected=phone_found,
            confidence=conf,
            bbox=bbox,
            temp_crop=temp_crop,
            hotspot=hotspot,
            average=average,
            alarm_condition=phone_found,
            model_mode="Inference",
            frame_rate=fps
        )

        # Visualization for your debug (optional)
        if phone_found:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    opc.stop()