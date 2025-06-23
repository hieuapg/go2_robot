import time
import cv2
import torch
import numpy as np
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# ----------------------------
# Preprocess and detect
# ----------------------------
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[..., ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img) / 255.0
    return torch.tensor(img).float().unsqueeze(0)

def detect_person(model, frame, conf_thresh=0.5):
    img = preprocess(frame)
    with torch.no_grad():
        preds = model(img)[0].cpu().numpy()
    boxes = []
    for *xyxy, conf, cls in preds:
        if conf > conf_thresh and int(cls) == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

# ----------------------------
# Main Camera Loop
# ----------------------------
def camera_loop(model):
    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()

    print("Camera and person detection running...")

    while True:
        code, data = video.GetImage()
        if code != 0:
            print("Camera error")
            continue

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        boxes = detect_person(model, frame)

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Go2 Camera - Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 detect_person_live.py [network_interface]")
        sys.exit(1)

    ChannelFactoryInitialize(0, sys.argv[1])

    model = torch.jit.load("best.torchscript.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    camera_loop(model)
