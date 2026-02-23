import os
import torch
import numpy as np
import cv2
import random
from sam2.build_sam import build_sam2_video_predictor

# --- Configuration (CPU & Tiny Model) ---
CHECKPOINT = r"./checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = r"configs/sam2.1/sam2.1_hiera_t.yaml" 
VIDEO_PATH = r"D:\AA HILCPS Projects\openvino-segment-anything-interactive-demo\world.mp4"
TEMP_FRAME_DIR = r"./temp_frames"

# 1. Initialize SAM 2 Predictor for CPU
device = torch.device("cpu")
# We use float32 for CPU as bfloat16/float16 support varies on processors
predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT, device=device)

# 2. Extract Frames (Required for SAM 2 Video API)
if not os.path.exists(TEMP_FRAME_DIR):
    os.makedirs(TEMP_FRAME_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_idx = 0
print("Extracting frames to disk...")
while True:
    ret, frame = cap.read()
    if not ret: break
    # SAM 2 requires .jpg format
    cv2.imwrite(os.path.join(TEMP_FRAME_DIR, f"{frame_idx:05d}.jpg"), frame)
    frame_idx += 1
cap.release()

# 3. Initialize State
inference_state = predictor.init_state(video_path=TEMP_FRAME_DIR)

# 4. Interactive Selection (First Frame)
first_frame_path = os.path.join(TEMP_FRAME_DIR, "00000.jpg")
first_frame = cv2.imread(first_frame_path)
cv2.namedWindow("Select Object")

current_points = []
current_labels = []
obj_count = 1

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        current_points.append([x, y])
        current_labels.append(1) # Positive click
        cv2.drawMarker(first_frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

cv2.setMouseCallback("Select Object", on_mouse)

print("\n--- SELECTION MODE ---")
print("1. Click on an object.")
print("2. Press 'N' to confirm object and select another.")
print("3. Press 'ENTER' to start CPU tracking.")

while True:
    cv2.imshow("Select Object", first_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('n'):
        if current_points:
            # Register object in SAM 2 memory
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_count,
                points=np.array(current_points, dtype=np.float32),
                labels=np.array(current_labels, dtype=np.int32),
            )
            print(f"Object {obj_count} added.")
            obj_count += 1
            current_points, current_labels = [], []
            
    elif key == 13: # Enter
        break

cv2.destroyAllWindows()

# 5. CPU Propagation & Video Output
out = cv2.VideoWriter("output_cpu_sam2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("\nTracking on CPU... (This will take time)")
# propagate_in_video returns masks frame-by-frame
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    frame = cv2.imread(os.path.join(TEMP_FRAME_DIR, f"{out_frame_idx:05d}.jpg"))
    
    for i, obj_id in enumerate(out_obj_ids):
        # Threshold the logits to get a boolean mask
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        
        # Overlay a random color for each object
        color = np.array([random.randint(100, 255) for _ in range(3)], dtype=np.uint8)
        mask_overlay = np.zeros_like(frame)
        mask_overlay[mask] = color
        frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)
    
    out.write(frame)
    cv2.imshow("CPU Tracking", frame)
    cv2.waitKey(1)

out.release()

print("\nSuccess! Video saved to output_cpu_sam2.mp4")
