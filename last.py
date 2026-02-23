import os
import torch
import numpy as np
import cv2
import random
from sam2.build_sam import build_sam2_video_predictor

# --- Configuration ---
CHECKPOINT = r"./checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = r"configs/sam2.1/sam2.1_hiera_t.yaml" 
VIDEO_PATH = r"D:\AA HILCPS Projects\openvino-segment-anything-interactive-demo\world.mp4"
TEMP_FRAME_DIR = r"./temp_frames"

device = torch.device("cpu")
predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT, device=device)

# --- Frame Extraction ---
if not os.path.exists(TEMP_FRAME_DIR): os.makedirs(TEMP_FRAME_DIR)
cap = cv2.VideoCapture(VIDEO_PATH)

# Using index numbers to avoid AttributeError: 3=Width, 4=Height, 5=FPS
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)

if width == 0 or height == 0:
    print("Error: Could not read video properties. Check your video path.")
    exit()

frame_idx = 0
print(f"Extracting frames ({width}x{height} @ {fps}fps)...")
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(os.path.join(TEMP_FRAME_DIR, f"{frame_idx:05d}.jpg"), frame)
    frame_idx += 1
cap.release()

inference_state = predictor.init_state(video_path=TEMP_FRAME_DIR)

# --- Selection State ---
first_frame = cv2.imread(os.path.join(TEMP_FRAME_DIR, "00000.jpg"))
obj_metadata = {} # obj_id -> {"label": str, "is_base": bool}
current_points, current_labels = [], []
obj_count = 1

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        current_points.append([x, y])
        current_labels.append(1)
        cv2.drawMarker(first_frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

cv2.namedWindow("Selection")
cv2.setMouseCallback("Selection", on_mouse)

print("\n--- STEP 1: SELECT THE BASE ---")
print("Click on the BASE area. Press 'N' to confirm.")

while True:
    cv2.imshow("Selection", first_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('n') and current_points:
        if obj_count == 1:
            label = "BASE"
        else:
            # Simple terminal prompt for labels
            print(f"\a") # Beep
            label = input(f"Enter label for Object {obj_count}: ")
        
        predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_count,
            points=np.array(current_points, dtype=np.float32),
            labels=np.array(current_labels, dtype=np.int32),
        )
        
        obj_metadata[obj_count] = {"label": label, "is_base": (obj_count == 1)}
        print(f"Registered: {label}")
        
        obj_count += 1
        current_points, current_labels = [], []
        
        if obj_count == 2: 
            print("\n--- STEP 2: SELECT SUCCESSIVE OBJECTS ---")
            print("Select an object, press 'N' to label it. Press ENTER to start tracking.")
            
    elif key == 13: # Enter
        break

cv2.destroyAllWindows()

def get_bbox(mask):
    y, x = np.where(mask)
    if len(x) == 0: return None
    return (x.min(), y.min(), x.max(), y.max())

# --- Tracking & Coordinate Logic ---
out = cv2.VideoWriter("output_logic.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("\nProcessing video... (CPU Mode)")
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    frame = cv2.imread(os.path.join(TEMP_FRAME_DIR, f"{out_frame_idx:05d}.jpg"))
    
    frame_bboxes = {}
    active_labels = []

    # 1. First Pass: Locate the Base
    base_bbox = None
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        bbox = get_bbox(mask)
        frame_bboxes[obj_id] = bbox
        if obj_metadata[obj_id]["is_base"]:
            base_bbox = bbox

    # 2. Second Pass: Check Coordinates
    for obj_id, bbox in frame_bboxes.items():
        if obj_metadata[obj_id]["is_base"] or bbox is None: continue
        
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        
        if base_bbox:
            # Logic: Is the center of object inside Base's bounding box?
            if (base_bbox[0] < cx < base_bbox[2]) and (base_bbox[1] < cy < base_bbox[3]):
                active_labels.append(obj_metadata[obj_id]["label"])

        # Optional: Draw green box for objects
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # 3. Draw UI Sidebar
    sidebar_width = 350
    overlay = frame.copy()
    cv2.rectangle(overlay, (width - sidebar_width, 0), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, "OBJECTS IN BASE:", (width - sidebar_width + 10, 60), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    for i, label in enumerate(active_labels):
        y_pos = 120 + (i * 45)
        cv2.putText(frame, f"{i+1}. {label}", (width - sidebar_width + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

out.release()
cv2.destroyAllWindows()

print("\nSuccess! Video saved to output_logic.mp4")
