import random
import argparse
import cv2
import numpy as np
import openvino as ov

# ==============================
# Utility & State
# ==============================

def postprocess_results(predicted_logits, predicted_iou):
    sorted_ids = np.argsort(-predicted_iou, axis=-1)
    predicted_logits = np.take_along_axis(predicted_logits, sorted_ids[..., None, None], axis=2)
    return predicted_logits[0, 0, 0, :, :] >= 0

class Mouse:
    click_pos = (0, 0)
    left_clicked = False
    right_clicked = False

    @staticmethod
    def event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            Mouse.click_pos = (x, y)
            Mouse.left_clicked = True
        elif event == cv2.EVENT_RBUTTONUP:
            Mouse.right_clicked = True

def mask_to_bbox(mask):
    """Converts a binary mask to a (int x, int y, int w, int h) bounding box."""
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0: return None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    # Ensure values are standard Python ints for OpenCV trackers
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

# ==============================
# MAIN
# ==============================

def main(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup SAM Model
    compiled_model = ov.compile_model("efficient-sam-vits.xml", "CPU")
    cv2.namedWindow("Selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Selection", Mouse.event)

    ret, first_frame = cap.read()
    if not ret: return

    final_bboxes = []
    current_mask_bool = None
    current_mask_vis = None
    confirmed_vis = np.zeros_like(first_frame)

    print("Selection: [L-Click] Segment | [N] Confirm | [R-Click] Undo | [Enter] Start Tracking")
    
    while True:
        display = cv2.addWeighted(first_frame, 1.0, confirmed_vis, 0.5, 0)
        if current_mask_vis is not None:
            display = cv2.addWeighted(display, 1.0, current_mask_vis, 0.7, 0)

        cv2.imshow("Selection", display)
        key = cv2.waitKey(1) & 0xFF

        if Mouse.left_clicked:
            Mouse.left_clicked = False
            inference_img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)/255.0
            pts = np.array([Mouse.click_pos], np.int32).reshape(1,1,1,2)
            res = compiled_model({"batched_images": inference_img[None,...], "batched_points": pts, "batched_point_labels": np.ones((1,1,1), dtype=np.int32)})
            
            current_mask_bool = postprocess_results(np.expand_dims(res[0][:,0,:,:,:], 1), np.expand_dims(res[1][:,0,:], 1))
            current_mask_vis = np.zeros_like(first_frame)
            current_mask_vis[current_mask_bool] = [0, 255, 0]

        if key == ord('n') and current_mask_bool is not None:
            bbox = mask_to_bbox(current_mask_bool)
            if bbox:
                final_bboxes.append(bbox)
                confirmed_vis[current_mask_bool] = [random.randint(100, 255) for _ in range(3)]
                current_mask_bool, current_mask_vis = None, None
                print(f"Object {len(final_bboxes)} Added.")

        if Mouse.right_clicked:
            Mouse.right_clicked = False
            if final_bboxes: 
                final_bboxes.pop()
                print("Last object removed.")

        if key == 13: break # Enter
        if key == 27: # Esc
            cap.release(); cv2.destroyAllWindows(); return

    cv2.destroyWindow("Selection")

    # ------------------------------
    # TRACKER INITIALIZATION
    # ------------------------------
    # Using a list of individual trackers instead of MultiTracker
    trackers = []
    for bbox in final_bboxes:
        # Try different creation methods depending on your OpenCV version
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            # For some versions that moved it to legacy but still have the attribute
            tracker = cv2.legacy.TrackerCSRT_create()
            
        tracker.init(first_frame, bbox)
        trackers.append(tracker)

    # Output Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (width, height))

    print(f"Tracking {len(trackers)} objects...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Obj {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done. Saved to output_tracking.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()
    main(args)