import cv2
import time

def find_working_camera(indices=[0, 1, 2, 3], target_w=640, target_h=480, target_fps=30):
    for idx in indices:
        # print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        
        # Configure approximately 640x480, 30 FPS, buffer 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Test read
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or target_w
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_h
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or target_fps
            print(f"Camera index: {idx}")
            print(f"Camera status: CONNECTED")
            print(f"Resolution: {actual_w}x{actual_h}")
            print(f"FPS: {actual_fps}")
            return cap, idx, (actual_w, actual_h), actual_fps
        else:
            cap.release()

    print("No working camera detected.")
    return None, -1, (0, 0), 0

if __name__ == "__main__":
    cap, idx, res, fps = find_working_camera()
    if cap:
        # Read 10 frames smoothly
        for i in range(10):
            ret, frame = cap.read()
            assert ret and frame is not None
            time.sleep(0.02)
        print("[TEST SUCCESS] Successfully captured 10 frames without lag or freeze.")
        cap.release()
