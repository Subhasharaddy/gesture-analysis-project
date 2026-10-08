import cv2
import time
from utils.camera_manager import CameraManager
from utils.hand_tracker import UniversalHandTracker
from utils.landmark_processor import LandmarkProcessor
from utils.gesture_engine import GestureEngine
import config

def trace_pipeline():
    print("=" * 65)
    print("TRACING COMPLETE 42-GESTURE RECOGNITION PIPELINE")
    print("=" * 65)

    cam = CameraManager()
    if not cam.is_opened():
        print("[ERROR] Camera not opened.")
        return

    tracker = UniversalHandTracker()
    engine = GestureEngine(
        model_path=config.MODEL_PKL_PATH,
        label_encoder_path=config.LABEL_ENCODER_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        smoothing_window=config.STABILITY_WINDOW_SIZE
    )

    print(f"Model loaded: {engine.model is not None}")
    print(f"Label encoder loaded: {engine.label_encoder is not None}")
    if engine.label_encoder:
        print(f"Model classes ({len(engine.label_encoder.classes_)}): {engine.label_encoder.classes_[:5]}...")

    print(f"Confidence threshold: {engine.confidence_threshold}")
    print("\nReading 30 live camera frames...")

    hand_seen = 0
    for i in range(30):
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        hands = tracker.process(frame)
        if hands:
            hand_seen += 1
            h = hands[0]
            raw = LandmarkProcessor.extract_raw_landmarks(h)
            feat = LandmarkProcessor.extract_feature_vector(h)

            # Model prediction check
            probs = engine.model.predict_proba([feat])[0]
            best_idx = probs.argmax()
            conf = probs[best_idx]
            pred_name = engine.label_encoder.classes_[best_idx]

            # Engine process_frame check
            res = engine.process_frame(hands, image_shape=frame.shape[:2])
            confirmed = res.get("confirmed_gesture")
            status = res.get("status")

            print(f"Frame {i:2d}: Hand: YES | Lms: {len(h.landmark)} | Feat: {len(feat)} | RawPred: '{pred_name}' (Conf: {conf*100:.1f}%) | Confirmed: '{confirmed}' | Status: '{status}'")
        else:
            print(f"Frame {i:2d}: Hand: NO (No hand in camera view)")
        time.sleep(0.05)

    print(f"\nTotal frames with hand: {hand_seen}/30")
    cam.release()
    tracker.close()

if __name__ == "__main__":
    trace_pipeline()
