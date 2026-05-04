"""Run live webcam inference with the trained YOLO model.

Each frame is annotated with bounding boxes, the per-class count of objects
currently detected, and the cumulative number of seconds at least one object
of that class has been on screen since the script started.

Usage:
    python scripts/live_inference.py

Override defaults:
    python scripts/live_inference.py --weights path/to/weights.pt --conf 0.4

Press Q (or Escape) to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_webcam(index: int = 0):
    """Open the webcam, falling back to DSHOW on Windows."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    raise RuntimeError(
        "Could not open webcam. On macOS, grant camera access to your terminal "
        "in System Settings > Privacy & Security > Camera."
    )


def draw_overlay(
    frame,
    class_counts: dict[str, int],
    on_screen_seconds: dict[str, float],
) -> None:
    """Write per-class count and cumulative time as text on the frame."""
    y = 30
    line_height = 28
    all_classes = sorted(set(list(class_counts) + list(on_screen_seconds)))
    if not all_classes:
        cv2.putText(frame, "no detections", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        return
    for cls_name in all_classes:
        count = class_counts.get(cls_name, 0)
        seconds = on_screen_seconds.get(cls_name, 0.0)
        text = f"{cls_name}: {count}  |  {seconds:.1f}s total"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y += line_height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/detect/run1/weights/best.pt",
                        help="Path to the trained .pt weights file")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Inference image size (match training imgsz for best results)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(
            f"Weights not found at {weights_path}. Train first with "
            "`python scripts/train.py`, then re-run this script."
        )

    print(f"Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    cap = open_webcam(args.camera)
    print("Press Q to quit.")

    on_screen_seconds: dict[str, float] = defaultdict(float)
    last_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_time
        last_time = now

        # Run inference. verbose=False keeps the terminal quiet.
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]
        annotated = result.plot()  # frame with boxes, labels, and confidences drawn

        # Tally what we saw this frame, then add dt to any class that had >= 1
        # detection. We count, not "is present at all", so a frame with three
        # markers contributes 1*dt to the marker timer (you only get to be on
        # screen once per frame), but the "now" counter reflects the count.
        class_counts: dict[str, int] = defaultdict(int)
        for cls_id in result.boxes.cls.cpu().numpy().astype(int):
            class_counts[model.names[int(cls_id)]] += 1
        for cls_name in class_counts:
            on_screen_seconds[cls_name] += dt

        draw_overlay(annotated, class_counts, on_screen_seconds)
        cv2.imshow("YOLO live inference (Q=quit)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if on_screen_seconds:
        print("\nFinal totals:")
        for cls_name, seconds in sorted(on_screen_seconds.items()):
            print(f"  {cls_name}: {seconds:.1f}s")


if __name__ == "__main__":
    main()
