"""Capture webcam frames for a YOLO training dataset.

Press SPACE to save the current frame to disk.
Press Q to quit.

Frames are saved to data/captured/ as JPGs with a timestamp-based filename.
Run from the project root: `python scripts/capture.py`.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2


def open_webcam(index: int = 0):
    """Open the webcam, falling back to the DirectShow backend on Windows.

    Some Windows machines need cv2.CAP_DSHOW explicitly or VideoCapture(0)
    silently fails. Mac and Linux work with the default backend.
    """
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    raise RuntimeError(
        "Could not open webcam. On macOS, grant camera access to your terminal in "
        "System Settings > Privacy & Security > Camera. On Linux, ensure your user "
        "is in the 'video' group."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/captured",
                        help="Directory to save captured frames")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default 0)")
    parser.add_argument("--prefix", default="frame",
                        help="Filename prefix for saved frames")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_webcam(args.camera)

    print(f"Saving frames to {output_dir.resolve()}")
    print("SPACE to save, Q to quit.")

    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        # Show a copy so the saved file does not include the on-screen counter.
        display = frame.copy()
        cv2.putText(display, f"saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("capture (SPACE=save, Q=quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            timestamp = int(time.time() * 1000)
            path = output_dir / f"{args.prefix}_{timestamp}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"  [{saved}] {path.name}")
        elif key in (ord("q"), 27):  # Q or Escape
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} frames to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
