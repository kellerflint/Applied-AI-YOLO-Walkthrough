"""Record webcam video clips to mp4 files.

Press R to start a recording, R again to stop and save it. Press Q
to quit (any active recording is saved on the way out).

Saved clips go to data/videos/ as mp4 files named clip_<timestamp>.mp4.
Run from the project root: `python scripts/record.py`.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2


def open_webcam(index: int = 0):
    """Open the webcam, falling back to the DirectShow backend on Windows."""
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
    parser.add_argument("--output-dir", default="data/videos",
                        help="Directory to save recorded clips")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default 0)")
    parser.add_argument("--prefix", default="clip",
                        help="Filename prefix for saved clips")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the camera's reported FPS (default: auto, fallback 30)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_webcam(args.camera)

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {width}x{height} @ {fps:.1f}fps")
    print(f"Saving clips to {output_dir.resolve()}")
    print("R to start/stop recording, Q to quit.")

    writer = None
    frame_count = 0
    saved_clips = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        # Build a display copy so we don't burn the REC indicator into the saved video.
        display = frame.copy()
        if writer is not None:
            cv2.circle(display, (20, 25), 8, (0, 0, 255), -1)
            cv2.putText(display, f"REC  {frame_count} frames", (40, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(frame)
            frame_count += 1
        else:
            cv2.putText(display, "press R to record", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imshow("record (R=toggle, Q=quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            if writer is None:
                timestamp = int(time.time() * 1000)
                path = output_dir / f"{args.prefix}_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    print("Could not open VideoWriter. Check the codec and output path.")
                    writer = None
                    continue
                frame_count = 0
                print(f"Recording -> {path.name}")
            else:
                writer.release()
                writer = None
                saved_clips += 1
                print(f"  Saved ({frame_count} frames)")
        elif key in (ord("q"), 27):
            if writer is not None:
                writer.release()
                saved_clips += 1
                print(f"  Saved current recording ({frame_count} frames)")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved_clips} clip(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
