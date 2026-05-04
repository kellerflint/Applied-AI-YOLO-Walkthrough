"""Train a YOLO11n model on the prepared dataset.

Defaults are tuned for fast iteration on a typical laptop without a GPU:
- yolo11n.pt is the smallest YOLO11 variant.
- imgsz=320 cuts training time roughly 4x compared to the default 640.
- batch=8 fits in most laptop RAM; bump it up if you have a GPU.

Run from the project root after running prepare_dataset.py:
    python scripts/train.py

Tweak augmentation hyperparameters by editing the
defaults below. Run `python scripts/train.py --help` for the full list.

Outputs land in runs/detect/<name>/ . The trained weights are at
runs/detect/<name>/weights/best.pt.
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Core training parameters.
    parser.add_argument("--data", default="data/dataset/dataset.yaml",
                        help="Path to dataset.yaml (created by prepare_dataset.py)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base model to fine-tune (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Training image size (320 is fast, 640 is the default)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size per training step")
    parser.add_argument("--name", default="run1",
                        help="Run name. Outputs land in runs/detect/<name>/")
    parser.add_argument("--device", default=None,
                        help="Force a device (e.g. 'cpu', 'mps', '0' for GPU 0). "
                             "Default lets Ultralytics auto-detect.")

    # Augmentation hyperparameters. These match Ultralytics defaults but are
    # exposed here so you can tune them once you have looked at the output of
    # visualize_augmentations.py and decided which ones make sense for your
    # specific object.
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="Disable mosaic for the last N epochs (helps final accuracy)")

    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
    )


if __name__ == "__main__":
    main()
