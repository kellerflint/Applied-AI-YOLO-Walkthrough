# YOLO Walkthrough: Train Your Own Object Detector

A step-by-step pair-programming walkthrough for fine-tuning a YOLO11 model on an object you choose. By the end you will have a webcam app that draws bounding boxes around your object, counts how many it sees, and tracks how long it has been on screen.

Everything runs locally in a Python virtual environment or docker container.

## What you will build

By the end of this walkthrough you will have:

1. Your raw dataset. A folder of webcam images that contain the specific object(s) you picked.
2. Bounding-box labels for each of those images, drawn in Label Studio.
3. A single PDF showing what each YOLO data augmentation does to your image.
4. A trained YOLO11n model fine-tuned on your object.
5. A live webcam script that draws boxes, counts objects, and tracks on-screen time per class.

## Prerequisites

- A laptop with a webcam (Mac, Windows, or Linux all work).
- Python 3.9 or newer installed and on your PATH. Install python at https://www.python.org/downloads/
- Git (just for cloning this repo).
- Docker Desktop installed (used only for Label Studio, optional if you choose the pip install path for Label Studio instead).
- About 2 GB of free disk space (most of it is the YOLO weights and PyTorch).

## One-time setup

Pick an object you want to train the model to detect. Something specific.

### 1. Clone and enter the project folder

Clone this repo and open it in your editor.

### 2. Create a virtual environment

The point of a venv is to keep this project's dependencies separate from anything else on your machine, so installing or upgrading something here cannot break a different project.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You will see `(venv)` appear at the start of your prompt. That tells you the environment is active. Every terminal window you open later needs this same activate command before running scripts. Skipping the activate is the most common cause of "ModuleNotFoundError: No module named 'ultralytics'".

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This pulls in:
- `ultralytics` (the YOLO library, which also installs PyTorch)
- `opencv-python` (for webcam access and drawing on frames)
- `matplotlib` (for the augmentation PDF)

Expect this to take a few minutes the first time. PyTorch is a large download.

### 4. Verify the install

```bash
python -c "from ultralytics import YOLO; import cv2; print('ok')"
```

If that prints `ok` and exits cleanly, you are ready to go. If it errors, jump to the [Troubleshooting](#troubleshooting) section.

---

## Step 1: Capture training images

Run the capture script and it opens a webcam window:

```bash
python scripts/capture.py
```

The first time you run the script on macOS, the OS may prompt you to grant camera access to your terminal. Click Allow. You may need to restart your editor/terminal.

Press SPACE to save the current frame. Press Q (or Escape) to quit. Frames land in `data/captured/`.

**Aim for at least 30 images.** YOLO11 fine-tuning needs surprisingly few samples for a small set of classes. Variety matters more than quantity. Capture your object:

- From at least 5 different angles.
- At 3 different distances (close-up, mid, far).
- Under different lighting conditions if you can manage it.
- Against different backgrounds (move it around the room).
- With different positions in the frame (not always centered).
- A handful of "negative" frames with no object visible can also help.

The more your training set looks like the conditions you will use the model in, the better it will perform. If you only ever capture the object in the center of a clean white background, the model will struggle when the background is messy or the object is at the edge.

---

## Step 2: Label the images in Label Studio

Label Studio is a free, open-source labeling tool. You draw bounding boxes around your object in each image, give them a class name, and export the labels in YOLO format.

### Run Label Studio

You have two options. Pick one. Both are documented at <https://labelstud.io/guide/quick_start>.

**Option A: Docker (recommended).** Single command, no Python install conflicts:

```bash
docker run -it -p 8080:8080 -v $(pwd)/data/labelstudio:/label-studio/data heartexlabs/label-studio:latest
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`. On Windows cmd.exe, replace it with `%cd%`.

**Option B: pip install.** This installs Label Studio into your Python venv:

```bash
pip install -U label-studio
label-studio start
```

Both options serve a web UI at <http://localhost:8080>. Create an account the first time (it's local, the email and password do not leave your machine).

### Create a project and import images

1. Click "Sign Up" (it's all local, so put whatever you like for the email/password)
2. Click "Create" to start a new project.
3. Give it a name like "yolo-walkthrough".
4. On the "Data Import" tab, drag in all the JPGs from `data/captured/`.
5. On the "Labeling Setup" tab, choose the "Computer Vision > Object Detection with Bounding Boxes" template.
6. Edit the labels in the template to match your classes. If you only have one class (e.g. "marker") delete the placeholder labels and add yours.
7. Click "Save".

### Draw bounding boxes

Click each task to open it. Press the keyboard shortcut for your label, then draw a tight box around the object. Press the submit button and go to the next image. Spend roughly 5 to 10 seconds per image.

If the image does not have the object in it, just click submit. YOLO can use these as negative samples.

### Export the labels

1. From the project page, click "Export".
2. Pick "YOLO with Images" as the format (not YOLOv8 OBB or any other variant).
3. Download the zip file and extract it to the 'data' folder in this project. The extracted folder will look like:
   ```
   project-1-at-2026-05-04/
       classes.txt
       images/
       labels/
       notes.json
   ```

---

## Step 3: Prepare the dataset

YOLO expects images and labels split into `train/` and `val/` folders, with a `dataset.yaml` describing the paths and class names. The export from Label Studio is not split, so we can run a script to do it.

```bash
python scripts/prepare_dataset.py --export-dir path/to/project-1-at-2026-05-04
```

Replace the path with wherever you extracted your Label Studio export. The script:

- Shuffles your images with a fixed seed (42, so re-running gives the same split).
- Copies 80% to `data/dataset/images/train/` and the matching labels to `data/dataset/labels/train/`.
- Copies the remaining 20% to the `val/` folders.
- Writes `data/dataset/dataset.yaml` with the right paths and your class names pulled from `classes.txt`.

You can change the split ratio with `--val-fraction 0.15` or pick a different seed with `--seed 123`.

If you re-label more images later and re-export, re-run this script. It wipes and rewrites `data/dataset/` each time so the split stays consistent.

---

## Step 4: Visualize the augmentations

Before you train, run the augmentation visualizer. This produces a single PDF showing what each YOLO augmentation does to one of your captured images:

```bash
python scripts/visualize_augmentations.py
```

You get `augmentations.pdf` in the project root. Open it. You will see a grid with one panel per augmentation:

- **hsv_h, hsv_s, hsv_v**: hue, saturation, brightness shifts.
- **degrees**: rotation around the center.
- **translate**: shift the image around within the frame.
- **scale**: zoom in or out.
- **shear**: skew the image diagonally.
- **perspective**: warp as if viewing from a different angle.
- **flipud**: flip vertically (upside down).
- **fliplr**: flip horizontally (mirror).
- **mosaic**: combine 4 different training images into one.
- **mixup**: blend two training images on top of each other.

The values in the visualizer are exaggerated so the effect is visible. Real training applies these as random magnitudes within bounds, so a value like `hsv_h=0.015` produces subtle variation each step rather than the strong shift you see in the PDF.

**Use this PDF to decide which augmentations to enable for your object.** The defaults in `train.py` are reasonable starting points but think about:

- Does flipping make sense? A mirrored marker is still a marker, but mirrored letters or numbers are garbage.
- Does rotation make sense?
- Mosaic is on by default and can be very powerful. It teaches the model to detect partially visible objects at the edges of frames. However in some cases you may not want to trigger detection unless the entire object is visible. It always depends on your usecase!

---

## Step 5: Train the model

```bash
python scripts/train.py
```

The defaults are:
- `yolo11n.pt` (nano model, smallest and fastest).
- 50 epochs.
- Image size 320 (down from the default 640 to speed up CPU training).
- Batch size 8.
- The augmentation hyperparameters from the previous step.

The first time you run this, Ultralytics downloads `yolo11n.pt` (about 6 MB).

You will see a table print every epoch with:
- `box_loss`, `cls_loss`, `dfl_loss`: training losses. They should decrease.
- `mAP50`, `mAP50-95`: validation accuracy metrics. They should increase. Higher is better. mAP50 above 0.8 on a small custom dataset is a good sign.

If training is too slow, try `--imgsz 256 --epochs 30` to cut the work roughly in half. Each epoch shouldn't take more than a few seconds to complete.

When training finishes, the trained weights are at `runs/detect/run1/weights/best.pt`. The training plots, confusion matrix, and sample predictions land in `runs/detect/run1/`.

### Looking at the results

Open `runs/detect/run1/results.png` to see the loss and accuracy curves over training. If `box_loss` is still decreasing at epoch 50, you probably want to train for more epochs. If `mAP50` plateaued by epoch 30, you can shorten next time.

Open `runs/detect/run1/val_batch0_labels.jpg` to see the model's predictions on your validation images.

---

## Step 6: Run live inference

Plug the trained model into a live webcam:

```bash
python scripts/live_inference.py
```

By default it loads `runs/detect/run1/weights/best.pt`. If you used a different run name, point it at the right path:

```bash
python scripts/live_inference.py --weights runs/detect/run2/weights/best.pt
```

A window opens showing your webcam with bounding boxes drawn on detected objects. Above the boxes, an overlay shows:

- The current count of each class in the frame.
- The cumulative seconds at least one of that class has been on screen since the script started.

Press Q to quit. The script prints final on-screen totals to the terminal.

A confidence of 0.5 means "only show detections the model is at least 50% sure about". Lowering this reveals weak detections, which can help you understand whether the model is failing because it's confused (sees something but at low confidence) or because it has no idea (no detections at any threshold).

You can test different confidence thresholds with the 'conf' flag.

```bash
python scripts/live_inference.py --conf 0.25
```

## Discussion Questions

With your partner, answer the following:
Cover roughly these topics, in this order:
1. Why did we split the dataset into train and val, and what would go wrong if
   we trained and validated on the same images?
2. Why does ariety in the training images (angles, backgrounds, lighting) matters
   more than having lots of similar images?
3. What do all of the different ways we can augment images? What is the purpose of each? When might you want to avoid mosaic?
4. What does the confidence threshold (--conf) controls and why might you raise or lower it during live inference.
5. Which file inside runs/detect/run1/ is the trained model, and where in the project does it get loaded for live inference?
6. What would you check first if the model performs well on validation images but poorly on the live webcam feed?

---

## Augmentation parameter reference

For tuning beyond what the visualizer shows.

| Parameter | Default | What it does | When to disable |
|-----------|---------|--------------|-----------------|
| `hsv_h` | 0.015 | Random hue shift up to ±this fraction | Object identity depends on color (e.g. distinguishing red vs blue markers) |
| `hsv_s` | 0.7 | Random saturation shift | Almost never |
| `hsv_v` | 0.4 | Random brightness shift | Almost never |
| `degrees` | 0.0 | Random rotation up to ±this many degrees | Disabled by default; enable for 10-15° if your object orientation varies |
| `translate` | 0.1 | Random translation up to ±this fraction of image | Almost never |
| `scale` | 0.5 | Random scaling up to ±this fraction | Almost never |
| `shear` | 0.0 | Random shear up to ±this many degrees | Disabled by default; enable up to 5° for variety |
| `perspective` | 0.0 | Random perspective warp | Usually leave at 0 |
| `flipud` | 0.0 | Probability of vertical flip | Disabled by default. Most objects do not appear upside-down naturally |
| `fliplr` | 0.5 | Probability of horizontal flip | Object has text or asymmetric features (a left shoe is not a right shoe) |
| `mosaic` | 1.0 | Probability of combining 4 images | Almost never; helps learn partial views |
| `mixup` | 0.0 | Probability of blending 2 images | Off by default; enable at 0.1-0.15 if overfitting |
| `close_mosaic` | 10 | Disable mosaic for last N epochs | Helps final convergence; raise if you train for many more epochs |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ultralytics'"
You forgot to activate the venv. Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) and try again.

### "Could not open webcam"
Another app may already have the camera open. Close Zoom, Teams, OBS, browser tabs with camera permissions. Exit, then try again.

### "Could not open webcam" on macOS
System Settings > Privacy & Security > Camera. Add or enable your terminal app (Terminal, iTerm, VS Code, whatever you ran the script from). You may need to fully quit and relaunch the terminal app after granting access.

### Label Studio export does not have `images/` and `labels/` folders
You probably exported in the wrong format. The Label Studio export dropdown has multiple "YOLO" variants. Pick plain "YOLO with Images", not "YOLO", "YOLOv8 OBB", or "YOLO with COCO-style annotations".

### "Need at least 4 images" when running visualize_augmentations.py
The mosaic augmentation needs 4 images. Capture more frames with `capture.py` first, or point the script at a different folder with `--image-dir`.

### Trained model misses obvious objects in live inference
A few things to try, in order:
1. Lower the confidence threshold: `--conf 0.05`.
2. Check `runs/detect/run1/val_batch0_pred.jpg` — if the model is bad on validation images too, the issue is training, not inference.
3. Capture more diverse training images. If you only labeled the object on a white desk, the model will not generalize to other backgrounds.
4. Train for more epochs: `--epochs 100`.
5. Try a larger base model: `--model yolo11s.pt` (slower but more accurate).

---

## Project structure

```
testing/yolo-walkthrough/
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── .gitignore                 # excludes data/, runs/, venv/, etc.
└── scripts/
    ├── capture.py             # webcam frame capture (Step 1)
    ├── visualize_augmentations.py  # render augmentation grid PDF (Step 4)
    ├── prepare_dataset.py     # split Label Studio export into train/val (Step 3)
    ├── train.py               # YOLO11n training (Step 5)
    └── live_inference.py      # live webcam detection (Step 6)
```

Files that get created as you go through the walkthrough (all gitignored):

```
data/
    captured/                  # frames from capture.py
    labelstudio/               # Label Studio's database (if using Docker)
    dataset/                   # split dataset created by prepare_dataset.py
        images/{train,val}/
        labels/{train,val}/
        dataset.yaml
runs/
    detect/run1/               # one folder per training run
        weights/best.pt        # trained model
        results.png            # training curves
        confusion_matrix.png
        val_batch0_pred.jpg
augmentations.pdf              # output of visualize_augmentations.py
```
