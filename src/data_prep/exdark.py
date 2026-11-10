"""Prepare ExDark for YOLO: convert labels, then draw a few for a visual check.

    python src/data_prep/exdark.py

Input:  data/ExDark/images/<Class>/<name>.<ext>
        data/ExDark/annotations/<Class>/<name>.<ext>.txt   ("% bbGt version=3" + "Class left top width height ...")
Output: data/ExDark/labels/<Class>/<name>.txt              ("coco_id cx cy w h", normalized)
        results/label_check/<Class>_<name>.jpg
"""
import random
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXDARK = ROOT / "data/ExDark"
N_CHECK, CHECK_SEED = 5, 0  # how many check images to draw; change the seed to see different ones

# ExDark class -> COCO-80 class ID (ultralytics ordering)
COCO_ID = {"People": 0, "Bicycle": 1, "Car": 2, "Motorbike": 3, "Bus": 5, "Boat": 8, "Cat": 15,
           "Dog": 16, "Bottle": 39, "Cup": 41, "Chair": 56, "Table": 60}
COCO_NAME = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 8: "boat", 15: "cat",
             16: "dog", 39: "bottle", 41: "cup", 56: "chair", 60: "dining table"}


def list_images():
    return sorted(p for p in (EXDARK / "images").rglob("*") if p.is_file() and not p.name.startswith("."))


def label_path(img: Path) -> Path:
    return EXDARK / "labels" / img.parent.name / f"{img.stem}.txt"


def strip_exif_rotation(img: Path):
    """ExDark boxes are drawn on raw pixels, but OpenCV/YOLO rotate images by their EXIF tag on
    load. Reset any rotation tag so the image is read as it was annotated."""
    with Image.open(img) as im:
        exif = im.getexif()
        if exif.get(0x0112, 1) in (0, 1):  # 0x0112 = Orientation; 0/1 = no rotation
            return
        exif[0x0112] = 1
        im.load()
        im.save(img, im.format, exif=exif.tobytes(), quality=95, subsampling=0)
        print(f"stripped EXIF rotation: {img.name}")


def convert_labels():
    """bbGt line 'Class left top width height ...' -> YOLO line 'id cx cy w h' (normalized)."""
    images = list_images()
    assert len({label_path(p) for p in images}) == len(images), "two images map to one label file"
    n_boxes = 0
    for img in images:
        strip_exif_rotation(img)
        with Image.open(img) as im:
            W, H = im.size
        lines = []
        ann = EXDARK / "annotations" / img.parent.name / f"{img.name}.txt"
        for row in ann.read_text().splitlines()[1:]:  # line 1 is the "% bbGt version=3" header
            if not row.strip():
                continue
            cls, l, t, w, h = row.split()[:5]
            x0, y0 = max(float(l), 0), max(float(t), 0)  # clip boxes to the image
            x1, y1 = min(float(l) + float(w), W), min(float(t) + float(h), H)
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                lines.append(f"{COCO_ID[cls]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                             f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
        label_path(img).parent.mkdir(parents=True, exist_ok=True)
        label_path(img).write_text("".join(lines))
        n_boxes += len(lines)
    print(f"labels: {len(images)} images, {n_boxes} boxes -> {label_path(images[0]).parents[1].relative_to(ROOT)}")


def check_labels(n=N_CHECK, seed=CHECK_SEED, out=ROOT / "results/label_check"):
    """Draw the YOLO labels back on n random images (brightened) to confirm the conversion by eye.
    OpenCV applies EXIF rotation like YOLO does, so a correct overlay checks the full round trip."""
    out.mkdir(parents=True, exist_ok=True)
    for img_path in random.Random(seed).sample(list_images(), n):
        img = cv2.convertScaleAbs(cv2.imread(str(img_path)), alpha=2.0, beta=20)
        H, W = img.shape[:2]
        t = max(2, max(H, W) // 400)  # line thickness
        for row in label_path(img_path).read_text().splitlines():
            c, cx, cy, w, h = map(float, row.split())
            x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            cv2.rectangle(img, (x0, y0), (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 255, 0), t)
            cv2.putText(img, COCO_NAME[int(c)], (x0 + 3, max(y0 - 6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, t / 3, (0, 255, 255), t // 2)
        s = min(1, 800 / max(H, W))
        dest = out / f"{img_path.parent.name}_{img_path.stem}.jpg"
        cv2.imwrite(str(dest), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        print(f"check: {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    convert_labels()
    check_labels()
