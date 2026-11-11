"""Prepare a reproducible COCO 2017 subset for motion-blur experiments.

Examples:
    python coco.py --download annotations
    python coco.py --prepare
    python coco.py --download selected
    python coco.py --download images
    python coco.py --download all --prepare
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import zipfile
from pathlib import Path


class COCODataset:
    URLS = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "images": "http://images.cocodataset.org/zips/val2017.zip",
    }
    TARGET_CLASSES = {
        # People & Transportation
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",

        # Animals
        "cat",
        "dog",
        "bird",

        # Outdoor Objects
        "traffic light",
        "bench",

        # Indoor Objects
        "chair",
        "bottle",
        "laptop",
        "cell phone",
    }
    def __init__(self, root: str | Path = "data/coco") -> None:
        self.root = Path(root)
        self.annotation_path = self.root / "annotations" / "instances_val2017.json"
        self.images_dir = self.root / "val2017"
        self.splits_dir = self.root / "splits"

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        """Download via curl using the same HTTP endpoint verified on this Mac."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already downloaded: {destination}")
            return
        partial = destination.with_name(destination.name + ".part")
        try:
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "3",
                 "--output", str(partial), url],
                check=True,
            )
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError(f"Empty download: {url}")
            if destination.suffix == ".zip" and not zipfile.is_zipfile(partial):
                raise RuntimeError(f"Downloaded content is not a ZIP archive: {url}")
            if destination.suffix.lower() == ".jpg":
                with partial.open("rb") as stream:
                    if stream.read(3) != b"\xff\xd8\xff":
                        raise RuntimeError(f"Downloaded content is not a JPEG: {url}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    def download(self, what: str) -> None:
        """Download annotations, full validation images, or only selected images."""
        if what == "selected":
            self.download_selected_images()
            return
        types = ("annotations", "images") if what == "all" else (what,)
        for kind in types:
            url = self.URLS[kind]
            archive = self.root / url.rsplit("/", 1)[-1]
            target = self.annotation_path if kind == "annotations" else self.images_dir
            if target.exists() and (kind == "annotations" or any(target.glob("*.jpg"))):
                print(f"Already extracted: {target}")
                continue
            self._download_file(url, archive)
            print(f"Extracting: {archive}")
            with zipfile.ZipFile(archive) as z:
                z.extractall(self.root)
            print(f"Ready: {target}")

    def get_photos(self, count: int = 400, seed: int = 42) -> list[int]:
        """Select COCO image IDs containing at least one target class.

        Only annotations are needed; JPEGs need not be downloaded yet.
        """
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Missing {self.annotation_path}; run --download annotations first"
            )
        with self.annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        category_ids = {
            category["id"]
            for category in coco["categories"]
            if category["name"] in self.TARGET_CLASSES
        }
        eligible_ids = {
            ann["image_id"]
            for ann in coco["annotations"]
            if ann["category_id"] in category_ids and ann.get("iscrowd", 0) == 0
        }
        # Restrict to images listed in this annotation file.
        valid_ids = {img["id"] for img in coco["images"]}
        candidates = sorted(eligible_ids & valid_ids)
        if len(candidates) < count:
            raise ValueError(f"Only {len(candidates)} eligible images; requested {count}")
        return random.Random(seed).sample(candidates, count)

    @staticmethod
    def split_photos(photo_ids: list[int], dev_size: int = 100) -> tuple[list[int], list[int]]:
        """Split already-shuffled unique image IDs into dev and test sets."""
        if not 0 < dev_size < len(photo_ids):
            raise ValueError("dev_size must be > 0 and less than the number of photos")
        if len(photo_ids) != len(set(photo_ids)):
            raise ValueError("photo_ids must not contain duplicates")
        return photo_ids[:dev_size], photo_ids[dev_size:]

    def prepare(self, count: int = 400, dev_size: int = 100, seed: int = 42) -> None:
        """Save stable ID manifests without copying or downloading image files."""
        photo_ids = self.get_photos(count=count, seed=seed)
        dev_ids, test_ids = self.split_photos(photo_ids, dev_size=dev_size)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        for name, ids in (("development", dev_ids), ("test", test_ids)):
            path = self.splits_dir / f"{name}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(ids, f, indent=2)
            print(f"Saved {len(ids)} {name} IDs: {path}")

    def download_selected_images(self) -> None:
        """Download only the JPEGs listed in the saved split manifests."""
        all_ids: set[int] = set()
        for name in ("development", "test"):
            path = self.splits_dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; run --prepare first")
            with path.open("r", encoding="utf-8") as f:
                all_ids.update(json.load(f))
        self.images_dir.mkdir(parents=True, exist_ok=True)
        for i, image_id in enumerate(sorted(all_ids), 1):
            filename = f"{image_id:012d}.jpg"
            path = self.images_dir / filename
            if not path.exists():
                self._download_file(
                    f"http://images.cocodataset.org/val2017/{filename}", path
                )
            if i % 50 == 0 or i == len(all_ids):
                print(f"Images checked: {i}/{len(all_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./coco"))
    parser.add_argument(
        "--download", choices=("annotations", "images", "selected", "all"),
        help="What to download (selected requires saved splits from --prepare)",
    )
    parser.add_argument("--prepare", action="store_true", help="Create development/test ID manifests")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = COCODataset(args.root)
    if args.download and args.download != "selected":
        dataset.download(args.download)
    if args.prepare:
        dataset.prepare(args.count, args.dev_size, args.seed)
    if args.download == "selected":
        dataset.download("selected")
    if not args.download and not args.prepare:
        parser.print_help()


if __name__ == "__main__":
    main()
