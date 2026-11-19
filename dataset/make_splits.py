"""Write fixed dev/test file lists to data/splits/ (tracked in git).

Run from the project root:
    python dataset/make_splits.py
    python dataset/make_splits.py --seed 253

LOL (paths to the low image; the reference is the same name under /high/):
    test: all 15 official eval15 pairs + 85 sampled from our485   -> 100 pairs
    dev : 50 more sampled from our485
ExDark (paths to images; labels follow the ultralytics images/ -> labels/ convention):
    dev : 50 from the official Val split (imageclasslist.txt == 2), class-stratified
    test: 17 per class from the official Test split (== 3)          -> 204 images

Reads data/LOL/ and data/ExDark/{images,imageclasslist.txt} (see README "Data").
Writes data/splits/{lol,exdark}_{dev,test}.txt with paths relative to the repo root.
Re-running with the same seed reproduces the same lists; labels come from dataset/exdark.py.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EXDARK_CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog",
                  "Motorbike", "People", "Table"]  # imageclasslist.txt class ids 1..12
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def rel(p: Path) -> str:
    """Return a path relative to the repo root in POSIX form, as stored in the split files."""
    return p.relative_to(PROJECT_ROOT).as_posix()


def stratified(rng: random.Random, by_class: dict[str, list[Path]], n_total: int) -> list[Path]:
    """Sample n_total items spread as evenly as possible across classes."""
    classes = sorted(by_class)
    base, extra = divmod(n_total, len(classes))
    bonus = set(rng.sample(classes, extra))
    out = []
    for c in classes:
        out += rng.sample(sorted(by_class[c]), base + (c in bonus))
    return sorted(out)


def lol_splits(rng: random.Random, n_test: int = 100, n_dev: int = 50) -> tuple[list[Path], list[Path]]:
    """Return (dev, test) LOL low-light paths: test = eval15 + sampled our485, dev = more our485."""
    lol = PROJECT_ROOT / "data" / "LOL"
    eval15 = sorted((lol / "eval15" / "low").glob("*.png"), key=lambda p: int(p.stem))
    pool = sorted((lol / "our485" / "low").glob("*.png"), key=lambda p: int(p.stem))
    for p in eval15 + pool:
        if not (p.parent.parent / "high" / p.name).exists():
            raise FileNotFoundError(f"Missing reference for {p}")
    picked = rng.sample(pool, (n_test - len(eval15)) + n_dev)
    test = eval15 + sorted(picked[: n_test - len(eval15)], key=lambda p: int(p.stem))
    dev = sorted(picked[n_test - len(eval15):], key=lambda p: int(p.stem))
    return dev, test


def exdark_splits(rng: random.Random, n_dev: int = 50, n_test_per_class: int = 17) -> tuple[list[Path], list[Path]]:
    """Return (dev, test) ExDark image paths, class-stratified from the official Val and Test splits."""
    ex = PROJECT_ROOT / "data" / "ExDark"
    by_split: dict[int, defaultdict[str, list[Path]]] = {2: defaultdict(list), 3: defaultdict(list)}
    for line in (ex / "imageclasslist.txt").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split()
        if len(f) < 5:
            continue
        name, cls, split = f[0], EXDARK_CLASSES[int(f[1]) - 1], int(f[4])
        if split in by_split:
            path = ex / "images" / cls / name
            if path.suffix.lower() not in IMG_EXTS:
                raise ValueError(f"Unsupported image extension {path.suffix!r} for {path}; expected {sorted(IMG_EXTS)}")
            if not path.exists():
                raise FileNotFoundError(f"Missing image {path}")
            by_split[split][cls].append(path)
    dev = stratified(rng, by_split[2], n_dev)
    test = stratified(rng, by_split[3], n_test_per_class * len(EXDARK_CLASSES))
    return dev, test


def main() -> None:
    """Make and save the LOL and ExDark dev/test lists."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=253)
    args = parser.parse_args()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, fn in (("lol", lol_splits), ("exdark", exdark_splits)):
        # One RNG per dataset, so changing one dataset's sampling leaves the other's lists intact.
        dev, test = fn(random.Random(f"{args.seed}-{name}"))
        if set(dev) & set(test):
            raise ValueError(f"{name}: dev and test overlap")
        for split, items in (("dev", dev), ("test", test)):
            path = SPLITS_DIR / f"{name}_{split}.txt"
            path.write_text("".join(rel(p) + "\n" for p in items), encoding="utf-8")
            print(f"Saved {len(items)} paths: {rel(path)}")


if __name__ == "__main__":
    main()
