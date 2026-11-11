"""Write fixed dev/test file lists to data/splits/ (tracked in git).

LOL (paths to the low image; the reference is the same name under /high/):
    test: all 15 official eval15 pairs + 85 sampled from our485   -> 100 pairs
    dev : 50 more sampled from our485
ExDark (paths to images; labels follow the ultralytics images/ -> labels/ convention):
    dev : 50 from the official Val split (imageclasslist.txt == 2), class-stratified
    test: 17 per class from the official Test split (== 3)          -> 204 images

Paths are relative to the repo root. 
Usage:
    python src/data_prep/make_splits.py [--seed 253]
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXDARK_CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog",
                  "Motorbike", "People", "Table"]  # imageclasslist.txt class ids 1..12
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def lol_splits(rng, n_test=100, n_dev=50):
    lol = ROOT / "data" / "LOL"
    eval15 = sorted((lol / "eval15" / "low").glob("*.png"), key=lambda p: int(p.stem))
    pool = sorted((lol / "our485" / "low").glob("*.png"), key=lambda p: int(p.stem))
    for p in eval15 + pool:
        assert (p.parent.parent / "high" / p.name).exists(), f"missing reference for {p}"
    picked = rng.sample(pool, (n_test - len(eval15)) + n_dev)
    test = eval15 + sorted(picked[: n_test - len(eval15)], key=lambda p: int(p.stem))
    dev = sorted(picked[n_test - len(eval15):], key=lambda p: int(p.stem))
    return dev, test


def stratified(rng, by_class, n_total):
    """Sample n_total items spread as evenly as possible across classes."""
    classes = sorted(by_class)
    base, extra = divmod(n_total, len(classes))
    bonus = set(rng.sample(classes, extra))
    out = []
    for c in classes:
        out += rng.sample(sorted(by_class[c]), base + (c in bonus))
    return sorted(out)


def exdark_splits(rng, n_dev=50, n_test_per_class=17):
    ex = ROOT / "data" / "ExDark"
    by_split = {2: defaultdict(list), 3: defaultdict(list)}
    for line in (ex / "imageclasslist.txt").read_text().splitlines()[1:]:
        f = line.split()
        if len(f) < 5:
            continue
        name, cls, split = f[0], EXDARK_CLASSES[int(f[1]) - 1], int(f[4])
        if split in by_split:
            path = ex / "images" / cls / name
            assert path.exists() and path.suffix.lower() in IMG_EXTS, f"missing image {path}"
            by_split[split][cls].append(path)
    dev = stratified(rng, by_split[2], n_dev)
    test = stratified(rng, by_split[3], n_test_per_class * len(EXDARK_CLASSES))
    return dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=253)
    args = ap.parse_args()
    out_dir = ROOT / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in (("lol", lol_splits), ("exdark", exdark_splits)):
        dev, test = fn(random.Random(f"{args.seed}-{name}"))
        assert not set(dev) & set(test)
        for split, items in (("dev", dev), ("test", test)):
            path = out_dir / f"{name}_{split}.txt"
            path.write_text("".join(rel(p) + "\n" for p in items))
            print(f"{rel(path)}: {len(items)}")


if __name__ == "__main__":
    main()
