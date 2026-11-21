"""Download TrashNet, TACO and the Kaggle Waste Classification dataset into data/raw/.

Usage:
    python scripts/download_data.py            # all three datasets
    python scripts/download_data.py --only taco
"""
import argparse
import io
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
TACO_ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
KAGGLE_ID = "techsash/waste-classification-data"

# TACO originals are up to ~4000 px; bounding boxes are rescaled at crop time.
TACO_MAX_SIDE = 2048


def fetch(url, timeout=60, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise


def download_trashnet():
    out = RAW / "trashnet"
    if out.exists() and any(out.rglob("*.jpg")):
        print(f"[trashnet] already present at {out}")
        return
    print("[trashnet] downloading dataset-resized.zip ...")
    data = fetch(TRASHNET_URL, timeout=300)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(out)
    shutil.rmtree(out / "__MACOSX", ignore_errors=True)
    n = sum(1 for _ in out.rglob("*.jpg"))
    print(f"[trashnet] {n} images -> {out}")


def load_taco_image(content, width, height):
    """Return the image in the orientation the annotations were made in."""
    img = Image.open(io.BytesIO(content))
    upright = ImageOps.exif_transpose(img)
    if upright.size == (width, height):
        img = upright
    elif img.size != (width, height) and upright.size[::-1] != (width, height):
        img = upright.resize((width, height))
    return img.convert("RGB")


def download_taco(workers=16):
    out = RAW / "taco"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out / "annotations.json"
    if not ann_path.exists():
        ann_path.write_bytes(fetch(TACO_ANN_URL))
    coco = json.loads(ann_path.read_text(encoding="utf-8"))

    def job(im):
        target = img_dir / Path(im["file_name"]).with_suffix(".jpg")
        if target.exists():
            return None
        last_err = None
        for key in ("flickr_url", "flickr_640_url"):
            url = im.get(key)
            if not url:
                continue
            try:
                img = load_taco_image(fetch(url), im["width"], im["height"])
                scale = min(1.0, TACO_MAX_SIDE / max(img.size))
                if scale < 1.0:
                    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
                target.parent.mkdir(parents=True, exist_ok=True)
                img.save(target, quality=92)
                return None
            except Exception as e:  # noqa: BLE001 - any failure falls through to the next URL
                last_err = e
        return f"{im['id']}\t{im['file_name']}\t{last_err}"

    failures = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(job, im) for im in coco["images"]]
        for f in tqdm(as_completed(futures), total=len(futures), desc="[taco] images"):
            if (msg := f.result()) is not None:
                failures.append(msg)
    (out / "failed_downloads.txt").write_text("\n".join(failures), encoding="utf-8")
    n = sum(1 for _ in img_dir.rglob("*.jpg"))
    print(f"[taco] {n}/{len(coco['images'])} images -> {img_dir} ({len(failures)} failed)")


def download_kaggle():
    out = RAW / "kaggle_organic"
    if out.exists() and any(out.rglob("*.jpg")):
        print(f"[kaggle] already present at {out}")
        return
    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_ID)) / "DATASET"
    for split in ("TRAIN", "TEST"):
        dst = out / split.lower()
        shutil.copytree(src / split / "O", dst, dirs_exist_ok=True)
    n = sum(1 for _ in out.rglob("*.jpg"))
    print(f"[kaggle] {n} organic images -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["trashnet", "taco", "kaggle"])
    args = ap.parse_args()
    steps = {"trashnet": download_trashnet, "taco": download_taco, "kaggle": download_kaggle}
    for name, fn in steps.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
