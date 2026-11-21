"""Build the cleaned BinSight dataset from data/raw/ (run scripts/download_data.py first).

Pipeline:
  1. collect candidates: TrashNet images, TACO bounding-box crops, a random sample of Kaggle organics
  2. map every original label to (material, bin) with data/label_map.csv
  3. quality filters: unreadable, too small, grayscale, blank
  4. CLIP zero-shot filter on Kaggle organics (drops drawings, text, collages, mixed piles, ...)
  5. near-duplicate removal with perceptual hashing
  6. grouped + stratified 70/15/15 split (all crops of one TACO photo stay in one split)
  7. pad to square, resize, save; write manifest, report and review contact sheets

Outputs (data/processed/):
  images/<id>.jpg   manifest.csv   removed.csv   report.md   review/*.png

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --no-clip --kaggle-sample 1000
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LABEL_MAP = ROOT / "data" / "label_map.csv"

SEED = 42
TACO_MARGIN = 0.10
MIN_SIDE = 64
KAGGLE_MIN_SIDE = 100
GRAY_TOL = 3.0
BLANK_STD = 5.0
PHASH_MAX_DIST = 4
OUT_SIZE = 256
PAD_COLOR = (124, 116, 104)  # ImageNet mean in 0-255
N_FOLDS = 20                  # 14 / 3 / 3 folds -> 70 / 15 / 15
TRAIN_FOLDS, VAL_FOLDS = 14, 3
SOURCE_PRIORITY = {"trashnet": 0, "taco": 1, "kaggle": 2}

CLIP_KEEP = [
    "a photo of food waste",
    "a photo of leftover food",
    "a photo of fruit or vegetable peels",
    "a photo of a fruit",
    "a photo of a vegetable",
    "a photo of cooked food on a plate",
    "a photo of bread",
    "a photo of eggshells",
    "a photo of rotten fruit",
    "a photo of kitchen scraps",
]
# Kaggle "O" is web-scraped and full of plants/flowers; Toronto's Green Bin does not take yard waste.
CLIP_REJECT = [
    "a cartoon or clip art illustration",
    "a drawing or sketch",
    "an image of text, a logo or a poster",
    "a diagram or infographic",
    "a collage of several photos",
    "a photo of a big pile of mixed garbage",
    "a photo of a landfill",
    "a photo of a plastic bottle or plastic packaging",
    "a photo of a product in its packaging",
    "a photo of a person",
    "a photo of human hands or feet",
    "a photo of trees, grass or a garden",
    "a photo of flowers or a bouquet",
    "a photo of a houseplant in a pot",
    "a photo of green leaves",
    "a photo of a bird, insect or animal",
    "a photo of fabric, burlap or a woven texture",
    "a photo of wood or furniture",
    "a photo of soil or dirt",
    "a photo of a basket",
]


# ----------------------------------------------------------------------------- helpers
def pad_square(img, size=OUT_SIZE):
    s = max(img.size)
    canvas = Image.new("RGB", (s, s), PAD_COLOR)
    canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def quality_issue(img, min_side, check_gray):
    if min(img.size) < min_side:
        return "too_small"
    arr = np.asarray(img.resize((64, 64)), dtype=np.float32)
    if arr.std() < BLANK_STD:
        return "blank"
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if check_gray and (np.abs(r - g).mean() + np.abs(g - b).mean()) / 2 < GRAY_TOL:
        return "grayscale"
    return None


def load_label_map():
    df = pd.read_csv(LABEL_MAP, keep_default_na=False)
    return {(r.source, r.label): (r.material, r.bin) for r in df.itertuples()}


# ----------------------------------------------------------------------------- candidates
def iter_trashnet():
    root = RAW / "trashnet" / "dataset-resized"
    for path in sorted(root.glob("*/*.jpg")):
        cls = path.parent.name
        sid = f"trashnet_{path.stem}"
        yield dict(id=sid, source="trashnet", label=cls, group=sid, src=str(path)), (lambda p=path: Image.open(p))


def iter_taco():
    coco = json.loads((RAW / "taco" / "annotations.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id in sorted(by_image):
        im = images[image_id]
        path = RAW / "taco" / "images" / Path(im["file_name"]).with_suffix(".jpg")
        cache = {}

        def load_full(p=path, c=cache):
            if "img" not in c:
                c["img"] = Image.open(p).convert("RGB")
            return c["img"]

        for ann in by_image[image_id]:
            rec = dict(id=f"taco_{image_id}_{ann['id']}", source="taco", label=cats[ann["category_id"]],
                       group=f"taco_{image_id}", src=str(path))
            if not path.exists():
                yield rec, None
                continue

            def load_crop(a=ann, W=im["width"]):
                full = load_full()
                s = full.width / W
                x, y, w, h = (v * s for v in a["bbox"])
                mx, my = TACO_MARGIN * w, TACO_MARGIN * h
                box = (max(0, x - mx), max(0, y - my), min(full.width, x + w + mx), min(full.height, y + h + my))
                return full.crop(tuple(round(v) for v in box))

            yield rec, load_crop


def iter_kaggle(sample_size, rng):
    files = sorted((RAW / "kaggle_organic").rglob("*.jpg"))
    for path in sorted(rng.sample(files, min(sample_size, len(files)))):
        sid = f"kaggle_{path.parent.name}_{path.stem}"
        yield dict(id=sid, source="kaggle", label="O", group=sid, src=str(path)), (lambda p=path: Image.open(p))


# ----------------------------------------------------------------------------- CLIP filter
def clip_scores(paths, batch_size=64):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    with torch.no_grad():
        txt = model.encode_text(tokenizer(CLIP_KEEP + CLIP_REJECT))
        txt = txt / txt.norm(dim=-1, keepdim=True)
        scores = []
        for i in tqdm(range(0, len(paths), batch_size), desc="CLIP filter"):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
            img = model.encode_image(batch)
            img = img / img.norm(dim=-1, keepdim=True)
            probs = (100.0 * img @ txt.T).softmax(dim=-1)
            scores.append(probs[:, :len(CLIP_KEEP)].sum(dim=-1))
    return torch.cat(scores).numpy()


# ----------------------------------------------------------------------------- dedup + split
def find_duplicates(df):
    """Union-find over pHash Hamming distance; returns {removed_id: reason}."""
    hashes = np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(df) - 1):
        close = np.nonzero(np.bitwise_count(hashes[i] ^ hashes[i + 1:]) <= PHASH_MAX_DIST)[0] + i + 1
        for j in close:
            parent[root(int(j))] = root(i)

    clusters = {}
    for i in range(len(df)):
        clusters.setdefault(root(i), []).append(i)

    removed = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        sub = df.iloc[members]
        if sub["bin"].nunique() > 1:
            for sid in sub["id"]:
                removed[sid] = "duplicate_label_conflict"
            continue
        keep = sub.assign(pri=sub["source"].map(SOURCE_PRIORITY)).sort_values(["pri", "id"]).iloc[0]["id"]
        for sid in sub["id"]:
            if sid != keep:
                removed[sid] = f"duplicate_of:{keep}"
    return removed


def assign_splits(df):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strata = df["source"] + "|" + df["bin"]
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, strata, groups=df["group"])):
        fold[idx] = k
    return np.where(fold < TRAIN_FOLDS, "train", np.where(fold < TRAIN_FOLDS + VAL_FOLDS, "val", "test"))


# ----------------------------------------------------------------------------- outputs
def contact_sheet(rows, path, title_fn, cols=8, thumb=128):
    rows = list(rows)
    if not rows:
        return
    n_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, n_rows * (thumb + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 14)
        sheet.paste(Image.open(r["thumb_path"]).convert("RGB").resize((thumb, thumb)), (x, y))
        draw.text((x + 2, y + thumb), title_fn(r)[:22], fill="black")
    sheet.save(path)


def table(df_counts):
    cols = [str(c) for c in df_counts.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df_counts.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def write_report(df, removed_df, n_candidates, args):
    parts = [
        "# BinSight data cleaning report\n",
        f"- candidates: {n_candidates}, kept: {len(df)}, removed: {len(removed_df)}",
        f"- seed {SEED}; Kaggle sample {args.kaggle_sample}; CLIP threshold "
        f"{'off' if args.no_clip else args.clip_threshold}; pHash distance <= {PHASH_MAX_DIST}\n",
        "## Removed (reason x source)\n",
        table(pd.crosstab(removed_df["reason"].str.split(":").str[0], removed_df["source"], margins=True)),
        "\n## Kept (source x bin)\n",
        table(pd.crosstab(df["source"], df["bin"], margins=True)),
        "\n## Split x bin\n",
        table(pd.crosstab(df["split"], df["bin"], margins=True)),
        "\n## Split x material\n",
        table(pd.crosstab(df["split"], df["material"], margins=True)),
        "\n## Review sheets\n",
        "Open `review/*.png` and spot-check each sheet; record the error rate you observe here.",
    ]
    (OUT / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-sample", type=int, default=1500)
    ap.add_argument("--clip-threshold", type=float, default=0.6)
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    label_map = load_label_map()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "review").mkdir()
    tmp = OUT / "_orig"
    tmp.mkdir()

    kept, removed = [], []
    sources = [("trashnet", iter_trashnet()), ("taco", iter_taco()), ("kaggle", iter_kaggle(args.kaggle_sample, rng))]
    n_candidates = 0
    for name, it in sources:
        for rec, loader in tqdm(it, desc=f"collect {name}"):
            n_candidates += 1
            material, bin_ = label_map.get((rec["source"], rec["label"]), ("", "drop"))
            if bin_ == "drop":
                removed.append({**rec, "reason": "label_dropped"})
                continue
            if loader is None:
                removed.append({**rec, "reason": "image_missing"})
                continue
            try:
                img = loader().convert("RGB")
            except Exception:  # noqa: BLE001 - any decode failure means the file is unusable
                removed.append({**rec, "reason": "unreadable"})
                continue
            # Real photos of grey/white objects (cans, paper) look grayscale, so only web images are checked.
            is_web = name == "kaggle"
            issue = quality_issue(img, KAGGLE_MIN_SIDE if is_web else MIN_SIDE, check_gray=is_web)
            if issue:
                removed.append({**rec, "reason": issue})
                continue
            orig_path = tmp / f"{rec['id']}.jpg"
            img.save(orig_path, quality=95)
            kept.append({**rec, "material": material, "bin": bin_, "phash": str(imagehash.phash(img)),
                         "orig_path": str(orig_path)})

    df = pd.DataFrame(kept)
    df["clip_p_keep"] = np.nan

    if not args.no_clip:
        is_k = df["source"] == "kaggle"
        df.loc[is_k, "clip_p_keep"] = clip_scores(df.loc[is_k, "orig_path"].tolist())
        rejected = is_k & (df["clip_p_keep"] < args.clip_threshold)
        df["thumb_path"] = df["orig_path"]
        contact_sheet(df[rejected].sample(min(48, rejected.sum()), random_state=SEED).to_dict("records"),
                      OUT / "review" / "kaggle_rejected_by_clip.png", lambda r: f"p={r['clip_p_keep']:.2f}")
        borderline = df[is_k & ~rejected].nsmallest(48, "clip_p_keep")
        contact_sheet(borderline.to_dict("records"), OUT / "review" / "kaggle_kept_lowest_score.png",
                      lambda r: f"p={r['clip_p_keep']:.2f}")
        removed += [{**r, "reason": "clip_not_food"} for r in df[rejected].to_dict("records")]
        df = df[~rejected].reset_index(drop=True)

    dup = find_duplicates(df)
    removed += [{**r, "reason": dup[r["id"]]} for r in df[df["id"].isin(dup)].to_dict("records")]
    df = df[~df["id"].isin(dup)].reset_index(drop=True)

    df["split"] = assign_splits(df)

    for r in tqdm(df.itertuples(), total=len(df), desc="write images"):
        pad_square(Image.open(r.orig_path)).save(OUT / "images" / f"{r.id}.jpg", quality=92)
    df["path"] = "images/" + df["id"] + ".jpg"
    df["thumb_path"] = str(OUT) + "/" + df["path"]

    for (src, b), sub in df.groupby(["source", "bin"]):
        contact_sheet(sub.sample(min(48, len(sub)), random_state=SEED).to_dict("records"),
                      OUT / "review" / f"{src}_{b}.png", lambda r: f"{r['material']}|{r['label']}")

    cols = ["id", "path", "source", "label", "material", "bin", "group", "split", "phash", "clip_p_keep", "src"]
    df[cols].to_csv(OUT / "manifest.csv", index=False)
    removed_df = pd.DataFrame(removed)[["id", "source", "label", "reason", "src"]]
    removed_df.to_csv(OUT / "removed.csv", index=False)
    write_report(df, removed_df, n_candidates, args)
    shutil.rmtree(tmp)
    print((OUT / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
