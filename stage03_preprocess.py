import argparse
import logging
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model.eval().to(device)
    return model, preprocess


def encode_image(path: Path, model, preprocess, device: str) -> torch.Tensor:
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)
    return F.normalize(feat, dim=-1).cpu()


def load_images_from_dir(src_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)


def load_images_from_hf(keyword: str) -> list[Path]:
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    tmp_dir = Path(f"/tmp/hf_cache/{keyword}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=config.HF_TOKEN)
    all_files = list(api.list_repo_files(config.HF_REPO_ID, repo_type="dataset", token=config.HF_TOKEN))
    parquet_files = [f for f in all_files if f.startswith(f"data/{keyword}/") and f.endswith(".parquet")]

    paths = []
    for pq_file in tqdm(parquet_files, desc=f"Downloading {keyword} batches"):
        raw = api.hf_hub_download(
            repo_id=config.HF_REPO_ID,
            filename=pq_file,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )
        table = pq.read_table(raw)
        for i in range(table.num_rows):
            fname = table.column("filename")[i].as_py()
            img_bytes = table.column("image_bytes")[i].as_py()
            dest = tmp_dir / fname
            if not dest.exists():
                dest.write_bytes(img_bytes)
            paths.append(dest)
    return paths


def phash_dedup(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    hashes = {}
    for p in tqdm(paths, desc="pHash"):
        try:
            hashes[p] = imagehash.phash(Image.open(p))
        except Exception:
            pass

    kept, removed = [], []
    path_list = list(hashes.keys())
    to_remove: set[Path] = set()

    for i, p1 in enumerate(path_list):
        if p1 in to_remove:
            continue
        for p2 in path_list[i + 1 :]:
            if p2 in to_remove:
                continue
            if hashes[p1] - hashes[p2] <= config.DEDUP_PHASH_THRESHOLD:
                to_remove.add(p2)

    for p in path_list:
        (removed if p in to_remove else kept).append(p)

    log.info("pHash dedup: %d kept, %d removed", len(kept), len(removed))
    return kept, removed


def histogram_dedup(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    def hist(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            return np.zeros(768)
        h = [cv2.calcHist([img], [c], None, [256], [0, 256]) for c in range(3)]
        return np.concatenate([cv2.normalize(x, x).flatten() for x in h])

    histograms = {p: hist(p) for p in tqdm(paths, desc="Histogram")}
    path_list = list(histograms.keys())
    to_remove: set[Path] = set()

    for i, p1 in enumerate(path_list):
        if p1 in to_remove:
            continue
        h1 = histograms[p1].reshape(-1, 1).astype(np.float32)
        for p2 in path_list[i + 1 :]:
            if p2 in to_remove:
                continue
            h2 = histograms[p2].reshape(-1, 1).astype(np.float32)
            score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            if score > config.DEDUP_HIST_THRESHOLD:
                to_remove.add(p2)

    kept = [p for p in path_list if p not in to_remove]
    removed = [p for p in path_list if p in to_remove]
    log.info("Histogram dedup: %d kept, %d removed", len(kept), len(removed))
    return kept, removed


def ssim_dedup(paths: list[Path], review_dir: Path) -> tuple[list[Path], list[Path]]:
    def load_gray(p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (256, 256)) if img is not None else None

    grays = {p: load_gray(p) for p in tqdm(paths, desc="SSIM")}
    path_list = [p for p, g in grays.items() if g is not None]
    to_remove: set[Path] = set()
    review_dir.mkdir(parents=True, exist_ok=True)

    for i, p1 in enumerate(path_list):
        if p1 in to_remove:
            continue
        for p2 in path_list[i + 1 :]:
            if p2 in to_remove:
                continue
            score, _ = ssim(grays[p1], grays[p2], full=True)
            if score > config.DEDUP_SSIM_HIGH:
                to_remove.add(p2)
            elif score > config.DEDUP_SSIM_REVIEW_LOW:
                shutil.copy2(p2, review_dir / f"review_{p1.stem}_vs_{p2.stem}_{score:.3f}{p2.suffix}")

    kept = [p for p in path_list if p not in to_remove]
    removed = [p for p in path_list if p in to_remove]
    log.info("SSIM dedup: %d kept, %d removed, review queue in %s", len(kept), len(removed), review_dir)
    return kept, removed


def clip_dedup(paths: list[Path], model, preprocess, device: str) -> tuple[list[Path], list[Path]]:
    embeddings = []
    valid_paths = []
    for p in tqdm(paths, desc="CLIP embed (dedup)"):
        try:
            embeddings.append(encode_image(p, model, preprocess, device))
            valid_paths.append(p)
        except Exception:
            pass

    if not embeddings:
        return paths, []

    mat = torch.cat(embeddings, dim=0)
    sim_matrix = mat @ mat.T
    n = len(valid_paths)
    to_remove: set[int] = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if sim_matrix[i, j].item() > config.DEDUP_CLIP_THRESHOLD:
                to_remove.add(j)

    kept = [valid_paths[i] for i in range(n) if i not in to_remove]
    removed = [valid_paths[i] for i in to_remove]
    log.info("CLIP dedup: %d kept, %d removed", len(kept), len(removed))
    return kept, removed


def deduplicate(paths: list[Path], model, preprocess, device: str, review_dir: Path) -> list[Path]:
    remaining, _ = phash_dedup(paths)
    remaining, _ = histogram_dedup(remaining)
    remaining, _ = ssim_dedup(remaining, review_dir)
    remaining, _ = clip_dedup(remaining, model, preprocess, device)
    log.info("Deduplication complete: %d/%d images retained", len(remaining), len(paths))
    return remaining


def reference_sort(
    paths: list[Path],
    ref_path: Path,
    model,
    preprocess,
    device: str,
    threshold: float,
) -> list[tuple[Path, float]]:
    ref_embed = encode_image(ref_path, model, preprocess, device)
    results = []
    for p in tqdm(paths, desc="Similarity scoring"):
        try:
            embed = encode_image(p, model, preprocess, device)
            score = F.cosine_similarity(ref_embed, embed).item()
            if score >= threshold:
                results.append((p, score))
        except Exception:
            pass
    results.sort(key=lambda x: -x[1])
    return results


def save_sorted_output(results: list[tuple[Path, float]], keyword: str) -> Path:
    out_dir = config.SORTED_OUTPUT_DIR / keyword
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, (src, score) in enumerate(results, start=1):
        dest = out_dir / f"rank{rank:04d}_sim{score:.3f}{src.suffix.lower()}"
        shutil.copy2(src, dest)
    return out_dir


def build_html_report(results: list[tuple[Path, float]], ref_path: Path, keyword: str, out_dir: Path) -> Path:
    cards = ""
    for rank, (p, score) in enumerate(results, start=1):
        img_name = f"rank{rank:04d}_sim{score:.3f}{p.suffix.lower()}"
        pct = int(score * 100)
        cards += f"""
        <div class="card">
            <img src="{img_name}" alt="rank {rank}" loading="lazy">
            <div class="meta">
                <span>#{rank} &nbsp; {score:.3f}</span>
                <progress value="{pct}" max="100"></progress>
            </div>
        </div>"""

    rel_ref = f"../../reference_images/{ref_path.name}" if ref_path.exists() else ""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{keyword} — Sorted Results</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 1rem; }}
  h1 {{ text-align: center; }}
  .ref {{ display: flex; justify-content: center; margin: 1rem 0; }}
  .ref img {{ max-height: 300px; border: 3px solid #4af; border-radius: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }}
  .card {{ background: #222; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; height: 200px; object-fit: cover; }}
  .meta {{ padding: 0.5rem; font-size: 0.85rem; }}
  progress {{ width: 100%; height: 6px; }}
</style>
</head>
<body>
<h1>{keyword} — {len(results)} matching images</h1>
<div class="ref"><img src="{rel_ref}" alt="reference"></div>
<div class="grid">{cards}
</div>
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def process_keyword(keyword: str, source: str, model, preprocess, device: str) -> None:
    ref_path = config.REFERENCE_IMAGES_DIR / f"{keyword}_no_person.jpg"
    review_dir = config.REVIEW_DIR / keyword

    if source == "hf":
        paths = load_images_from_hf(keyword)
    else:
        src_dir = config.RAW_IMAGES_DIR / keyword
        if not src_dir.exists():
            log.warning("No local images for '%s'", keyword)
            return
        paths = load_images_from_dir(src_dir)

    if not paths:
        log.warning("No images found for keyword '%s'", keyword)
        return

    log.info("Processing '%s': %d images", keyword, len(paths))
    deduped = deduplicate(paths, model, preprocess, device, review_dir)

    if not ref_path.exists():
        log.warning(
            "Reference image not found: %s — skipping similarity sort for '%s'",
            ref_path,
            keyword,
        )
        return

    results = reference_sort(deduped, ref_path, model, preprocess, device, config.SIMILARITY_THRESHOLD)
    print(f"\nTotal matching images for '{keyword}': {len(results)}")

    out_dir = save_sorted_output(results, keyword)
    report = build_html_report(results, ref_path, keyword, out_dir)
    log.info("Saved sorted output to %s (report: %s)", out_dir, report)


def main():
    parser = argparse.ArgumentParser(description="Stage 03 — Deduplication and reference sorting")
    parser.add_argument("--keyword", nargs="+", default=config.KEYWORDS, metavar="KW")
    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="hf",
        help="Load images from local raw_images/ or HuggingFace",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    model, preprocess = load_clip(device)

    for kw in args.keyword:
        process_keyword(kw, args.source, model, preprocess, device)


if __name__ == "__main__":
    main()
