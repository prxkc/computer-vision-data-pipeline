import argparse
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_manifest(keyword: str) -> dict:
    config.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MANIFESTS_DIR / f"{keyword}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "keyword": keyword,
        "uploaded": [],
        "batch_count": 0,
        "total_count": 0,
        "last_updated": None,
    }


def save_manifest(keyword: str, manifest: dict) -> None:
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    path = config.MANIFESTS_DIR / f"{keyword}.json"
    path.write_text(json.dumps(manifest, indent=2))


def retry_with_backoff(fn, max_retries: int = 5, base_delay: float = 2.0):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            log.warning("Attempt %d/%d failed: %s — retrying in %.0fs", attempt + 1, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2


def build_parquet_buffer(image_paths: list[Path]) -> io.BytesIO:
    images_bytes, filenames = [], []
    for p in image_paths:
        images_bytes.append(p.read_bytes())
        filenames.append(p.name)
    table = pa.table(
        {
            "image_bytes": pa.array(images_bytes, type=pa.large_binary()),
            "filename": pa.array(filenames),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    return buf


def upload_batch(
    api: HfApi, image_paths: list[Path], keyword: str, batch_id: int
) -> bool:
    remote_path = f"data/{keyword}/batch_{batch_id:04d}.parquet"
    buf = build_parquet_buffer(image_paths)

    def _push():
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=remote_path,
            repo_id=config.HF_REPO_ID,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )

    retry_with_backoff(_push)
    log.info("Uploaded batch %04d for '%s' (%d images)", batch_id, keyword, len(image_paths))
    return True


def verify_upload(api: HfApi, keyword: str, batch_id: int) -> bool:
    remote_path = f"data/{keyword}/batch_{batch_id:04d}.parquet"
    try:
        files = list(api.list_repo_files(config.HF_REPO_ID, repo_type="dataset", token=config.HF_TOKEN))
        return remote_path in files
    except Exception as exc:
        log.warning("Verification failed for batch %04d: %s", batch_id, exc)
        return False


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_keyword(api: HfApi, keyword: str, batch_size: int, dry_run: bool) -> None:
    src_dir = config.RAW_IMAGES_DIR / keyword
    if not src_dir.exists():
        log.warning("No images directory for keyword '%s': %s", keyword, src_dir)
        return

    manifest = load_manifest(keyword)
    uploaded_set = set(manifest["uploaded"])

    pending = [
        p for p in sorted(src_dir.iterdir())
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        and p.name not in uploaded_set
    ]

    if not pending:
        log.info("No new images to upload for '%s'", keyword)
        return

    log.info("Keyword '%s': %d images pending upload in batches of %d", keyword, len(pending), batch_size)

    for batch in tqdm(list(chunks(pending, batch_size)), desc=f"Uploading {keyword}"):
        if dry_run:
            log.info("[dry-run] would upload %d images (batch %04d)", len(batch), manifest["batch_count"] + 1)
            continue

        batch_id = manifest["batch_count"] + 1

        uploaded = upload_batch(api, batch, keyword, batch_id)
        if not uploaded:
            log.error("Upload failed for batch %04d — skipping deletion", batch_id)
            continue

        if not verify_upload(api, keyword, batch_id):
            log.error("Verification failed for batch %04d — skipping deletion", batch_id)
            continue

        for p in batch:
            p.unlink(missing_ok=True)

        manifest["uploaded"].extend(p.name for p in batch)
        manifest["batch_count"] = batch_id
        manifest["total_count"] += len(batch)
        save_manifest(keyword, manifest)
        log.info("Batch %04d committed: %d images deleted locally", batch_id, len(batch))


def main():
    parser = argparse.ArgumentParser(description="Stage 02 — HuggingFace batch uploader")
    parser.add_argument("--keyword", nargs="+", default=config.KEYWORDS, metavar="KW")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=config.HF_TOKEN)

    try:
        api.create_repo(
            config.HF_REPO_ID,
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token=config.HF_TOKEN,
        )
        log.info("Dataset repo ready: %s", config.HF_REPO_ID)
    except Exception as exc:
        log.warning("Repo creation skipped: %s", exc)

    readme_path = "README.md"
    try:
        api.upload_file(
            path_or_fileobj=_dataset_readme().encode(),
            path_in_repo=readme_path,
            repo_id=config.HF_REPO_ID,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )
    except Exception:
        pass

    for kw in args.keyword:
        process_keyword(api, kw, args.batch_size, dry_run=args.dry_run)


def _dataset_readme() -> str:
    splits_yaml = "\n".join(
        f"  - name: {kw}" for kw in config.KEYWORDS
    )
    return f"""---
dataset_info:
  features:
  - name: image_bytes
    dtype: binary
  - name: filename
    dtype: string
  splits:
{splits_yaml}
---
# Fashion Product Images
Scraped fashion product images for computer vision training.
Keywords: {', '.join(config.KEYWORDS)}
"""


if __name__ == "__main__":
    main()
