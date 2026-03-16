import argparse
import base64
import logging
import shutil
import time
from io import BytesIO
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from groq import Groq
from PIL import Image
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_LLM_MIN_INTERVAL = 4.0  # enforce max 15 req/min (Groq free tier limit: 30 req/min)
_last_llm_call: float = 0.0


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


def image_to_base64(path: Path) -> str:
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def llm_has_person(client: Groq, path: Path) -> bool:
    global _last_llm_call
    wait = _LLM_MIN_INTERVAL - (time.time() - _last_llm_call)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = client.chat.completions.create(
            model=config.GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(path)}"},
                        },
                        {
                            "type": "text",
                            "text": "Does this image contain a visible person? Respond only YES or NO.",
                        },
                    ],
                }
            ],
            max_tokens=5,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().upper().startswith("YES")
    except Exception as exc:
        log.warning("LLM vision call failed for %s: %s", path.name, exc)
        return False
    finally:
        _last_llm_call = time.time()


def filter_keyword(keyword: str, model, preprocess, device: str, groq_client: Groq) -> None:
    src_dir = config.SORTED_OUTPUT_DIR / keyword
    if not src_dir.exists():
        log.warning("sorted_output/%s not found — run stage03 first", keyword)
        return

    ref_path = config.REFERENCE_IMAGES_DIR / f"{keyword}_no_person.jpg"
    if not ref_path.exists():
        log.warning("Reference image not found: %s", ref_path)
        return

    out_dir = config.FINAL_OUTPUT_DIR / keyword
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_embed = encode_image(ref_path, model, preprocess, device)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    candidates = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)

    kept, excluded, reviewed = 0, 0, 0
    final_items: list[tuple[Path, float]] = []

    for img_path in tqdm(candidates, desc=f"Filtering {keyword}"):
        try:
            embed = encode_image(img_path, model, preprocess, device)
            score = F.cosine_similarity(ref_embed, embed).item()
        except Exception as exc:
            log.debug("embed error %s: %s", img_path.name, exc)
            excluded += 1
            continue

        if score >= config.NO_PERSON_THRESHOLD:
            final_items.append((img_path, score))
            kept += 1
        elif score >= config.BORDERLINE_LOW:
            reviewed += 1
            if not llm_has_person(groq_client, img_path):
                final_items.append((img_path, score))
                kept += 1
            else:
                excluded += 1
        else:
            excluded += 1

    final_items.sort(key=lambda x: -x[1])
    for rank, (src, score) in enumerate(final_items, start=1):
        dest = out_dir / f"rank{rank:04d}_sim{score:.3f}{src.suffix.lower()}"
        shutil.copy2(src, dest)

    print(
        f"\n[{keyword}] kept={kept}, excluded={excluded}, LLM-reviewed={reviewed} "
        f"→ final_output/{keyword}/ ({len(final_items)} images)"
    )
    log.info("'%s' final output saved to %s", keyword, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Stage 04 — No-person product image filter")
    parser.add_argument("--keyword", nargs="+", default=config.KEYWORDS, metavar="KW")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    model, preprocess = load_clip(device)
    groq_client = Groq(api_key=config.GROQ_API_KEY)

    for kw in args.keyword:
        filter_keyword(kw, model, preprocess, device, groq_client)


if __name__ == "__main__":
    main()
