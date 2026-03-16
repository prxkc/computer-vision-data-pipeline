# Fashion Image Scraper & Preprocessor

End-to-end pipeline for building a clean fashion product image dataset — scraping, cloud storage, deduplication, and person filtering across 6 Islamic/South Asian garment keywords.

## Stages

| Script | What it does | Where to run |
|--------|-------------|--------------|
| `stage01_scraper.py` | DFS crawl → LLM URL filter → Playwright image download | Local |
| `stage02_uploader.py` | Batch upload to HuggingFace (300/batch), verify, delete local | Local |
| `stage03_preprocess.py` | 4-layer dedup + CLIP reference sorting + HTML report | Kaggle/Colab |
| `stage04_filter.py` | No-person filter (CLIP + Groq vision) | Kaggle/Colab |

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# then edit .env with your GROQ_API_KEY, HF_TOKEN, and HF_REPO_ID
```

## Usage

```bash
# Stage 01 — scrape all keywords
python stage01_scraper.py

# skip re-crawl if you already have url_cache.json
python stage01_scraper.py --skip-crawl

# Stage 02 — upload to HuggingFace
python stage02_uploader.py

# Stage 03 — dedup + sort (needs reference images in reference_images/)
python stage03_preprocess.py --source hf

# Stage 04 — person filter
python stage04_filter.py
```

For Stages 3 & 4, use `kaggle_notebook.ipynb` on Kaggle or Colab (T4 GPU). It downloads everything from HuggingFace automatically.

## Reference Images

One image per keyword in `reference_images/{keyword}_no_person.jpg` — flat lay or hanger shot, no person visible. Used as the similarity anchor for sorting and filtering.

## Output

```
sorted_output/{keyword}/
    rank0001_sim0.923.jpg
    rank0002_sim0.911.jpg
    report.html

final_output/{keyword}/
    rank0001_sim0.941.jpg
    ...
```

## HuggingFace Dataset

https://huggingface.co/datasets/prxkc/scraping-cv-pipeline

## Kaggle Notebook

https://www.kaggle.com/code/prxkc/fashion-image-preprocessing-pipeline
