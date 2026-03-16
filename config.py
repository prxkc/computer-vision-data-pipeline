import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO_ID = os.environ.get("HF_REPO_ID", "prxkc/scraping-cv-pipeline")

KEYWORDS = ["panjabi", "burkha", "jubba", "thobe", "shalwar_kameez", "abaya"]

SEED_URLS = [
    "https://anzaarlifestyle.com/",
    "https://soldiersbd.com/",
    "https://jdot-bangladesh.com/",
]

KEYWORD_SEEDS = {
    "panjabi": ["https://easyfashion.com.bd/product-category/panjabi/"],
    "shalwar_kameez": ["https://infinitymegamall.com/product-category/women/kameez/salwar-kameez-salwar-kameez/"],
    "burkha": ["https://iraniborkabazar.com/", "https://emaanbd.com/", "https://dubaiborka.com.bd/"],
}

MIN_IMAGE_DIM = 500
BATCH_SIZE = 300
MAX_CRAWL_PAGES = 2000
LLM_BATCH_SIZE = 50
LLM_FILTER_THRESHOLD = 0.6

DEDUP_PHASH_THRESHOLD = 8
DEDUP_HIST_THRESHOLD = 0.98
DEDUP_CLIP_THRESHOLD = 0.97
DEDUP_SSIM_HIGH = 0.92
DEDUP_SSIM_REVIEW_LOW = 0.80

SIMILARITY_THRESHOLD = 0.55
NO_PERSON_THRESHOLD = 0.75
BORDERLINE_LOW = 0.50

GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

RAW_IMAGES_DIR = Path("raw_images")
MANIFESTS_DIR = Path("manifests")
SORTED_OUTPUT_DIR = Path("sorted_output")
FINAL_OUTPUT_DIR = Path("final_output")
REFERENCE_IMAGES_DIR = Path("reference_images")
REVIEW_DIR = Path("review_queue")
