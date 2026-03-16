import argparse
import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from random import uniform
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from groq import Groq
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class RobotChecker:
    def __init__(self):
        self._cache: dict[str, RobotFileParser] = {}

    def _load(self, origin: str) -> RobotFileParser:
        if origin not in self._cache:
            rp = RobotFileParser(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            self._cache[origin] = rp
        return self._cache[origin]

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return self._load(origin).can_fetch("*", url)

    def crawl_delay(self, url: str) -> float:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        delay = self._load(origin).crawl_delay("*")
        return float(delay) if delay else 1.0


class AntiDetection:
    def __init__(self):
        self._ua = UserAgent()
        self.session = requests.Session()

    def headers(self) -> dict:
        return {
            "User-Agent": self._ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.session.get(url, headers=self.headers(), timeout=15, **kwargs)

    def throttle(self, base: float = 1.0, jitter: float = 2.5) -> None:
        time.sleep(uniform(base, base + jitter))

    def random_ua(self) -> str:
        return self._ua.random


_OVERLAY_SELECTORS = [
    'button[aria-label="close"]',
    'button[aria-label="Close"]',
    ".modal-close",
    ".popup-close",
    ".cc-btn.cc-dismiss",
    "#gdpr-accept",
    '[data-dismiss="modal"]',
    '[class*="cookie"] button',
    '[id*="cookie"] button',
    '[class*="popup"] button[class*="close"]',
    '[class*="newsletter"] button[class*="close"]',
]


def dismiss_overlays(page) -> None:
    for sel in _OVERLAY_SELECTORS:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                page.wait_for_timeout(300)
        except Exception:
            pass


def normalize_url(url: str) -> str:
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, "", "", ""))


class DFSCrawler:
    def __init__(self, seeds: list[str], robots: RobotChecker, max_pages: int):
        self._seeds = seeds
        self._robots = robots
        self._max_pages = max_pages
        self._allowed = {urlparse(u).netloc for u in seeds}

    def _is_internal(self, url: str) -> bool:
        return urlparse(url).netloc in self._allowed

    def _extract_links(self, html: str, base: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if href.startswith(("javascript:", "mailto:", "tel:", "#", "data:")):
                continue
            full = urljoin(base, href)
            p = urlparse(full)
            if p.scheme in ("http", "https") and self._is_internal(full):
                links.append(normalize_url(full))
        return links

    def crawl(self, anti: AntiDetection) -> list[str]:
        visited: set[str] = set()
        stack = [normalize_url(u) for u in self._seeds]
        with tqdm(desc="DFS crawl", unit="pages") as pbar:
            while stack and len(visited) < self._max_pages:
                url = stack.pop()
                if url in visited:
                    continue
                if not self._robots.can_fetch(url):
                    visited.add(url)
                    continue
                try:
                    resp = anti.get(url, allow_redirects=True)
                    ct = resp.headers.get("Content-Type", "")
                    if "text/html" not in ct:
                        visited.add(url)
                        continue
                    for lnk in self._extract_links(resp.text, resp.url):
                        if lnk not in visited:
                            stack.append(lnk)
                    visited.add(url)
                    pbar.update(1)
                    anti.throttle(max(self._robots.crawl_delay(url), 1.0))
                except Exception as exc:
                    log.debug("crawl skip %s: %s", url, exc)
                    visited.add(url)
        log.info("Crawl done: %d unique URLs discovered", len(visited))
        return list(visited)


class LLMFilter:
    def __init__(self, client: Groq, keyword: str):
        self._client = client
        self._keyword = keyword

    def _score_batch(self, urls: list[str]) -> list[dict]:
        body = "\n".join(f"{i}. {u}" for i, u in enumerate(urls))
        prompt = (
            f"Score each URL 0.0–1.0 for likelihood of containing '{self._keyword}' "
            f"fashion product images.\n"
            f"Return ONLY a valid JSON array: [{{\"url\": \"...\", \"score\": 0.0}}]\n\n{body}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=config.GROQ_TEXT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a URL relevance classifier. Respond only with JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
            return json.loads(text)
        except Exception as exc:
            log.warning("LLM batch failed: %s", exc)
            return [{"url": u, "score": 0.0} for u in urls]

    def filter(self, urls: list[str]) -> list[str]:
        results: list[dict] = []
        for i in range(0, len(urls), config.LLM_BATCH_SIZE):
            batch = urls[i : i + config.LLM_BATCH_SIZE]
            results.extend(self._score_batch(batch))
            time.sleep(uniform(0.5, 1.2))
        score_map = {r["url"]: r.get("score", 0.0) for r in results}
        relevant = [u for u, s in score_map.items() if s >= config.LLM_FILTER_THRESHOLD]
        relevant.sort(key=lambda u: score_map[u], reverse=True)
        log.info(
            "LLM filter '%s': %d/%d URLs relevant", self._keyword, len(relevant), len(urls)
        )
        return relevant


def _parse_srcset(srcset: str, base: str) -> list[str]:
    urls = []
    for part in srcset.split(","):
        tokens = part.strip().split()
        if tokens:
            urls.append(urljoin(base, tokens[0]))
    return urls


def _check_dimensions(data: bytes) -> bool:
    try:
        img = Image.open(BytesIO(data))
        return img.width >= config.MIN_IMAGE_DIM and img.height >= config.MIN_IMAGE_DIM
    except Exception:
        return False


_IMG_ATTRS = ("src", "data-src", "data-lazy-src", "data-original", "data-image", "data-full")
_IMG_EXT_RE = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.IGNORECASE)


def _extract_image_urls(page, page_url: str) -> list[str]:
    try:
        html = page.content()
    except Exception:
        return []
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()
    for img in soup.find_all("img"):
        for attr in _IMG_ATTRS:
            val = img.get(attr, "").strip()
            if val and not val.startswith("data:"):
                found.add(urljoin(page_url, val))
        if img.get("srcset"):
            found.update(_parse_srcset(img["srcset"], page_url))
    for source in soup.find_all("source"):
        if source.get("srcset"):
            found.update(_parse_srcset(source["srcset"], page_url))
    return [u for u in found if _IMG_EXT_RE.search(u)]


def scrape_keyword(
    keyword: str,
    urls: list[str],
    anti: AntiDetection,
    robots: RobotChecker,
    dry_run: bool = False,
) -> int:
    out_dir = config.RAW_IMAGES_DIR / keyword
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=anti.random_ua(),
            viewport={"width": 1280, "height": 800},
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        for url in tqdm(urls, desc=f"Scraping {keyword}"):
            if not robots.can_fetch(url):
                continue
            page = None
            try:
                page = ctx.new_page()
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                page.wait_for_timeout(800)
                dismiss_overlays(page)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(500)

                if dry_run:
                    img_urls = _extract_image_urls(page, url)
                    log.info("[dry-run] %s → %d candidate images", url, len(img_urls))
                    page.close()
                    continue

                for img_url in _extract_image_urls(page, url):
                    try:
                        resp = anti.get(img_url)
                        if resp.status_code != 200:
                            continue
                        data = resp.content
                        if not _check_dimensions(data):
                            continue
                        name = Path(urlparse(img_url).path).name
                        if not name or "." not in name:
                            name = f"img_{abs(hash(img_url)) % 10**8:08d}.jpg"
                        dest = out_dir / name
                        n = 1
                        while dest.exists():
                            stem, ext = name.rsplit(".", 1)
                            dest = out_dir / f"{stem}_{n}.{ext}"
                            n += 1
                        dest.write_bytes(data)
                        total += 1
                        anti.throttle(0.1, 0.2)
                    except Exception as exc:
                        log.debug("img dl error %s: %s", img_url, exc)

                anti.throttle(max(robots.crawl_delay(url), 0.5), 1.0)
            except PlaywrightTimeout:
                log.warning("timeout: %s", url)
            except Exception as exc:
                log.warning("page error %s: %s", url, exc)
            finally:
                if page and not page.is_closed():
                    page.close()

        browser.close()

    log.info("Keyword '%s': %d images saved to %s", keyword, total, out_dir)
    return total


def main():
    parser = argparse.ArgumentParser(description="Stage 01 — Web scraping pipeline")
    parser.add_argument("--keyword", nargs="+", default=config.KEYWORDS, metavar="KW")
    parser.add_argument("--max-pages", type=int, default=config.MAX_CRAWL_PAGES)
    parser.add_argument("--dry-run", action="store_true", help="Discover URLs without downloading")
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Load URLs from url_cache.json instead of re-crawling",
    )
    args = parser.parse_args()

    robots = RobotChecker()
    anti = AntiDetection()
    groq_client = Groq(api_key=config.GROQ_API_KEY)

    cache_file = Path("url_cache.json")
    if args.skip_crawl and cache_file.exists():
        all_urls = json.loads(cache_file.read_text())
        log.info("Loaded %d URLs from cache", len(all_urls))
    else:
        crawler = DFSCrawler(config.SEED_URLS, robots, max_pages=args.max_pages)
        all_urls = crawler.crawl(anti)
        cache_file.write_text(json.dumps(all_urls, indent=2))
        log.info("Saved %d URLs to %s", len(all_urls), cache_file)

    for kw in args.keyword:
        kw_cache = Path(f"relevant_urls_{kw}.json")
        if kw_cache.exists():
            relevant = json.loads(kw_cache.read_text())
            log.info("Loaded %d relevant URLs for '%s' from cache", len(relevant), kw)
        else:
            kw_seeds = config.KEYWORD_SEEDS.get(kw)
            if kw_seeds:
                kw_crawler = DFSCrawler(kw_seeds, robots, max_pages=args.max_pages)
                kw_urls = kw_crawler.crawl(anti)
            else:
                kw_urls = all_urls
            llm = LLMFilter(groq_client, kw)
            relevant = llm.filter(kw_urls)
            kw_cache.write_text(json.dumps(relevant, indent=2))

        if not relevant:
            log.warning("No relevant URLs found for keyword '%s'", kw)
            continue

        scrape_keyword(kw, relevant, anti, robots, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
