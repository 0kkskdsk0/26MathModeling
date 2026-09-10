#!/usr/bin/env python3
"""Fetch official CUMCM excellent-paper metadata and optional page images."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse


INDEX_PAGES = {
    "2025": "https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/2025qgdxssxjmjslwzs/",
    "2024": "https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/2024qgdxssxjmjslwzs/",
    "2023": "https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/2023qgdxssxjmjslwzs/2023gjsbqgdxssxjmjslwzs.shtml",
}

USER_AGENT = "Mozilla/5.0 (Codex CUMCM reference indexer)"


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="ignore")


def fetch_binary(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def clean_html_text(raw: str) -> str:
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def parse_index(year: str, url: str) -> list[dict[str, object]]:
    raw = fetch_text(url)
    hrefs = []
    for match in re.finditer(r'href="([^"]*/zx/a/hd_sxjm_sxjmlw_[^"]+?\.shtml)"', raw):
        href = urljoin(url, match.group(1))
        if href not in hrefs:
            hrefs.append(href)

    items: list[dict[str, object]] = []
    for href in hrefs:
        needle = href.replace("https://dxs.moe.gov.cn", "")
        pos = raw.find(needle)
        if pos < 0:
            pos = raw.find(href)
        chunk = raw[pos : pos + 1500] if pos >= 0 else raw
        title_match = re.search(r'title="([^"]*?展示（[^）]+）)"', chunk)
        if title_match:
            title = html.unescape(title_match.group(1))
        else:
            text_match = re.search(r"((?:20\d{2})高教社杯全国大学生数学建模竞赛[^<]{0,60}展示（[^）]+）)", clean_html_text(chunk))
            title = text_match.group(1) if text_match else f"{year}全国大学生数学建模竞赛论文展示"
        code_match = re.search(r"（([^）]+)）", title)
        code = code_match.group(1).strip() if code_match else "UNKNOWN"
        problem_type = code[0].upper() if code and code[0].isalpha() else "UNKNOWN"
        items.append(
            {
                "year": year,
                "problem_type": problem_type,
                "code": code,
                "title": title,
                "list_url": url,
                "detail_url": href,
            }
        )
    return items


def parse_detail_images(detail_url: str) -> list[str]:
    raw = fetch_text(detail_url)
    urls = []
    for match in re.finditer(r'<img[^>]+src="([^"]+)"[^>]+alt="([^"]*页面[^"]*)"', raw):
        src = html.unescape(match.group(1))
        url = urljoin(detail_url, src)
        if url not in urls:
            urls.append(url)
    if not urls:
        for match in re.finditer(r'<img[^>]+src="([^"]+)"', raw):
            src = html.unescape(match.group(1))
            if "upload/resources/image" in src:
                url = urljoin(detail_url, src)
                if url not in urls:
                    urls.append(url)
    return urls


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return safe or "file"


def infer_tags(code: str, title: str) -> list[str]:
    problem = code[:1].upper()
    tags = []
    if problem == "A":
        tags = ["optimization", "strategy", "constraints"]
    elif problem == "B":
        tags = ["physical-modeling", "mechanism", "engineering"]
    elif problem == "C":
        tags = ["data-analysis", "statistics", "machine-learning"]
    elif problem in {"D", "E"}:
        tags = ["open-ended", "interdisciplinary", "decision"]
    if "预测" in title:
        tags.append("prediction")
    if "优化" in title:
        tags.append("optimization")
    if "评价" in title:
        tags.append("evaluation")
    return sorted(set(tags))


def card_text(item: dict[str, object]) -> str:
    year = str(item["year"])
    code = str(item["code"])
    problem = str(item["problem_type"])
    title = str(item["title"])
    detail = str(item["detail_url"])
    tags = ", ".join(item.get("tags", []))
    page_count = item.get("page_count", 0)
    return f"""# {year} {code} 写法卡片

- Source: {detail}
- Title: {title}
- Problem type: {problem}
- Tags: {tags}
- Page images discovered: {page_count}
- Close-reading status: metadata-indexed

## Use First

This card currently records official metadata. Use it to select a nearby reference paper. For detailed writing imitation, OCR or close-read the paper pages first and update this card.

## Likely Useful Writing Angles

{problem_specific_guidance(problem)}

## Do Not Copy

Do not copy the paper's wording, title, figures, formulas, results, or structure mechanically. Extract only writing moves that fit the user's own model and results.
"""


def problem_specific_guidance(problem: str) -> str:
    if problem == "A":
        return "- Optimization narrative: variables -> objective -> constraints -> algorithm -> sensitivity.\n- Highlight model decomposition and verification of the chosen optimum."
    if problem == "B":
        return "- Physical-modeling narrative: mechanism -> approximation -> governing equation -> parameter solution -> consistency check.\n- Use figures to explain mechanism and measurement/result credibility."
    if problem == "C":
        return "- Data-analysis narrative: preprocessing -> feature/target relation -> model -> metric -> interpretation.\n- Explain why the selected model matches the data structure."
    if problem in {"D", "E"}:
        return "- Open-problem narrative: criterion definition -> indicator/model construction -> scenario analysis -> decision recommendation.\n- Keep interpretation separate from raw model output."
    return "- Identify the paper type before using it as a writing reference."


def download_pages(item: dict[str, object], out: Path, sleep: float) -> int:
    year = str(item["year"])
    code = str(item["code"])
    images = item.get("page_image_urls", [])
    if not isinstance(images, list):
        return 0
    dest = out / "raw" / year / code / "pages"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, url in enumerate(images, start=1):
        suffix = Path(urlparse(str(url)).path).suffix or ".jpg"
        path = dest / f"{idx:03d}{suffix}"
        if path.exists():
            count += 1
            continue
        try:
            path.write_bytes(fetch_binary(str(url)))
            count += 1
            time.sleep(sleep)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"WARN: failed to download {url}: {exc}", file=sys.stderr)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="references/excellent_papers", help="output folder")
    parser.add_argument("--years", nargs="+", default=["2025", "2024", "2023"], help="years to fetch")
    parser.add_argument("--download-pages", action="store_true", help="download page images into raw/")
    parser.add_argument("--max-papers", type=int, default=0, help="limit number of papers for testing")
    parser.add_argument("--sleep", type=float, default=0.15, help="delay between network requests")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "cards").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "ocr_text").mkdir(parents=True, exist_ok=True)

    items: list[dict[str, object]] = []
    for year in args.years:
        url = INDEX_PAGES.get(year)
        if not url:
            print(f"WARN: no configured index page for {year}", file=sys.stderr)
            continue
        print(f"Fetching index {year}: {url}", file=sys.stderr)
        for item in parse_index(year, url):
            try:
                images = parse_detail_images(str(item["detail_url"]))
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"WARN: failed detail parse {item['detail_url']}: {exc}", file=sys.stderr)
                images = []
            item["page_image_urls"] = images
            item["page_count"] = len(images)
            item["tags"] = infer_tags(str(item["code"]), str(item["title"]))
            items.append(item)
            time.sleep(args.sleep)
            if args.max_papers and len(items) >= args.max_papers:
                break
        if args.max_papers and len(items) >= args.max_papers:
            break

    items.sort(key=lambda x: (str(x["year"]), str(x["problem_type"]), str(x["code"])), reverse=True)

    index_path = out / "index.json"
    index_path.write_text(json.dumps({"source": INDEX_PAGES, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in items:
        card = out / "cards" / f"{item['year']}_{safe_filename(str(item['code']))}.md"
        if not card.exists() or "metadata-indexed" in card.read_text(encoding="utf-8", errors="ignore"):
            card.write_text(card_text(item), encoding="utf-8")
        if args.download_pages:
            downloaded = download_pages(item, out, args.sleep)
            item["downloaded_page_count"] = downloaded

    if args.download_pages:
        index_path.write_text(json.dumps({"source": INDEX_PAGES, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(items)} papers to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
