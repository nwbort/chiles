#!/usr/bin/env python3
"""
Scrape all Adrian Chiles article titles from The Guardian.

Uses the Guardian Content API (free, no sign-up needed with the 'test' key).
For higher rate limits, register for a free key at:
  https://open-platform.theguardian.com/access/

Usage:
    python scrape.py                     # uses the 'test' API key
    python scrape.py --api-key <your_key>
    python scrape.py --output results.json  # also save structured JSON
"""

import argparse
import json
import sys
import time

import requests

GUARDIAN_API = "https://content.guardianapis.com/search"
CONTRIBUTOR_TAG = "profile/adrian-chiles"
PAGE_SIZE = 50  # max 200 with a registered key; 10 with 'test'


def fetch_page(api_key: str, page: int) -> dict:
    params = {
        "tag": CONTRIBUTOR_TAG,
        "api-key": api_key,
        "page-size": PAGE_SIZE,
        "page": page,
        "show-fields": "headline,trailText,shortUrl",
        "order-by": "oldest",
    }
    resp = requests.get(GUARDIAN_API, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["response"]


def scrape_all(api_key: str) -> list[dict]:
    articles = []
    page = 1

    first = fetch_page(api_key, page)
    total_pages = first["pages"]
    total = first["total"]
    print(f"Found {total} articles across {total_pages} pages", flush=True)

    articles.extend(first["results"])
    print(f"  Page {page}/{total_pages}: {len(first['results'])} articles")

    for page in range(2, total_pages + 1):
        time.sleep(0.2)  # stay well within rate limits
        data = fetch_page(api_key, page)
        articles.extend(data["results"])
        print(f"  Page {page}/{total_pages}: {len(data['results'])} articles")

    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Adrian Chiles Guardian articles")
    parser.add_argument("--api-key", default="test", help="Guardian API key (default: test)")
    parser.add_argument("--output", default="titles.txt", help="Output file (default: titles.txt)")
    args = parser.parse_args()

    if args.api_key == "test":
        # test key caps page-size at 10
        global PAGE_SIZE
        PAGE_SIZE = 10
        print("Using 'test' API key (page size capped at 10). Register a free key for faster scraping.")

    print(f"Fetching articles…")
    try:
        articles = scrape_all(args.api_key)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)

    titles = [a.get("fields", {}).get("headline") or a["webTitle"] for a in articles]

    print(f"\n{len(titles)} articles total:\n")
    for i, title in enumerate(titles, 1):
        print(f"{i:>4}. {title}")

    txt_path = args.output if args.output.endswith(".txt") else args.output + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for title in titles:
            f.write(title + "\n")
    print(f"\nTitles saved to {txt_path}")

    if args.output.endswith(".json") or "--output" in sys.argv:
        json_path = args.output if args.output.endswith(".json") else txt_path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "title": a.get("fields", {}).get("headline") or a["webTitle"],
                        "url": a["webUrl"],
                        "date": a["webPublicationDate"],
                    }
                    for a in articles
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Structured data saved to {json_path}")


if __name__ == "__main__":
    main()
