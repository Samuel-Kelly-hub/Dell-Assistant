import json

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_sitemaps_from_jsonl(data_dir: Path) -> list[dict]:
    """Load sitemaps from JSONL files."""
    filepaths = sorted(data_dir.glob("dell_sitemaps_success_*.jsonl"))

    print("Number of files:", len(filepaths))

    sitemaps = []
    for filepath in filepaths:
        sitemap_dict = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sitemap_dict[record['sitemap_url']] = record['urls']
        sitemaps.append(sitemap_dict)

    return sitemaps


def clean_sitemaps(sitemaps: list[dict], data_dir: Path = None) -> pd.Series:
    """Extract relevant URLs from sitemaps and return as Series.

    Args:
        sitemaps: List of sitemap dictionaries (one per start URL)
        data_dir: Optional path to save output CSV

    Returns:
        pd.Series of unique URLs to scrape
    """
    urls_to_scrape = pd.Series(
        [url for url_list in sitemaps[0].values() for url in url_list if "en-us" in url] +
        [url for url_list in sitemaps[1].values() for url in url_list if "en-uk" in url],
        name="url"
    ).drop_duplicates()

    print(len(urls_to_scrape))

    if data_dir:
        urls_to_scrape.to_csv(data_dir / "urls_to_scrape.csv", index=False)

    return urls_to_scrape


if __name__ == "__main__":
    sitemaps = load_sitemaps_from_jsonl(DATA_DIR)
    clean_sitemaps(sitemaps, DATA_DIR)
