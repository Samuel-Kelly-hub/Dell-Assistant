import gzip
import json
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests

# Configuration
START_SITEMAP_URLS = ["https://www.dell.com/index-support-href-sitemap.xml.gz"]
SUCCESS_OUTPUT = "dell_sitemaps_success.jsonl"
FAILED_OUTPUT = "dell_sitemaps_failed.jsonl"


def _is_gzip_file(path: str) -> bool:
    """Check if a file is gzip-compressed by reading its magic number."""
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _download_to_tempfile(
        url: str,
        session: requests.Session,
        timeout: int,
        retries: int,
        backoff_seconds: float,
        user_agent: str,
) -> str:
    """Download a URL to a temporary file with exponential backoff retry logic."""
    headers = {"User-Agent": user_agent, "Accept": "*/*"}

    for attempt in range(retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                fd, tmp_path = tempfile.mkstemp(prefix="sitemap_", suffix=".xml")
                os.close(fd)
                with open(tmp_path, "wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        out.write(chunk)
                return tmp_path
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(backoff_seconds * (2 ** attempt))


def _iter_sitemap_entries(local_path: str) -> tuple[str, Iterable[str]]:
    """Parse a sitemap XML file and return its type (index/urlset) and list of URLs."""
    opener = gzip.open if _is_gzip_file(local_path) else open

    with opener(local_path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        root_tag = None
        in_element = None
        locs = []

        for event, elem in context:
            if root_tag is None and event == "start":
                root_tag = elem.tag

            if event == "start" and (elem.tag.endswith("sitemap") or elem.tag.endswith("url")):
                in_element = elem.tag
            elif event == "end":
                if elem.tag.endswith("loc") and in_element and (txt := (elem.text or "").strip()):
                    locs.append(txt)
                if elem.tag.endswith("sitemap") or elem.tag.endswith("url"):
                    in_element = None
                elem.clear()

        kind = "sitemapindex" if root_tag and root_tag.endswith("sitemapindex") else \
            "urlset" if root_tag and root_tag.endswith("urlset") else "unknown"
        return kind, iter(locs)


def _merge_jsonl_files(base_filename: str, worker_count: int, output_file: str):
    """Merge all JSONL files from workers into a single output file."""
    record_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for worker_id in range(worker_count):
            worker_file = f"{base_filename}-worker-{worker_id}.jsonl"
            try:
                with open(worker_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                            record_count += 1
                os.remove(worker_file)
            except FileNotFoundError:
                pass
    return record_count


def scrape_sitemap_tree(
        start_sitemap_urls: list[str],
        success_output: str,
        failed_output: str,
        sleep_seconds: float = 0.0,
        timeout: int = 30,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        user_agent: str = "Mozilla/5.0 (compatible; sitemap-scraper/1.0)",
        workers: int = 10,
) -> list[dict[str, list[str]]]:
    """Recursively scrape a sitemap tree with parallel workers and save results to JSONL files."""

    all_results = []

    for idx, start_sitemap_url in enumerate(start_sitemap_urls, start=1):
        print(f"\nProcessing sitemap {idx}/{len(start_sitemap_urls)}: {start_sitemap_url}")

        # Create unique output filenames for each sitemap
        success_file = success_output.replace(".jsonl", f"_{idx}.jsonl")
        failed_file = failed_output.replace(".jsonl", f"_{idx}.jsonl")

        session = requests.Session()
        sitemap_queue = deque([start_sitemap_url])
        seen_sitemaps: set[str] = set([start_sitemap_url])

        # Counter for processed sitemaps (for logging)
        processed_count = [0]
        worker_count = [0]

        def process_sitemap(sitemap_url: str, worker_id: int):
            """Process a single sitemap URL and write to worker-specific JSONL file."""
            processed_count[0] += 1
            count = processed_count[0]

            if sleep_seconds:
                time.sleep(sleep_seconds)

            # Worker-specific output files
            worker_success_file = f"{success_output}-worker-{worker_id}.jsonl"
            worker_failed_file = f"{failed_output}-worker-{worker_id}.jsonl"

            # Download
            try:
                tmp_path = _download_to_tempfile(
                    sitemap_url, session, timeout, retries, backoff_seconds, user_agent
                )
            except Exception as e:
                error_msg = str(e)
                print(f"[{count}] {sitemap_url} - ERROR: Failed to download - {error_msg}")
                # Write failure to worker's failed file
                with open(worker_failed_file, 'a', encoding='utf-8') as f:
                    json.dump({"sitemap_url": sitemap_url, "error": error_msg}, f)
                    f.write('\n')
                return

            # Parse
            try:
                kind, loc_iter = _iter_sitemap_entries(tmp_path)

                if kind == "sitemapindex":
                    new_sitemaps = list(loc_iter)
                    unseen_sitemaps = [s for s in new_sitemaps if s not in seen_sitemaps]
                    seen_sitemaps.update(unseen_sitemaps)
                    sitemap_queue.extend(unseen_sitemaps)
                    print(f"[{count}] {sitemap_url} - Found {len(new_sitemaps)} child sitemaps")
                else:
                    urls = list(loc_iter)
                    # Write success to worker's success file
                    with open(worker_success_file, 'a', encoding='utf-8') as f:
                        json.dump({"sitemap_url": sitemap_url, "urls": urls}, f)
                        f.write('\n')
                    print(f"[{count}] {sitemap_url} - Found {len(urls)} URLs")
            except Exception as e:
                error_msg = str(e)
                print(f"[{count}] {sitemap_url} - ERROR: Failed to parse - {error_msg}")
                # Write failure to worker's failed file
                with open(worker_failed_file, 'a', encoding='utf-8') as f:
                    json.dump({"sitemap_url": sitemap_url, "error": error_msg}, f)
                    f.write('\n')
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Process queue with worker threads
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = set()

            while sitemap_queue or futures:
                # Submit new tasks from queue
                while sitemap_queue and len(futures) < workers:
                    sitemap_url = sitemap_queue.popleft()
                    future = executor.submit(process_sitemap, sitemap_url, worker_count[0])
                    futures.add(future)
                    worker_count[0] += 1

                # Remove completed futures
                futures = {f for f in futures if not f.done()}

                time.sleep(0.01)

        # Merge intermediate files into final outputs
        print("\nMerging intermediate files...")
        success_count = _merge_jsonl_files(success_output, worker_count[0], success_file)
        failed_count = _merge_jsonl_files(failed_output, worker_count[0], failed_file)

        # Build results dict and calculate total URLs in one pass
        sitemap_results = {}
        total_urls = 0
        with open(success_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sitemap_results[record['sitemap_url']] = record['urls']
                    total_urls += len(record['urls'])

        print(f"Successful sitemaps: {success_count}")
        print(f"Failed sitemaps: {failed_count}")
        print(f"Total URLs found: {total_urls}")

        all_results.append(sitemap_results)

    return all_results


if __name__ == "__main__":
    scrape_sitemap_tree(
        start_sitemap_urls=START_SITEMAP_URLS,
        success_output=SUCCESS_OUTPUT,
        failed_output=FAILED_OUTPUT,
    )
