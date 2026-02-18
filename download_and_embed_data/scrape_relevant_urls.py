import os
import re
import time
import fitz
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image

from pathlib import Path
from playwright.sync_api import sync_playwright
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = Path.cwd().parent / "data"
PDF_DIR = Path(r"C:\Users\busin\Documents\dell_pdfs")
HTML_DIR = Path(r"C:\Users\busin\Documents\htmls")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _sanitise_url_for_filename(url: str) -> str:
    """
    Turn a URL into a Windows-safe filename base.
    """
    # Remove leading http:// or https://
    base = re.sub(r"^https?://", "", url)

    # Replace Windows-invalid filename characters with underscores
    base = re.sub(r'[\\/:*?"<>|]', "_", base)

    # Truncate to avoid long path errors
    base = base[:240]

    return base


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""

    error_str = str(error).lower()
    if ("timeout" in type(error).__name__.lower() or "timeout" in error_str or
            any(x in error_str for x in ["500", "502", "503", "504", "429"])):
        return True
    elif any(x in error_str for x in ["404", "403", "400"]):
        return False

    return True


def _scrape_pages(df, output_csv_path: str):
    first_write = not os.path.exists(output_csv_path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-GB',
            timezone_id='Europe/London',
            extra_http_headers={
                'Accept-Language': 'en-GB,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        page = context.new_page()

        for i, (index, row) in enumerate(df.iterrows(), start=1):

            url = row["url"]
            print(f"Processing URL {i}: {url}")
            fetch_ok = True
            html_content = None

            # Handle PDFs separately
            if url.lower().endswith('.pdf'):
                retry_delays = [5, 10, 30]
                for attempt in range(4):
                    try:
                        # Download the URL as a binary HTTP request (not the viewer HTML)
                        api_response = context.request.get(url, timeout=0)

                        content_type = api_response.headers.get("content-type", "").lower()
                        if (not api_response.ok) or ("pdf" not in content_type):
                            raise ValueError(
                                f"URL did not return a PDF (status={api_response.status}, content-type={content_type})"
                            )

                        pdf_bytes = api_response.body()

                        # Save PDF to disk with ...-1.pdf, ...-2.pdf, etc.
                        base_file_name = _sanitise_url_for_filename(url)
                        file_attempt = 1
                        while True:
                            pdf_filename = f"{base_file_name}-{file_attempt}.pdf"
                            pdf_path = PDF_DIR / pdf_filename
                            if not pdf_path.exists():
                                with open(pdf_path, "wb") as f:
                                    f.write(pdf_bytes)
                                break
                            file_attempt += 1

                        # Try multiple extraction methods with PyMuPDF
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        text = ""

                        # Method 1: Standard text extraction
                        for pdf_page in doc:
                            page_text = pdf_page.get_text()
                            if page_text.strip():
                                text += page_text

                        # Method 2: If no text, try "blocks" extraction
                        if not text.strip():
                            for pdf_page in doc:
                                blocks = pdf_page.get_text("blocks")
                                for block in blocks:
                                    if len(block) >= 5:
                                        text += block[4] + "\n"

                        # Method 3: If still no text, use OCR (scanned image)
                        if not text.strip():
                            print(f"PDF is scanned image, using OCR: {url}")
                            for page_num, pdf_page in enumerate(doc):
                                pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                                page_text = pytesseract.image_to_string(img)
                                text += page_text + "\n"

                                print(f"  OCR page {page_num + 1}/{len(doc)}")

                        doc.close()

                        text_length = len(text.strip())
                        if text_length == 0:
                            print(f"Warning: OCR failed, no text extracted: {url}")
                        else:
                            print(f"Success: {i} (PDF) - Extracted {text_length} chars")

                        df.loc[index, ["title", "text", "html_path"]] = ["", text, ""]
                        fetch_ok = True
                        break

                    except Exception as e:
                        if _is_retryable_error(e) and attempt < 3:
                            print(f"Attempt {attempt + 1}/4 failed for {url}: {e}")
                            time.sleep(retry_delays[attempt])
                        else:
                            print(str(e))
                            df.loc[index, ["title", "text", "html_path"]] = ["", None, None]
                            fetch_ok = False
                            break

                # Write row to CSV
                row_df = df.loc[[index]]
                row_df.to_csv(output_csv_path, mode="a", index=False, header=first_write)
                first_write = False

                if fetch_ok:
                    print(f"Worker downloaded PDF from {url}")

                time.sleep(1)
                continue

            # Retry page loading with exponential backoff
            retry_delays = [5, 10, 30]  # Delays in seconds for retries
            for attempt in range(4):  # 1 initial + 3 retries
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    html_content = page.content()

                    # Successfully loaded page, save HTML file
                    base_file_name = _sanitise_url_for_filename(url)
                    file_attempt = 1
                    while True:
                        filename = f"{base_file_name}-{file_attempt}.html"
                        try:
                            with open(HTML_DIR / filename, "x", encoding="utf-8") as f:
                                f.write(html_content)
                            df.at[index, "html_path"] = filename
                            break
                        except FileExistsError:
                            file_attempt += 1
                    break  # Success, exit retry loop

                except Exception as e:
                    # Check if error is retryable
                    if _is_retryable_error(e) and attempt < 3:
                        print(f"Attempt {attempt + 1}/4 failed for {url}: {e}")
                        time.sleep(retry_delays[attempt])
                    else:
                        print(str(e))
                        df.loc[index, ["title", "text", "html_path"]] = [None, None, None]
                        fetch_ok = False
                        break  # Exit retry loop on non-retryable error

            if fetch_ok and html_content:
                try:
                    body_text = page.inner_text("body")
                except Exception:
                    body_text = page.inner_text("*")

                df.loc[index, ["title", "text"]] = [page.title(), body_text]
                print("Success:", i)

            row_df = df.loc[[index]]
            row_df.to_csv(output_csv_path, mode="a", index=False, header=first_write)
            first_write = False

            time.sleep(5)

        context.close()
        browser.close()


def run_pool_with_retries(worker_fn, splits, output_paths, max_workers, pause_seconds, urls_already_scraped=None):

    attempt = 1
    while True:
        print(f"Starting ProcessPoolExecutor attempt {attempt}")
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(worker_fn, splits, output_paths))
            print(f"ProcessPoolExecutor attempt {attempt} completed successfully")
            return

        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping without further retries.")
            raise

        except Exception as e:
            print(f"ProcessPoolExecutor failed on attempt {attempt} with error: {e}")

            if urls_already_scraped is not None:
                csv_paths = [p for p in output_paths if os.path.exists(p)]
                if csv_paths:
                    dfs = [pd.read_csv(p) for p in csv_paths]
                    done_df = pd.concat(dfs, ignore_index=True)

                    if "url" in done_df.columns:
                        urls_already_scraped.update(done_df["url"].dropna())

                    combined_df = pd.concat(splits, ignore_index=False)
                    if "url" in combined_df.columns:
                        remaining_df = combined_df[~combined_df["url"].isin(urls_already_scraped)]
                    else:
                        remaining_df = combined_df

                    if remaining_df.empty:
                        print("No URLs left to scrape after failure. Stopping retries.")
                        return

                    n_splits = len(splits) or 1
                    splits = np.array_split(remaining_df, n_splits)

            if attempt >= 3: return
            print(f"Retrying in {pause_seconds} seconds...")
            time.sleep(pause_seconds)
            attempt += 1


def scrape_property_listings(df, base_name:str):

    # URLs that have already been scraped in a previous run
    final_csv_path = DATA_DIR / f"{base_name}-final.csv"
    already_scraped_df = set()
    if final_csv_path.exists():
        already_scraped_df = pd.read_csv(final_csv_path, usecols=["url"])

    urls_already_scraped = set(already_scraped_df["url"].dropna())

    # URLS that have crashed PyMuPDF (only 2), if there were more I may have excluded them in a more elegant way
    urls_already_scraped.update(
        {
            "https://downloads.dell.com/manuals/all-products/esuprt_laptop/esuprt_xps_laptop/xps-13-l321x-mlk_setup%20guide2_en-us.pdf",
            "https://downloads.dell.com/manuals/all-products/esuprt_printers_main/esuprt_printers_aio_inkjet/dell-v525w-inkjet-printer_user's%20guide_en-us.pdf"
        }
    )

    df[["title", "text", "html_path"]] = None

    df = df.sample(frac=1, random_state=0)

    # To delete
    df = df[~df["url"].isin(urls_already_scraped)]
    num_batches = len(df) // 100
    splits = np.array_split(df, num_batches)

    # Run extractors in parallel
    output_paths = [str(DATA_DIR / f"{base_name}-worker-{i + 1}.csv") for i in range(len(splits))]
    run_pool_with_retries(
        worker_fn=_scrape_pages,
        splits=splits,
        output_paths=output_paths,
        max_workers=5,
        pause_seconds=60,
        urls_already_scraped=urls_already_scraped,
    )

    # After all splits are done, merge the per-split CSVs into a final dataframe
    final_df_parts = []

    for split_id in range(1, len(splits) + 1):
        part_path = DATA_DIR / f"{base_name}-worker-{split_id}.csv"
        part_df = pd.read_csv(part_path)
        final_df_parts.append(part_df)

    # Concatenate all parts into final dataframe
    final_df = pd.concat(final_df_parts, ignore_index=True)

    # Write final dataframe to CSV
    final_df.to_csv(final_csv_path, mode="a", header=not final_csv_path.exists(), index=False)

    # Clean up intermediate CSV files after successful write
    for split_id in range(1, len(splits) + 1):
        try:
            os.remove(DATA_DIR / f"{base_name}-worker-{split_id}.csv")
        except OSError as e:
            pass

    return final_df


if __name__ == "__main__":
    scrape_property_listings(pd.read_csv(DATA_DIR / "urls_to_scrape.csv"), "scraped_pages")