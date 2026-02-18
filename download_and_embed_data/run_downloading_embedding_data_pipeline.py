"""
Pipeline orchestrator for the Dell Assistant data processing workflow.

Runs all 5 steps in sequence:
1. Scrape sitemaps - fetch URLs from Dell sitemaps
2. Clean sitemaps - extract relevant en-us/en-uk URLs
3. Scrape pages - download and extract text from URLs
4. Clean text - process and prepare text for embedding
5. Embed text - generate embeddings and store in Qdrant

Data is passed in memory between steps whilst still writing to files for persistence.
"""

from pathlib import Path

from scrape_sitemaps import scrape_sitemap_tree, START_SITEMAP_URLS, SUCCESS_OUTPUT, FAILED_OUTPUT
from clean_sitemaps import clean_sitemaps
from scrape_relevant_urls import scrape_property_listings
from clean_scraped_text import clean_scraped_text
from embed_text import embed_text

DATA_DIR = Path(__file__).parent.parent / "data"


def run_pipeline():
    """Run the full data processing pipeline."""

    print("=" * 60)
    print("Step 1: Scraping sitemaps")
    print("=" * 60)
    sitemaps = scrape_sitemap_tree(
        start_sitemap_urls=START_SITEMAP_URLS,
        success_output=str(DATA_DIR / SUCCESS_OUTPUT),
        failed_output=str(DATA_DIR / FAILED_OUTPUT),
    )

    print("\n" + "=" * 60)
    print("Step 2: Cleaning sitemaps - extracting relevant URLs")
    print("=" * 60)
    urls_to_scrape = clean_sitemaps(sitemaps, DATA_DIR)

    print("\n" + "=" * 60)
    print("Step 3: Scraping pages (this step is slow - ~13k URLs)")
    print("=" * 60)
    scraped_df = scrape_property_listings(urls_to_scrape.to_frame(), "scraped_pages")

    print("\n" + "=" * 60)
    print("Step 4: Cleaning scraped text")
    print("=" * 60)
    df_to_embed = clean_scraped_text(scraped_df, DATA_DIR)

    print("\n" + "=" * 60)
    print("Step 5: Embedding text into Qdrant")
    print("=" * 60)
    embed_text(df_to_embed)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
