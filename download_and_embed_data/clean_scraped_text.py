import re

import numpy as np
import pandas as pd

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def clean_scraped_text(df: pd.DataFrame, data_dir: Path = None) -> pd.DataFrame:
    """Clean scraped text and return DataFrame ready for embedding.

    Args:
        df: DataFrame with columns: url, title, text
        data_dir: Optional path to save output CSV

    Returns:
        DataFrame with columns: url, text, title, product
    """
    # Separate webpages and PDFs for cleaning
    is_pdf = df["url"].str.endswith("pdf", na=False)
    html_df = df.loc[~is_pdf].copy()
    pdf_df = df.loc[is_pdf].copy()

    # Cleaning text from webpages
    partition = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
    pre_clean = lambda text: re.sub(
        r"[ \t]+", " ",
        re.sub(r" *\n *", "\n",
               str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", " ").replace("\xa0", " ")
        ).strip().lower()
    )

    html_df["cleaned_text"] = html_df["text"].apply(lambda t: t if pd.isna(t) else re.sub(r"\s*\|\s*Dell UK\s*", " ", pre_clean(t)).strip())
    html_df["cleaned_title"] = html_df["title"].apply(lambda t: t if pd.isna(t) else pre_clean(t).replace("\n", " "))

    # Deleting headers/footers in webpages
    doc_sents = html_df["cleaned_text"].dropna().map(
        lambda text: {s.strip() for s in partition.split(text) if s.strip()}
    )
    n_docs = len(doc_sents)

    common_html_df = (doc_sents.explode().value_counts() / n_docs).reset_index()
    common_html_df.columns = ["sentence", "proportion"]

    sentences_to_delete = set(common_html_df.loc[common_html_df["proportion"] == 1., "sentence"])

    html_df["cleaned_text"] = html_df["cleaned_text"].apply(
        lambda t: t if pd.isna(t) else "\n".join(
            s.strip() for s in partition.split(t) if s.strip() and s.strip() not in sentences_to_delete
        )
    )

    finding_product_mask = html_df["url"].fillna("").str.contains(r"/supportedos/", regex=True)

    html_df["product"] = np.nan
    html_df.loc[finding_product_mask, "product"] = html_df.loc[finding_product_mask, "url"].str.extract(r"/supportedos/([^/?#]+)", expand=False)

    html_df["product"] = html_df["product"].fillna("general")

    # Cleaning PDF texts (Upon inspection the vast majority of scraped text from PDF documents was the text I wanted to scrape.)
    pdf_df = pdf_df.dropna(subset=["text"])
    pdf_df["product"] = pdf_df["url"].str.extract(r"/([^/_]+)(?:_[^/]*)?\.pdf$", expand=False)

    html_df = html_df.drop(["text", "title"], axis=1)
    html_df = html_df.rename({"cleaned_text": "text", "cleaned_title": "title"}, axis=1)

    # Combining the 2 dfs
    df_to_embed = pd.concat([html_df, pdf_df], ignore_index=True)

    if data_dir:
        df_to_embed.to_csv(data_dir / "text_to_embed.csv")
        df_to_embed["product"].drop_duplicates().to_csv(DATA_DIR / "product_list.csv", index=False)

    return df_to_embed


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "scraped_pages-final.csv", usecols=["url", "title", "text"])
    clean_scraped_text(df, DATA_DIR)
