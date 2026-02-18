from pathlib import Path
import csv
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache


_DASH_MAP = str.maketrans({c: "-" for c in "–—−‐-‒"})
_KEEP = re.compile(r"[^a-z0-9-]+")
_WS = re.compile(r"\s+")
_DASHES = re.compile(r"-{2,}")


def get_product_candidates(user_product: str, data_dir: Path, k: int = 10) -> tuple[str, list[str], bool]:
    """
    Returns (canonicalised_input, top_k_candidates, exact_match).
    Candidates are allowlisted product slugs from DATA_DIR/"product_list.csv", best-first.
    """
    def canonicalise(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = _WS.sub("-", s.translate(_DASH_MAP).lower().strip().replace("_", "-"))
        s = _DASHES.sub("-", _KEEP.sub("-", s)).strip("-")
        return s

    @lru_cache(maxsize=1)
    def _load_allowlist(csv_path: str) -> tuple[list[str], set[str]]:
        path = Path(csv_path)
        rows = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].strip():
                    rows.append(row[0].strip())

        slug_to_raw: dict[str, str] = {}
        for raw in rows:
            slug = canonicalise(raw)
            prev = slug_to_raw.setdefault(slug, raw)
            if prev != raw:
                raise ValueError(f"Canonicalisation collision: {prev!r} vs {raw!r} -> {slug!r}")

        slugs = list(slug_to_raw.keys())
        return slugs, set(slugs)

    slugs, slug_set = _load_allowlist(str(data_dir / "product_list.csv"))
    q = canonicalise(user_product)
    qtoks = set(q.split("-")) if q else set()

    def score(p: str) -> float:
        ptoks = set(p.split("-"))
        jacc = len(qtoks & ptoks) / max(len(qtoks | ptoks), 1)
        prefix = (p.startswith(q) or q.startswith(p)) if q else False
        return (p == q) * 3.0 + prefix * 0.75 + jacc * 0.75 + SequenceMatcher(None, q, p).ratio()

    top = sorted(slugs, key=score, reverse=True)[:k]
    return q, top, q in slug_set

