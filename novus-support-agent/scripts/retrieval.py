"""
retrieval.py — Intent-aware retrieval improvements for the Novus Support Agent.

B2.1  Metadata filtering   — only load docs relevant to the classified intent
B2.2  Chunk deduplication  — remove near-duplicate chunks via Jaccard similarity

Both operate on the data/products/*.md files (Week 1's in-memory corpus).
Week 3 will replace this with pgvector-backed vector search; these two
improvements make the in-memory retrieval meaningfully smarter in the meantime.

Usage:
    from scripts.retrieval import retrieve_filtered, deduplicate_chunks
"""

from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "products"

# ---------------------------------------------------------------------------
# B2.1 — Intent → document filter map
# ---------------------------------------------------------------------------

INTENT_DOC_FILTERS: dict[str, list[str] | None] = {
    "return_or_refund":   ["personal_loan.md"],
    "order_status":       ["savings_account.md", "personal_loan.md"],
    "billing_or_payment": ["personal_loan.md", "savings_account.md"],
    "product_info":       ["personal_loan.md", "savings_account.md"],
    "membership":         ["savings_account.md", "personal_loan.md"],
    "general":            None,   # no filter — load all docs
}


def load_chunks(doc_files: list[Path]) -> list[dict]:
    """Split each document into paragraph-level chunks.

    Each chunk is a dict: {id, doc_id, content}.
    Splitting on double-newlines mirrors the sentence_aware strategy from Project A.
    """
    chunks: list[dict] = []
    for path in doc_files:
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "id": f"{doc_id}:{i}",
                "doc_id": doc_id,
                "content": para,
            })
    return chunks


def retrieve_filtered(intent: str) -> list[dict]:
    """Return chunks filtered to the docs most relevant for this intent.

    If no filter is defined (general intent), all docs are loaded.
    This reduces context noise: a membership query should not receive
    personal_loan paragraphs about EMI schedules.

    Args:
        intent: One of the 6 standard intent classes.

    Returns:
        List of chunk dicts: [{id, doc_id, content}, ...]
    """
    allowed_files = INTENT_DOC_FILTERS.get(intent)

    if allowed_files is None:
        # No filter — load everything
        doc_files = sorted(DATA_DIR.glob("*.md"))
    else:
        doc_files = [DATA_DIR / fname for fname in allowed_files if (DATA_DIR / fname).exists()]

    return load_chunks(doc_files)


# ---------------------------------------------------------------------------
# B2.2 — Chunk deduplication via word-level Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.

    Returns 0.0 if both sets are empty (avoid division by zero).
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def deduplicate_chunks(chunks: list[dict], threshold: float = 0.75) -> list[dict]:
    """Remove near-duplicate chunks using word-level Jaccard similarity.

    Two chunks are considered near-duplicates if Jaccard(A, B) >= threshold.
    The first occurrence is kept; all subsequent near-duplicates are dropped.

    Why Jaccard: it is symmetric, parameter-free, and fast to compute on
    small in-memory corpora. For a vector-backed corpus, cosine distance
    between embeddings would be more appropriate.

    Args:
        chunks:    List of chunk dicts with 'content' key.
        threshold: Jaccard threshold above which a chunk is considered a duplicate.
                   0.75 means 75% word overlap → near-identical.

    Returns:
        Deduplicated list of chunks (preserving original order).
    """
    seen_word_sets: list[set[str]] = []
    unique: list[dict] = []
    removed: list[dict] = []

    for chunk in chunks:
        words = set(chunk["content"].lower().split())
        is_dup = any(
            jaccard(words, seen) >= threshold
            for seen in seen_word_sets
            if words and seen
        )
        if is_dup:
            removed.append(chunk)
        else:
            unique.append(chunk)
            seen_word_sets.append(words)

    return unique, removed


# ---------------------------------------------------------------------------
# CLI — B2.1 and B2.2 smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- B2.1: before/after filtering comparison ---
    print("=" * 65)
    print("B2.1 — Metadata Filtering: before vs after")
    print("=" * 65)

    for test_intent, label in [("membership", "membership query"), ("return_or_refund", "warranty/prepayment query")]:
        all_chunks = load_chunks(sorted(DATA_DIR.glob("*.md")))
        filtered_chunks = retrieve_filtered(test_intent)

        print(f"\nIntent: '{test_intent}' ({label})")
        print(f"  Without filter : {len(all_chunks)} chunks from {len(list(DATA_DIR.glob('*.md')))} docs")
        print(f"  With filter    : {len(filtered_chunks)} chunks from {INTENT_DOC_FILTERS[test_intent]}")
        print(f"  Docs retrieved : {sorted(set(c['doc_id'] for c in filtered_chunks))}")

    # --- B2.2: deduplication on membership query ---
    print("\n" + "=" * 65)
    print("B2.2 — Chunk Deduplication (Jaccard threshold=0.75)")
    print("=" * 65)

    mem_chunks = retrieve_filtered("membership")
    unique, removed = deduplicate_chunks(mem_chunks, threshold=0.75)

    print(f"\nIntent: 'membership'")
    print(f"  Before dedup: {len(mem_chunks)} chunks")
    print(f"  After  dedup: {len(unique)} chunks")
    print(f"  Removed     : {len(removed)} near-duplicate(s)")

    if removed:
        print("\n  Removed chunks (first 200 chars each):")
        for c in removed:
            print(f"    [{c['doc_id']}] {c['content'][:100]}…")
    else:
        print("  No near-duplicates found in membership docs (corpus is small).")
        print("  Deduplication has more impact when the same FAQ appears across multiple docs.")
