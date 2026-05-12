"""
pii_anonymizer.py — Reversible PII anonymizer for the Novus Support Agent.

PiiAnonymizer detects PII in a query, replaces each span with a typed
placeholder (e.g. EMAIL_ADDRESS_a3f7c2b1), and stores the reverse mapping so
the original values can be restored in the generated answer.

Design rules:
  - One instance per request. NEVER share across requests.
  - The LLM receives only anonymized text; raw PII never enters a prompt.
  - Placeholders are deterministic per request (same entity → same placeholder
    within a single anonymize() call, different UUIDs across requests).
  - ORDER_ID is treated as PII (see module docstring for rationale).

Why ORDER_ID is PII:
  An order ID uniquely identifies a customer transaction. Combined with any
  other identifier it can expose order history, delivery address, billing
  details, and payment method. Under India's DPDPA (Digital Personal Data
  Protection Act 2023), any data that can identify an individual directly or
  indirectly is "personal data" — an order ID, cross-referenced with name or
  phone, qualifies. Even without explicit linkage, logging order IDs in LLM
  prompts creates an audit trail that can be subpoenaed or leaked.

Week 4 additions (P3.3):
  - PiiAnonymizer.get_pii_types() — list of entity type names detected
  - redaction_audit_log()         — append JSONL entry for each PII event:
      {timestamp, trace_id, pii_types, query_hash (SHA256), intent}
    Never logs the original query. Only called when has_pii() is True.

Usage (import):
    from scripts.pii_anonymizer import PiiAnonymizer, redaction_audit_log

    anon = PiiAnonymizer()          # new instance per request
    clean = anon.anonymize(query)   # replace PII
    # ... run pipeline on clean ...
    final = anon.restore(raw_answer) # put original values back
    if anon.has_pii():
        redaction_audit_log(query, anon, intent=intent, trace_id=trace_id)

Usage (CLI):
    python scripts/pii_anonymizer.py              # P3.1 demo: 5 queries
    python scripts/pii_anonymizer.py --audit-test # P3.3 demo: 20 queries → audit.jsonl
"""

import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider


# ---------------------------------------------------------------------------
# Build the analyzer once at module level (expensive to initialise)
# ---------------------------------------------------------------------------

def _build_analyzer() -> AnalyzerEngine:
    """Return an AnalyzerEngine with a custom ORDER_ID recogniser added."""
    analyzer = AnalyzerEngine()

    # Custom recogniser: ORDER_ID matches ORD-<digits>
    # Treated as PII because order IDs uniquely link to customer identity.
    order_id_recognizer = PatternRecognizer(
        supported_entity="ORDER_ID",
        patterns=[
            Pattern(
                name="order_id_pattern",
                regex=r"\bORD-\d+\b",
                score=0.9,
            )
        ],
    )
    analyzer.registry.add_recognizer(order_id_recognizer)
    return analyzer


_ANALYZER = _build_analyzer()


# ---------------------------------------------------------------------------
# PiiAnonymizer
# ---------------------------------------------------------------------------

class PiiAnonymizer:
    """Reversible PII anonymizer — scoped to one request.

    Never share an instance across requests; each request gets a fresh UUID
    namespace so placeholders from one request cannot collide with another.
    """

    # Entities to detect. spaCy NER handles PERSON / LOCATION;
    # presidio built-ins handle EMAIL, PHONE, CREDIT_CARD, DATE_TIME.
    ENTITIES = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "PERSON",
        "CREDIT_CARD",
        "DATE_TIME",
        "LOCATION",
        "ORDER_ID",
    ]

    def __init__(self) -> None:
        self._map: dict[str, str] = {}   # placeholder → original value
        self._analyzer = _ANALYZER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize(self, text: str) -> str:
        """Detect PII and replace each span with a reversible placeholder.

        Replacements happen right-to-left so earlier spans are not shifted
        by later replacements.
        """
        results = self._analyzer.analyze(
            text=text,
            entities=self.ENTITIES,
            language="en",
        )

        if not results:
            return text

        # Remove overlapping spans: keep highest-score entity for each overlap
        results = _remove_overlaps(results)

        # Sort descending by start position — replace from end to preserve indices
        results.sort(key=lambda r: r.start, reverse=True)

        chars = list(text)
        for result in results:
            original_value = text[result.start:result.end]
            placeholder    = self._get_or_create_placeholder(result.entity_type, original_value)
            chars[result.start:result.end] = list(placeholder)

        return "".join(chars)

    def restore(self, text: str) -> str:
        """Replace all placeholders in text back with their original values."""
        for placeholder, original in self._map.items():
            text = text.replace(placeholder, original)
        return text

    def has_pii(self) -> bool:
        """Return True if anonymize() found any PII in the request."""
        return bool(self._map)

    def get_pii_types(self) -> list[str]:
        """Return sorted list of unique entity type names found by anonymize().

        Extracts the entity type prefix from each placeholder key.
        Placeholder format: ENTITY_TYPE_<8HEX>
        e.g. EMAIL_ADDRESS_7DFCB917 → EMAIL_ADDRESS
        """
        types: set[str] = set()
        for placeholder in self._map:
            # Last underscore + 8 hex chars = suffix; everything before = entity type
            parts = placeholder.rsplit("_", 1)
            if len(parts) == 2:
                types.add(parts[0])
        return sorted(types)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_placeholder(self, entity_type: str, original_value: str) -> str:
        """Return an existing placeholder for original_value or create a new one.

        Same value within a request always gets the same placeholder
        (idempotent within a single anonymize() call).
        """
        # Check if this exact original value already has a placeholder
        for ph, orig in self._map.items():
            if orig == original_value:
                return ph

        placeholder = f"{entity_type}_{uuid.uuid4().hex[:8].upper()}"
        self._map[placeholder] = original_value
        return placeholder


# ---------------------------------------------------------------------------
# Helper: remove overlapping recogniser results
# ---------------------------------------------------------------------------

def _remove_overlaps(results: list) -> list:
    """Given a list of RecognizerResult, remove lower-score overlapping spans."""
    # Sort by score descending so higher-confidence results win
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    kept = []
    for candidate in sorted_results:
        overlap = any(
            candidate.start < k.end and candidate.end > k.start
            for k in kept
        )
        if not overlap:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# P3.3 — Audit log for PII redaction events
# ---------------------------------------------------------------------------

_DEFAULT_AUDIT_LOG = Path(__file__).parent.parent / "data" / "audit.jsonl"


def redaction_audit_log(
    query:      str,
    anonymizer: "PiiAnonymizer",
    intent:     str             = "unknown",
    trace_id:   str | None      = None,
    log_path:   Path | None     = None,
) -> None:
    """Append one JSONL entry to the PII redaction audit log.

    Must only be called when anonymizer.has_pii() is True.  The original
    query is NEVER written — only a SHA-256 hex digest.

    Fields written per entry:
      timestamp  — ISO-8601 UTC (e.g. "2026-05-06T10:32:14.123456+00:00")
      trace_id   — LangFuse / pipeline trace ID, or null
      pii_types  — sorted list of entity type names (e.g. ["EMAIL_ADDRESS", "ORDER_ID"])
      query_hash — SHA-256(original_query.encode("utf-8")), hex string
      intent     — intent class from the pipeline (e.g. "return_or_refund")

    The log file is created (including parent directories) if it does not exist.
    Under India's DPDPA this log constitutes a processing record under Section 6
    (notice obligation) and Section 8(6) (data accuracy / retention obligations).

    Args:
        query:      The ORIGINAL (pre-anonymization) query — used only for hashing.
        anonymizer: The PiiAnonymizer instance that processed this request.
        intent:     Intent label assigned by the pipeline classifier.
        trace_id:   Optional trace ID from LangFuse or the pipeline result dict.
        log_path:   Override the default audit log path (tests / CI use this).
    """
    if log_path is None:
        log_path = _DEFAULT_AUDIT_LOG

    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "trace_id":   trace_id,
        "pii_types":  anonymizer.get_pii_types(),
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "intent":     intent,
    }

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI — P3.1 demo
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "My email is priya@gmail.com, order ORD-445521",
    "Call me at +91 98765 43210 about my return",
    "I'm Rahul Mehta and I was charged twice for my order",
    "My account email is test.user@novusbank.com and order ORD-999888",
    "Please refund ORD-112233 and call +91 99001 12345 to confirm",
]

# ---------------------------------------------------------------------------
# P3.3 audit demo — 20 queries (10 PII + 10 clean) → exactly 10 log entries
# ---------------------------------------------------------------------------

_AUDIT_TEST_QUERIES = [
    # ---- 10 queries WITH PII ----
    ("return_or_refund",    "My email is priya@gmail.com — where is my refund for ORD-445521?"),
    ("billing_or_payment",  "Call me at +91 98765 43210 about the wrong charge on my loan"),
    ("order_status",        "I'm Rahul Mehta, please check order ORD-887766"),
    ("order_status",        "Track order ORD-112233 for test.user@novusbank.com"),
    ("return_or_refund",    "Anita Singh here — refund for ORD-334455 never arrived"),
    ("billing_or_payment",  "Card 4111111111111111 was charged twice — please reverse"),
    ("order_status",        "My number +91 77001 22334 — update on ORD-998877"),
    ("return_or_refund",    "Email me at customer.xyz@gmail.com with refund status ORD-667788"),
    ("general",             "Priya Sharma from Mumbai — my account is frozen"),
    ("membership",          "I'm Arjun Mehta, membership ID issue since March 2026"),
    # ---- 10 queries WITHOUT PII ----
    ("product_info",        "What is the minimum balance for a savings account?"),
    ("product_info",        "What is the interest rate on a personal loan?"),
    ("membership",          "What are the benefits of Elite membership?"),
    ("order_status",        "How long does it take to activate my savings account?"),
    ("billing_or_payment",  "What is the processing fee for a personal loan?"),
    ("product_info",        "Can I prepay my personal loan early?"),
    ("membership",          "What is the AQB for Novus Plus membership?"),
    ("billing_or_payment",  "Can I foreclose my loan after 10 EMIs?"),
    ("product_info",        "What documents are needed for a personal loan?"),
    ("general",             "What are the Novus Bank branch hours?"),
]


def _run_audit_test() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    audit_path = Path(__file__).parent.parent / "data" / "audit.jsonl"

    # Clear any existing file so count is clean for this run
    if audit_path.exists():
        audit_path.unlink()

    print("=== P3.3 — Redaction Audit Log demo (20 queries) ===\n")
    print(f"Audit log path: {audit_path}\n")

    logged = 0
    for intent, query in _AUDIT_TEST_QUERIES:
        anon = PiiAnonymizer()
        _    = anon.anonymize(query)
        pii  = anon.has_pii()

        if pii:
            redaction_audit_log(
                query=query,
                anonymizer=anon,
                intent=intent,
                trace_id=None,
                log_path=audit_path,
            )
            logged += 1

        status = "LOGGED  " if pii else "skipped "
        types  = anon.get_pii_types() if pii else []
        print(f"  [{status}] {query[:70]:<70}  {types}")

    print(f"\nTotal queries : 20  |  PII found : 10  |  Log entries : {logged}")
    assert logged == 10, f"Expected 10 log entries, got {logged}"

    print(f"\n--- audit.jsonl ({logged} entries) ---")
    with open(audit_path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            print(json.dumps(entry, indent=2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PII Anonymizer demo")
    parser.add_argument("--audit-test", action="store_true",
                        help="P3.3: run 20 queries and verify audit.jsonl has exactly 10 entries")
    args = parser.parse_args()

    if args.audit_test:
        _run_audit_test()
        return

    # Default: P3.1 demo
    print("=== P3.1 — Reversible PII Anonymizer demo (5 queries) ===\n")

    for i, query in enumerate(DEMO_QUERIES, 1):
        anon = PiiAnonymizer()            # fresh instance per request
        clean_query = anon.anonymize(query)

        # Simulate a raw LLM answer that echoes the anonymized placeholders
        # (in production the LLM only ever sees clean_query)
        raw_answer = (
            f"We have received your request. Details noted: {clean_query}. "
            "Our team will follow up within 24 hours."
        )
        restored_answer = anon.restore(raw_answer)

        print(f"[{i}] ORIGINAL  : {query}")
        print(f"    ANONYMIZED: {clean_query}")
        print(f"    RAW ANSWER: {raw_answer}")
        print(f"    RESTORED  : {restored_answer}")
        print(f"    PII map   : { {ph: orig for ph, orig in anon._map.items()} }")
        print(f"    pii_redacted: {anon.has_pii()}")
        print()


if __name__ == "__main__":
    main()
