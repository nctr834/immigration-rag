"""Inspect the chunks ingest.py wrote to pgvector: are they junk? In one command.

Reuses the same DB connection config as ingest/retrieve (config.PG), so it always
targets the table you actually populated. Run it after every `python src/ingest.py`.

Usage:
    python scripts/inspect_chunks.py            # full report, 3 random samples
    python scripts/inspect_chunks.py -n 5       # 5 random samples
    python scripts/inspect_chunks.py --shortest # sample the shortest chunks instead
    python scripts/inspect_chunks.py --grep I-864   # only sample chunks matching a string

Exit code is non-zero if any junk/duplicate check trips, so it doubles as a
post-ingest smoke test in CI or a pre-commit hook.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg2

# Import the project's shared config the same way the other entry points do.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from config import PG, PG_TABLE_NAME  # noqa: E402

# LlamaIndex prefixes the configured name with "data_" for the actual table.
TABLE = f"data_{PG_TABLE_NAME}"

# A chunk shorter than this is almost certainly a stray fragment, not content.
SHORT_CHUNK_CHARS = 100


def _connect():
    """Open a psycopg2 connection from the same params ingest uses."""
    try:
        return psycopg2.connect(
            host=PG["host"],
            port=PG["port"],
            dbname=PG["database"],
            user=PG["user"],
            password=PG["password"],
        )
    except psycopg2.OperationalError as e:
        sys.exit(
            f"Could not connect to Postgres at {PG['host']}:{PG['port']}/"
            f"{PG['database']}.\nIs the pgvector container running?\n\n{e}"
        )


def _table_exists(cur) -> bool:
    cur.execute("SELECT to_regclass(%s)", (TABLE,))
    return cur.fetchone()[0] is not None


def report(cur, n_samples: int, shortest: bool, grep: str | None) -> int:
    """Print the inspection report. Return the number of failed checks."""
    failures = 0

    # --- Count ---
    cur.execute(f"SELECT count(*) FROM {TABLE}")
    total = cur.fetchone()[0]
    print(f"\n=== {TABLE} ===")
    print(f"total chunks: {total}")
    if total == 0:
        print("  [FAIL] table is empty: did ingest run?")
        return 1

    # --- Length distribution ---
    cur.execute(
        f"""SELECT min(length(text)), round(avg(length(text))),
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY length(text)),
                   max(length(text))
            FROM {TABLE}"""
    )
    lo, avg, median, hi = cur.fetchone()
    print(f"length (chars): min={lo}  avg={int(avg)}  median={int(median)}  max={hi}")

    # --- Per-file spread ---
    cur.execute(
        f"""SELECT metadata_->>'file_name', count(*)
            FROM {TABLE} GROUP BY 1 ORDER BY 2 DESC"""
    )
    print("\nchunks per source file:")
    for fname, cnt in cur.fetchall():
        print(f"  {cnt:>5}  {fname}")

    # --- Checks (each can fail the run) ---
    print("\nchecks:")

    cur.execute(f"SELECT count(*) FROM {TABLE} WHERE length(text) < %s", (SHORT_CHUNK_CHARS,))
    short = cur.fetchone()[0]
    failures += _check(f"chunks < {SHORT_CHUNK_CHARS} chars", short)

    cur.execute(
        f"""SELECT count(*) FROM (
                SELECT 1 FROM {TABLE} GROUP BY text HAVING count(*) > 1
            ) d"""
    )
    dupes = cur.fetchone()[0]
    failures += _check("duplicate chunk texts", dupes)

    cur.execute(f"SELECT count(*) FROM {TABLE} WHERE text ~ E'\\\\x00'")
    failures += _check("chunks with NUL bytes", cur.fetchone()[0])

    cur.execute(f"SELECT count(*) FROM {TABLE} WHERE text !~ '[a-zA-Z]{{3,}}'")
    failures += _check("chunks with no real words", cur.fetchone()[0])

    # Known PDF boilerplate that _clean_text() is supposed to strip.
    cur.execute(f"SELECT count(*) FROM {TABLE} WHERE text ~* 'Not for\\s*\\n\\s*Production'")
    failures += _check("residual draft watermarks", cur.fetchone()[0])

    cur.execute(f"SELECT count(*) FROM {TABLE} WHERE text ~ 'Page\\s+\\d+\\s+of\\s+\\d+'")
    failures += _check("residual page headers", cur.fetchone()[0])

    # --- Samples (eyeballing is the only check that finds bad boundaries) ---
    order = "length(text) ASC" if shortest else "random()"
    label = "shortest" if shortest else "random"
    params: tuple = ()
    where = ""
    if grep:
        where = "WHERE text ILIKE %s"
        params = (f"%{grep}%",)
    cur.execute(
        f"""SELECT metadata_->>'file_name', length(text), text
            FROM {TABLE} {where} ORDER BY {order} LIMIT %s""",
        params + (n_samples,),
    )
    rows = cur.fetchall()
    grep_note = f" matching '{grep}'" if grep else ""
    print(f"\n=== {len(rows)} {label} sample(s){grep_note} ===")
    for fname, length, text in rows:
        print(f"\n--- [{fname} | {length} chars] " + "-" * 30)
        print(text.strip())

    print(
        f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) failed'}"
    )
    return failures


def _check(label: str, count: int) -> int:
    """Print one check line; return 1 if it failed (non-zero count)."""
    ok = count == 0
    print(f"  {'[ok]' if ok else '[FAIL]'} {label}: {count}")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=3, help="number of sample chunks to print")
    p.add_argument(
        "--shortest",
        action="store_true",
        help="sample the shortest chunks instead of random ones",
    )
    p.add_argument("--grep", metavar="STR", help="only sample chunks containing STR")
    args = p.parse_args()

    conn = _connect()
    try:
        with conn.cursor() as cur:
            if not _table_exists(cur):
                sys.exit(f"Table {TABLE} does not exist. Run `python src/ingest.py` first.")
            failures = report(cur, args.n, args.shortest, args.grep)
    finally:
        conn.close()

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
