"""
Migration: add selectdb scheduling fields to crawlerdb and crawl result table
to selectdb.

Crawlerdb remains the only scheduling transaction source. The fields added to
url_state_current_* mirror the current selectdb selection snapshot so offerers
can prioritize selected URLs without cross-database transactions.

Usage:
    python scripts/migrate_add_selectdb_integration.py --dry-run
    python scripts/migrate_add_selectdb_integration.py
"""

from __future__ import annotations

import argparse
import logging

import psycopg2

try:
    from scripts.constants import CRAWLERDB, SELECTDB, NUM_SHARDS
except ModuleNotFoundError:
    from constants import CRAWLERDB, SELECTDB, NUM_SHARDS


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CURRENT_PREFIX = "url_state_current"


SELECTDB_CRAWLER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("is_selectdb_selected", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("selectdb_score", "DOUBLE PRECISION"),
    ("selectdb_run_id", "BIGINT"),
    ("selectdb_selected_at", "TIMESTAMPTZ"),
    ("selectdb_synced_at", "TIMESTAMPTZ"),
)


def current_table(shard_id: int) -> str:
    return f"{CURRENT_PREFIX}_{shard_id:03d}"


def iter_current_tables(num_shards: int = NUM_SHARDS):
    for shard_id in range(num_shards):
        yield current_table(shard_id)


def add_selectdb_column_sql(table: str, name: str, definition: str) -> str:
    return f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {definition}"


def selectdb_priority_index_name(shard_id: int) -> str:
    return f"idx_url_state_current_{shard_id:03d}_selectdb_priority"


def create_selectdb_priority_index_sql(shard_id: int) -> str:
    table = current_table(shard_id)
    return (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        f"{selectdb_priority_index_name(shard_id)} "
        f"ON {table} ("
        "is_selectdb_selected DESC, "
        "selectdb_score DESC NULLS LAST, "
        "domain_id, "
        "last_scheduled ASC NULLS FIRST, "
        "first_seen ASC"
        ") WHERE should_crawl = TRUE"
    )


SELECTDB_RESULT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS public.selected_url_crawl_results (
    url TEXT PRIMARY KEY,
    domain_id BIGINT NOT NULL,
    first_seen TIMESTAMPTZ,
    last_scheduled TIMESTAMPTZ,
    last_fetch_ok TIMESTAMPTZ,
    last_content_update TIMESTAMPTZ,
    last_modified TIMESTAMPTZ,
    num_scheduled_90d INTEGER,
    num_fetch_ok_90d INTEGER,
    num_fetch_fail_90d INTEGER,
    num_content_update_90d INTEGER,
    num_consecutive_fail INTEGER,
    last_fail_reason TEXT,
    content_hash TEXT,
    should_crawl BOOLEAN,
    url_score DOUBLE PRECISION,
    url_score_updated_at TIMESTAMPTZ,
    domain_score DOUBLE PRECISION,
    source SMALLINT,
    discovered_from TEXT,
    title TEXT,
    hreflang_count INTEGER,
    has_json_ld BOOLEAN,
    etag TEXT,
    cache_control TEXT,
    is_redirect BOOLEAN,
    redirect_hop_count SMALLINT,
    discovery_source_type SMALLINT,
    parent_page_score DOUBLE PRECISION,
    inlink_count_approx INTEGER,
    inlink_count_external INTEGER,
    anchor_text TEXT,
    robots_bits SMALLINT,
    is_selectdb_selected BOOLEAN NOT NULL DEFAULT TRUE,
    selectdb_score DOUBLE PRECISION,
    selectdb_run_id BIGINT,
    selectdb_selected_at TIMESTAMPTZ,
    selectdb_synced_at TIMESTAMPTZ,
    html_path TEXT,
    crawl_synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_selected_url_crawl_results_run_id
    ON public.selected_url_crawl_results (selectdb_run_id);

CREATE INDEX IF NOT EXISTS idx_selected_url_crawl_results_fetch_ok
    ON public.selected_url_crawl_results (last_fetch_ok);
"""


def add_crawlerdb_columns(conn, dry_run: bool) -> int:
    count = 0
    with conn.cursor() as cur:
        for table in iter_current_tables():
            for name, definition in SELECTDB_CRAWLER_COLUMNS:
                sql = add_selectdb_column_sql(table, name, definition)
                count += 1
                if dry_run:
                    log.info("[DRY-RUN] %s", sql)
                else:
                    cur.execute(sql)
    if not dry_run:
        conn.commit()
    return count


def create_crawlerdb_indexes(conn, dry_run: bool) -> int:
    count = 0
    previous_autocommit = conn.autocommit
    if not dry_run:
        conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for shard_id in range(NUM_SHARDS):
                sql = create_selectdb_priority_index_sql(shard_id)
                count += 1
                if dry_run:
                    log.info("[DRY-RUN] %s", sql)
                else:
                    cur.execute(sql)
    finally:
        if not dry_run:
            conn.autocommit = previous_autocommit
    return count


def ensure_selectdb_result_schema(conn, dry_run: bool) -> None:
    if dry_run:
        log.info("[DRY-RUN] %s", SELECTDB_RESULT_SCHEMA_SQL.strip())
        return
    with conn.cursor() as cur:
        cur.execute(SELECTDB_RESULT_SCHEMA_SQL)
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add selectdb integration fields and crawl result table"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    crawler_conn = psycopg2.connect(**CRAWLERDB)
    select_conn = psycopg2.connect(**SELECTDB)
    try:
        column_count = add_crawlerdb_columns(crawler_conn, args.dry_run)
        index_count = create_crawlerdb_indexes(crawler_conn, args.dry_run)
        ensure_selectdb_result_schema(select_conn, args.dry_run)
        log.info(
            "selectdb integration migration complete: columns=%s indexes=%s dry_run=%s",
            column_count,
            index_count,
            args.dry_run,
        )
    finally:
        crawler_conn.close()
        select_conn.close()


if __name__ == "__main__":
    main()
