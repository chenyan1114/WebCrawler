from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg2.extras import execute_values
from sqlalchemy.orm import sessionmaker

try:
    from scripts.migrate_add_selectdb_integration import SELECTDB_RESULT_SCHEMA_SQL
except ModuleNotFoundError:
    SELECTDB_RESULT_SCHEMA_SQL = ""


logger = logging.getLogger("selectdb_results")


CRAWL_RESULT_COLUMNS: tuple[str, ...] = (
    "url",
    "domain_id",
    "first_seen",
    "last_scheduled",
    "last_fetch_ok",
    "last_content_update",
    "last_modified",
    "num_scheduled_90d",
    "num_fetch_ok_90d",
    "num_fetch_fail_90d",
    "num_content_update_90d",
    "num_consecutive_fail",
    "last_fail_reason",
    "content_hash",
    "should_crawl",
    "url_score",
    "url_score_updated_at",
    "domain_score",
    "source",
    "discovered_from",
    "title",
    "hreflang_count",
    "has_json_ld",
    "etag",
    "cache_control",
    "is_redirect",
    "redirect_hop_count",
    "discovery_source_type",
    "parent_page_score",
    "inlink_count_approx",
    "inlink_count_external",
    "anchor_text",
    "robots_bits",
    "is_selectdb_selected",
    "selectdb_score",
    "selectdb_run_id",
    "selectdb_selected_at",
    "selectdb_synced_at",
)


@dataclass(frozen=True)
class SelectDBCrawlRecord:
    snapshot: dict[str, Any]
    shard_id: int
    status: str | None
    fetched_at: str | None
    content: str | None


class SelectDBCrawlResultWriter:
    def __init__(
        self,
        Session: sessionmaker,
        html_dir: str | Path,
        *,
        ensure_schema: bool = True,
    ):
        self.Session = Session
        self.html_dir = Path(html_dir)
        if ensure_schema:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        if not SELECTDB_RESULT_SCHEMA_SQL:
            return
        with self.Session.begin() as sess:
            with sess.connection().connection.cursor() as cur:
                cur.execute(SELECTDB_RESULT_SCHEMA_SQL)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _html_path(self, *, url: str, fetched_at: str | None, shard_id: int) -> Path:
        dt = self._parse_datetime(fetched_at)
        digest = hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()
        day = dt.strftime("%Y%m%d")
        ts = dt.strftime("%Y%m%dT%H%M%SZ")
        return self.html_dir / day / f"shard_{shard_id:03d}" / f"{digest}_{ts}.html"

    def _write_html(
        self,
        *,
        url: str,
        content: str | None,
        fetched_at: str | None,
        shard_id: int,
        status: str | None,
    ) -> str | None:
        if status != "ok" or not isinstance(content, str):
            return None
        path = self._html_path(url=url, fetched_at=fetched_at, shard_id=shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8", errors="replace")
        tmp.replace(path)
        return str(path)

    @staticmethod
    def _upsert_sql() -> str:
        columns = (*CRAWL_RESULT_COLUMNS, "html_path")
        update_cols = [c for c in CRAWL_RESULT_COLUMNS if c != "url"]
        assignments = [f"{c} = EXCLUDED.{c}" for c in update_cols]
        assignments.append(
            "html_path = COALESCE(EXCLUDED.html_path, selected_url_crawl_results.html_path)"
        )
        assignments.append("crawl_synced_at = NOW()")
        return f"""
        INSERT INTO public.selected_url_crawl_results ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
          {", ".join(assignments)}
        """

    def upsert_many(self, records: list[SelectDBCrawlRecord]) -> int:
        if not records:
            return 0

        rows = []
        for rec in records:
            snapshot = rec.snapshot
            url = snapshot["url"]
            html_path = self._write_html(
                url=url,
                content=rec.content,
                fetched_at=rec.fetched_at,
                shard_id=rec.shard_id,
                status=rec.status,
            )
            rows.append(
                tuple(snapshot.get(col) for col in CRAWL_RESULT_COLUMNS) + (html_path,)
            )

        with self.Session.begin() as sess:
            with sess.connection().connection.cursor() as cur:
                execute_values(
                    cur,
                    self._upsert_sql(),
                    rows,
                    page_size=min(len(rows), 1000),
                )

        logger.info(
            "selectdb_results.upserted",
            extra={"event": "selectdb_results.upserted", "count": len(rows)},
        )
        return len(rows)
