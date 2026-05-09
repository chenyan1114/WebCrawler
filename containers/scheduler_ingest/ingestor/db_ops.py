from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from psycopg2.extras import execute_values
from sqlalchemy.orm import sessionmaker

from libs.scoring.golden_discovery_runtime import GoldenDiscoveryRuntimeScorer
from .selectdb_results import (
    CRAWL_RESULT_COLUMNS,
    SelectDBCrawlRecord,
    SelectDBCrawlResultWriter,
)


logger = logging.getLogger("ingestor.db_ops")


@dataclass
class IngestResult:
    new_link: bool = False
    domain_id: int | None = None
    is_ok: bool = False
    is_upd: bool = False
    fail_reason: str | None = None


BATCH_SIZE = 500

# Catches already-queued oversized urls that bypassed the spider-side filter.
MAX_URL_LEN = 2500

ROBOTS_UNKNOWN = 0
ROBOTS_ALLOWED = 1
ROBOTS_DISALLOWED = 2
ROBOTS_FAIL_PREFIX = "IgnoreRequest Forbidden by robots.txt"


# Fail reasons that are highly concentrated on specific domains: once a
# domain is failing for one of these, nearly all its URLs will too. Pause
# the whole domain instead of wasting fetches one URL at a time.
DOMAIN_PAUSE_BASE = {
    "IgnoreRequest Forbidden by robots.txt": "1 day",
    "ConnectionRefusedError":                "12 hours",
    "HttpError 410":                         "1 day",
    "NonHTML content-type":                  "6 hours",
    "HttpError 403":                         "6 hours",
    "HttpError 400":                         "6 hours",
    "HttpError 429":                         "1 hour",
    "TimeoutError":                          "1 hour",
    "ResponseNeverReceived":                 "1 hour",
}


def _domain_pause_base(reason: str | None) -> str | None:
    if reason in DOMAIN_PAUSE_BASE:
        return DOMAIN_PAUSE_BASE[reason]
    if reason and reason.startswith("HttpError 5"):
        return "30 minutes"
    return None


_CUR_INSERT_COLS = (
    "url",
    "domain_id",
    "last_fetch_ok",
    "last_content_update",
    "last_modified",
    "num_fetch_ok_90d",
    "num_fetch_fail_90d",
    "num_content_update_90d",
    "num_consecutive_fail",
    "last_fail_reason",
    "content_hash",
    "should_crawl",
    "title",
    "hreflang_count",
    "has_json_ld",
    "etag",
    "cache_control",
    "is_redirect",
    "redirect_hop_count",
    "robots_bits",
)

# snapshot_id / snapshot_at have DB defaults so they're omitted here.
_HIST_COLS = (
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
)

_SELECTDB_MARKER_COLS = (
    "is_selectdb_selected",
    "selectdb_score",
    "selectdb_run_id",
    "selectdb_selected_at",
    "selectdb_synced_at",
)


class IngestDB:
    def __init__(
        self,
        Session: sessionmaker,
        inline_ranker: GoldenDiscoveryRuntimeScorer | None = None,
        selectdb_writer: SelectDBCrawlResultWriter | None = None,
    ):
        self.Session = Session
        self.inline_ranker = inline_ranker
        self.selectdb_writer = selectdb_writer

    def _tcur(self, shard_id: int) -> str:
        return f"url_state_current_{shard_id:03d}"

    def _this(self, shard_id: int) -> str:
        return f"url_state_history_{shard_id:03d}"

    def _tevt(self, shard_id: int) -> str:
        return f"url_event_counter_{shard_id:03d}"

    def _score_new_links(self, urls: list[str]) -> tuple[list[float], datetime | None]:
        if not urls:
            return [], None
        if self.inline_ranker is None:
            return [0.0 for _ in urls], None
        scores = [float(score) for score in self.inline_ranker.score_many(urls)]
        if len(scores) != len(urls):
            raise RuntimeError(
                f"inline ranker returned {len(scores)} scores for {len(urls)} URLs"
            )
        return scores, datetime.now(timezone.utc)

    @staticmethod
    def _parse_optional_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _robots_bits(is_ok: bool, fail_reason: str | None) -> int:
        if fail_reason and fail_reason.startswith(ROBOTS_FAIL_PREFIX):
            return ROBOTS_DISALLOWED
        if is_ok or fail_reason == "NonHTML content-type" or (fail_reason or "").startswith("HttpError "):
            return ROBOTS_ALLOWED
        return ROBOTS_UNKNOWN

    @staticmethod
    def _split_unique_urls(items: list[tuple[int, dict]]) -> list[list[tuple[int, dict]]]:
        """Split into sub-batches with unique URLs; ON CONFLICT DO UPDATE
        refuses to touch the same row twice in one statement."""
        seen: set[str] = set()
        for _, rec in items:
            url = rec["url"]
            if url in seen:
                break
            seen.add(url)
        else:
            return [items]

        sub_batches: list[list[tuple[int, dict]]] = []
        current: list[tuple[int, dict]] = []
        seen = set()
        for idx, rec in items:
            url = rec["url"]
            if url in seen:
                sub_batches.append(current)
                current = [(idx, rec)]
                seen = {url}
            else:
                current.append((idx, rec))
                seen.add(url)
        if current:
            sub_batches.append(current)
        return sub_batches

    def _bulk_results_unique(
        self,
        cur,
        shard_id: int,
        items: list[tuple[int, dict]],
        selectdb_records: list[SelectDBCrawlRecord] | None = None,
    ) -> list[tuple[int, IngestResult]]:
        """Process one sub-batch of crawl results (unique URLs) for a shard."""
        if not items:
            return []

        tcur = self._tcur(shard_id)
        this = self._this(shard_id)
        tevt = self._tevt(shard_id)

        # Decode records and collect URLs needing an old-hash lookup in one pass.
        decoded = []
        check_urls = []
        for idx, rec in items:
            is_ok = rec["status"] == "ok"
            content_hash = rec.get("content_hash")
            inc_ok = int(is_ok)
            inc_fail = 1 - inc_ok
            url = rec["url"]
            decoded.append({
                "idx": idx,
                "url": url,
                "domain_id": int(rec["domain_id"]),
                "fetched_at": (
                    datetime.fromisoformat(rec["fetched_at"])
                    if rec.get("fetched_at")
                    else datetime.now(timezone.utc)
                ),
                "last_modified": self._parse_optional_datetime(rec.get("last_modified")),
                "fail_reason": rec.get("fail_reason"),
                "content_hash": content_hash,
                "is_ok": is_ok,
                "is_upd": False,
                "inc_ok": inc_ok,
                "inc_fail": inc_fail,
                "inc_upd": 0,
                "title": rec.get("title") if is_ok else None,
                "hreflang_count": rec.get("hreflang_count") if is_ok else None,
                "has_json_ld": rec.get("has_json_ld") if is_ok else None,
                "etag": rec.get("etag"),
                "cache_control": rec.get("cache_control"),
                "is_redirect": rec.get("is_redirect"),
                "redirect_hop_count": rec.get("redirect_hop_count"),
                "robots_bits": self._robots_bits(is_ok, rec.get("fail_reason")),
                "status": rec.get("status"),
                "fetched_at_raw": rec.get("fetched_at"),
                "content": rec.get("content"),
            })
            if is_ok and content_hash is not None:
                check_urls.append(url)

        if check_urls:
            cur.execute(
                f"SELECT url, content_hash FROM {tcur} WHERE url = ANY(%s)",
                (check_urls,),
            )
            old_hashes = dict(cur.fetchall())
            for d in decoded:
                if d["is_ok"] and d["content_hash"] is not None:
                    if old_hashes.get(d["url"]) != d["content_hash"]:
                        d["is_upd"] = True
                        d["inc_upd"] = 1

        # ON CONFLICT branch reads ok/upd from EXCLUDED.num_fetch_ok_90d /
        # num_content_update_90d so we avoid per-row CASE params in VALUES.
        upsert_rows = [
            (
                d["url"],
                d["domain_id"],
                d["fetched_at"] if d["is_ok"] else None,
                d["fetched_at"] if d["is_upd"] else None,
                d["last_modified"],
                d["inc_ok"],
                d["inc_fail"],
                d["inc_upd"],
                d["inc_fail"],  # num_consecutive_fail seed: 0 if ok else 1
                None if d["is_ok"] else d["fail_reason"],
                d["content_hash"],
                False,
                d["title"],
                d["hreflang_count"],
                d["has_json_ld"],
                d["etag"],
                d["cache_control"],
                d["is_redirect"],
                d["redirect_hop_count"],
                d["robots_bits"],
            )
            for d in decoded
        ]
        upsert_sql = f"""
        INSERT INTO {tcur} ({", ".join(_CUR_INSERT_COLS)})
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
          last_fetch_ok = COALESCE(EXCLUDED.last_fetch_ok, {tcur}.last_fetch_ok),
          last_content_update = COALESCE(EXCLUDED.last_content_update, {tcur}.last_content_update),
          num_fetch_ok_90d = {tcur}.num_fetch_ok_90d + EXCLUDED.num_fetch_ok_90d,
          num_fetch_fail_90d = {tcur}.num_fetch_fail_90d + EXCLUDED.num_fetch_fail_90d,
          num_content_update_90d = {tcur}.num_content_update_90d + EXCLUDED.num_content_update_90d,
          num_consecutive_fail = CASE
            WHEN EXCLUDED.num_fetch_ok_90d = 1 THEN 0
            ELSE {tcur}.num_consecutive_fail + 1
          END,
          last_fail_reason = EXCLUDED.last_fail_reason,
          content_hash = CASE
            WHEN EXCLUDED.num_content_update_90d = 1 THEN EXCLUDED.content_hash
            ELSE {tcur}.content_hash
          END,
          should_crawl = FALSE,
          title = COALESCE(EXCLUDED.title, {tcur}.title),
          hreflang_count = COALESCE(EXCLUDED.hreflang_count, {tcur}.hreflang_count),
          has_json_ld = COALESCE(EXCLUDED.has_json_ld, {tcur}.has_json_ld),
          last_modified = COALESCE(EXCLUDED.last_modified, {tcur}.last_modified),
          etag = COALESCE(EXCLUDED.etag, {tcur}.etag),
          cache_control = COALESCE(EXCLUDED.cache_control, {tcur}.cache_control),
          is_redirect = COALESCE(EXCLUDED.is_redirect, {tcur}.is_redirect),
          redirect_hop_count = COALESCE(EXCLUDED.redirect_hop_count, {tcur}.redirect_hop_count),
          robots_bits = CASE
            WHEN EXCLUDED.robots_bits = {ROBOTS_UNKNOWN} THEN {tcur}.robots_bits
            ELSE EXCLUDED.robots_bits
          END
        RETURNING {", ".join(_HIST_COLS + _SELECTDB_MARKER_COLS)}, (xmax = 0) AS inserted
        """
        returned = execute_values(
            cur, upsert_sql, upsert_rows, page_size=len(upsert_rows), fetch=True,
        )

        if returned:
            history_rows = [row[:len(_HIST_COLS)] for row in returned]
            execute_values(
                cur,
                f"INSERT INTO {this} ({', '.join(_HIST_COLS)}) VALUES %s",
                history_rows,
                page_size=len(history_rows),
            )

        if returned and selectdb_records is not None:
            decoded_by_url = {d["url"]: d for d in decoded}
            result_len = len(CRAWL_RESULT_COLUMNS)
            for row in returned:
                snapshot = dict(zip(CRAWL_RESULT_COLUMNS, row[:result_len]))
                if not snapshot.get("is_selectdb_selected"):
                    continue
                decoded_row = decoded_by_url.get(snapshot["url"])
                if decoded_row is None:
                    continue
                selectdb_records.append(
                    SelectDBCrawlRecord(
                        snapshot=snapshot,
                        shard_id=shard_id,
                        status=decoded_row.get("status"),
                        fetched_at=decoded_row.get("fetched_at_raw"),
                        content=decoded_row.get("content"),
                    )
                )

        counter_rows = [
            (d["url"], d["inc_ok"], d["inc_fail"], d["inc_upd"]) for d in decoded
        ]
        execute_values(
            cur,
            f"""
            INSERT INTO {tevt}
              (url, event_date, num_fetch_ok, num_fetch_fail, num_content_update, accounted)
            VALUES %s
            ON CONFLICT (url, event_date) DO UPDATE SET
              num_fetch_ok = {tevt}.num_fetch_ok + EXCLUDED.num_fetch_ok,
              num_fetch_fail = {tevt}.num_fetch_fail + EXCLUDED.num_fetch_fail,
              num_content_update = {tevt}.num_content_update + EXCLUDED.num_content_update
            """,
            counter_rows,
            template="(%s, CURRENT_DATE, %s, %s, %s, TRUE)",
            page_size=len(counter_rows),
        )

        # Bump domain pause for fail records on concentrated reasons, reset
        # for any ok record. `crawl_paused_until < NOW()` guard prevents
        # repeated fails from pushing the deadline indefinitely forward.
        pause_by_interval: dict[str, set[int]] = defaultdict(set)
        ok_domain_ids: set[int] = set()
        for d in decoded:
            if d["is_ok"]:
                ok_domain_ids.add(d["domain_id"])
                continue
            base = _domain_pause_base(d["fail_reason"])
            if base:
                pause_by_interval[base].add(d["domain_id"])
        for interval, dids in pause_by_interval.items():
            cur.execute(
                f"""
                UPDATE domain_state
                SET domain_fail_count = domain_fail_count + 1,
                    crawl_paused_until = NOW() + (INTERVAL '{interval}' * POWER(2, LEAST(domain_fail_count, 6)))
                WHERE domain_id = ANY(%s)
                  AND (crawl_paused_until IS NULL OR crawl_paused_until < NOW())
                """,
                (list(dids),),
            )
        if ok_domain_ids:
            cur.execute(
                """
                UPDATE domain_state
                SET domain_fail_count = 0, crawl_paused_until = NULL
                WHERE domain_id = ANY(%s) AND domain_fail_count > 0
                """,
                (list(ok_domain_ids),),
            )

        # xmax = 0 in RETURNING means the row was inserted, not updated.
        inserted_flags = {row[0]: row[-1] for row in returned}
        return [
            (d["idx"], IngestResult(
                new_link=bool(inserted_flags.get(d["url"], False)),
                domain_id=d["domain_id"],
                is_ok=d["is_ok"],
                is_upd=d["is_upd"],
                fail_reason=d["fail_reason"],
            ))
            for d in decoded
        ]

    def _bulk_results(
        self,
        cur,
        shard_id: int,
        items: list[tuple[int, dict]],
        selectdb_records: list[SelectDBCrawlRecord] | None = None,
    ) -> list[tuple[int, IngestResult]]:
        results: list[tuple[int, IngestResult]] = []
        for sub in self._split_unique_urls(items):
            results.extend(
                self._bulk_results_unique(
                    cur,
                    shard_id,
                    sub,
                    selectdb_records=selectdb_records,
                )
            )
        return results

    def _bulk_links(
        self, cur, shard_id: int, items: list[tuple[int, dict]],
    ) -> list[tuple[int, bool]]:
        if not items:
            return []

        tcur = self._tcur(shard_id)
        this = self._this(shard_id)

        by_url: dict[str, dict] = {}
        first_idx_by_url: dict[str, int] = {}
        dup_results: list[tuple[int, bool]] = []
        for idx, rec in items:
            url = rec["url"]
            inc_approx = int(rec.get("inlink_count_approx", 1))
            inc_external = int(rec.get("inlink_count_external", 0))
            if url in by_url:
                by_url[url]["inlink_count_approx"] += inc_approx
                by_url[url]["inlink_count_external"] += inc_external
                dup_results.append((idx, False))
            else:
                first_idx_by_url[url] = idx
                by_url[url] = {
                    **rec,
                    "inlink_count_approx": inc_approx,
                    "inlink_count_external": inc_external,
                }

        unique_items = [(first_idx_by_url[url], rec) for url, rec in by_url.items()]

        score_urls = [rec["url"] for _, rec in unique_items]
        scores, score_updated_at = self._score_new_links(score_urls)

        link_rows = []
        for (_, rec), url_score in zip(unique_items, scores):
            link_rows.append(
                (
                    rec["url"],
                    int(rec["domain_id"]),
                    float(rec.get("domain_score", 0.0)),
                    float(url_score),
                    score_updated_at,
                    rec.get("discovered_from"),
                    int(rec.get("discovery_source_type", 0)),
                    rec.get("parent_page_score"),
                    int(rec.get("inlink_count_approx", 0)),
                    int(rec.get("inlink_count_external", 0)),
                    (rec.get("anchor_text") or None),
                )
            )
        inserted_rows = execute_values(
            cur,
            f"""
            INSERT INTO {tcur}
              (url, domain_id, domain_score, url_score, url_score_updated_at, discovered_from,
               discovery_source_type, parent_page_score,
               inlink_count_approx, inlink_count_external, anchor_text)
            VALUES %s
            ON CONFLICT (url) DO UPDATE SET
              inlink_count_approx = {tcur}.inlink_count_approx + EXCLUDED.inlink_count_approx,
              inlink_count_external = {tcur}.inlink_count_external + EXCLUDED.inlink_count_external,
              anchor_text = COALESCE({tcur}.anchor_text, EXCLUDED.anchor_text)
            RETURNING url, (xmax = 0) AS inserted
            """,
            link_rows,
            page_size=len(link_rows),
            fetch=True,
        )
        inserted_set = {row[0] for row in inserted_rows if row[1]}

        history_rows = [r for r in link_rows if r[0] in inserted_set]
        if history_rows:
            execute_values(
                cur,
                f"""
                INSERT INTO {this}
                  (url, domain_id, domain_score, url_score, url_score_updated_at, discovered_from,
                   discovery_source_type, parent_page_score,
                   inlink_count_approx, inlink_count_external, anchor_text)
                VALUES %s
                """,
                history_rows,
                page_size=len(history_rows),
            )

        results = [(idx, rec["url"] in inserted_set) for idx, rec in unique_items]
        results.extend(dup_results)
        return results

    @staticmethod
    def aggregate_links(recs: list[dict]) -> list[dict]:
        """Collapse discovery records by url: sum inlink counters, keep first
        non-null anchor. _bulk_links only dedups within one BATCH_SIZE chunk;
        folder-level dedup is what actually cuts the ~12x hot-url contention.
        """
        by_url: dict[str, dict] = {}
        for rec in recs:
            url = rec.get("url")
            if not url:
                continue
            inc_a = int(rec.get("inlink_count_approx", 1))
            inc_e = int(rec.get("inlink_count_external", 0))
            existing = by_url.get(url)
            if existing is None:
                by_url[url] = {**rec, "inlink_count_approx": inc_a, "inlink_count_external": inc_e}
            else:
                existing["inlink_count_approx"] += inc_a
                existing["inlink_count_external"] += inc_e
                if not existing.get("anchor_text") and rec.get("anchor_text"):
                    existing["anchor_text"] = rec["anchor_text"]
        return list(by_url.values())

    def process_batch(self, recs: list[dict]) -> list[IngestResult | bool | None]:
        """Group records by (kind, shard_id) and dispatch to bulk paths.

        All-or-nothing: any failure rolls back and re-raises.
        Returns IngestResult for results, bool for new links, in input order.
        """
        results: list[IngestResult | bool | None] = [None] * len(recs)
        selectdb_records: list[SelectDBCrawlRecord] = []

        results_by_shard: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        links_by_shard: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for i, rec in enumerate(recs):
            if len(rec.get("url", "")) > MAX_URL_LEN:
                continue
            sid = int(rec["shard_id"])
            if rec.get("status") == "new":
                links_by_shard[sid].append((i, rec))
            else:
                results_by_shard[sid].append((i, rec))

        with self.Session.begin() as sess:
            with sess.connection().connection.cursor() as cur:
                # Links first so that a URL that appears as both a discovered
                # link and a crawl result in the same batch is inserted with
                # its domain_score before the result UPSERT touches it.
                for sid, items in links_by_shard.items():
                    for idx, ok in self._bulk_links(cur, sid, items):
                        results[idx] = ok
                for sid, items in results_by_shard.items():
                    for idx, ir in self._bulk_results(
                        cur,
                        sid,
                        items,
                        selectdb_records=selectdb_records,
                    ):
                        results[idx] = ir
        if self.selectdb_writer is not None and selectdb_records:
            try:
                self.selectdb_writer.upsert_many(selectdb_records)
            except Exception as exc:
                logger.error(
                    "selectdb_results.upsert_error",
                    extra={
                        "event": "selectdb_results.upsert_error",
                        "error": str(exc),
                        "count": len(selectdb_records),
                    },
                )
        return results
