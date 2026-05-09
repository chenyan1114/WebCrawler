from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from .base import SelectionStrategy


class GoldenDiscoveryRankerV1Strategy(SelectionStrategy):
    """Offerer strategy for Golden Discovery Ranker v1 production scheduling.

    The ranker writes operational priority into url_score.
    url_score_updated_at is used only to prefer rows already refreshed by the
    background scorer; there is no separate score-version or experiment
    metadata in the scheduling path.
    """

    def __init__(self, Session: sessionmaker):
        self.Session = Session

    def _table(self, shard_id: int) -> str:
        return f"url_state_current_{shard_id:03d}"

    def _event_table(self, shard_id: int) -> str:
        return f"url_event_counter_{shard_id:03d}"

    def select_by_domain(
        self,
        shard_id: int,
        exclude_domain_ids: set[int],
        per_domain_cap: int,
        max_domains: int,
    ) -> dict[int, list[str]]:
        if max_domains <= 0 or per_domain_cap <= 0:
            return {}

        table = self._table(shard_id)
        event_table = self._event_table(shard_id)

        exclude_clause = ""
        params: dict = {
            "max_domains": max_domains,
            "per_domain_cap": per_domain_cap,
        }
        if exclude_domain_ids:
            exclude_clause = "AND domain_id NOT IN :exclude"
            params["exclude"] = tuple(exclude_domain_ids)

        sql = text(f"""
        WITH eligible_domains AS (
            SELECT
                domain_id,
                MAX(CASE WHEN is_selectdb_selected THEN 1 ELSE 0 END) AS has_selectdb_selected,
                MAX(CASE WHEN is_selectdb_selected THEN selectdb_score END) AS best_selectdb_score,
                MAX(CASE WHEN url_score_updated_at IS NOT NULL THEN url_score END) AS best_golden_discovery_score,
                MAX(url_score) AS best_any_score,
                MAX(domain_score) AS best_domain_score,
                MIN(first_seen) AS oldest_first_seen
            FROM {table}
            WHERE should_crawl = TRUE
              {exclude_clause}
              AND NOT EXISTS (
                SELECT 1 FROM domain_state d
                WHERE d.domain_id = {table}.domain_id
                  AND d.crawl_paused_until > NOW()
              )
            GROUP BY domain_id
            ORDER BY
                has_selectdb_selected DESC,
                best_selectdb_score DESC NULLS LAST,
                best_golden_discovery_score DESC NULLS LAST,
                best_any_score DESC NULLS LAST,
                best_domain_score DESC NULLS LAST,
                oldest_first_seen ASC NULLS LAST,
                domain_id
            LIMIT :max_domains
        ),
        picked AS (
            SELECT u.url, u.domain_id
            FROM eligible_domains d,
            LATERAL (
                SELECT url, domain_id
                FROM {table}
                WHERE should_crawl = TRUE AND domain_id = d.domain_id
                ORDER BY
                    CASE WHEN is_selectdb_selected THEN 0 ELSE 1 END,
                    selectdb_score DESC NULLS LAST,
                    CASE WHEN url_score_updated_at IS NULL THEN 1 ELSE 0 END,
                    url_score DESC NULLS LAST,
                    domain_score DESC NULLS LAST,
                    last_scheduled ASC NULLS FIRST,
                    first_seen ASC
                LIMIT :per_domain_cap
                FOR UPDATE SKIP LOCKED
            ) u
        ),
        updated AS (
            UPDATE {table} x
            SET
                should_crawl = FALSE,
                last_scheduled = CURRENT_TIMESTAMP,
                num_scheduled_90d = x.num_scheduled_90d + 1
            FROM picked
            WHERE x.url = picked.url
            RETURNING x.url, x.domain_id
        ),
        event_upsert AS (
            INSERT INTO {event_table} (url, event_date, num_scheduled, accounted)
            SELECT url, CURRENT_DATE, 1, TRUE
            FROM updated
            ON CONFLICT (url, event_date)
            DO UPDATE SET
                num_scheduled = {event_table}.num_scheduled + 1,
                accounted = TRUE
        )
        SELECT u.url, u.domain_id
        FROM updated u;
        """)

        with self.Session() as sess:
            rows = sess.execute(sql, params).fetchall()
            sess.commit()

        result: dict[int, list[str]] = defaultdict(list)
        for r in rows:
            result[r.domain_id].append(r.url)
        return dict(result)
