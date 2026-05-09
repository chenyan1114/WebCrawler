from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from containers.scheduler_control.offerer.selection.golden_discovery_ranker_v1_strategy import (
    GoldenDiscoveryRankerV1Strategy,
)
from containers.scheduler_control.scorer import service as scorer_service
from containers.scheduler_control.scorer.service import (
    GoldenDiscoveryRankerConfig,
    GoldenDiscoveryRankerService,
)
from containers.scheduler_ingest.ingestor import db_ops as ingest_db_ops
from containers.scheduler_ingest.ingestor.db_ops import IngestDB
from containers.scheduler_ingest.ingestor.selectdb_results import SelectDBCrawlResultWriter
from scripts import migrate_add_url_score_updated_at as migration
from scripts import migrate_add_selectdb_integration as selectdb_migration


class GoldenDiscoveryMigrationSqlTest(unittest.TestCase):
    def test_unscored_index_is_partial_current_shard_index(self):
        sql = migration.create_golden_discovery_unscored_index_sql(7)

        self.assertIn(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_url_state_current_007_golden_discovery_v1_unscored",
            sql,
        )
        self.assertIn(
            "ON url_state_current_007 (first_seen ASC NULLS LAST)",
            sql,
        )
        self.assertIn(
            "WHERE should_crawl = TRUE AND url_score_updated_at IS NULL",
            sql,
        )
        self.assertNotIn("url_state_history", sql)

    def test_selection_index_is_partial_current_shard_index(self):
        sql = migration.create_golden_discovery_selection_index_sql(7)

        self.assertIn(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_url_state_current_007_golden_discovery_v1_selection",
            sql,
        )
        self.assertIn(
            "ON url_state_current_007 (domain_id, "
            "((CASE WHEN url_score_updated_at IS NULL THEN 1 ELSE 0 END)), "
            "url_score DESC NULLS LAST, "
            "domain_score DESC NULLS LAST, "
            "last_scheduled ASC NULLS FIRST, "
            "first_seen ASC)",
            sql,
        )
        self.assertIn("WHERE should_crawl = TRUE", sql)
        self.assertNotIn("url_state_history", sql)

    def test_column_migration_targets_current_and_history_shards(self):
        tables = list(migration.iter_state_tables(num_shards=2))

        self.assertEqual(
            tables,
            [
                "url_state_current_000",
                "url_state_current_001",
                "url_state_history_000",
                "url_state_history_001",
            ],
        )
        self.assertEqual(
            migration.add_url_score_updated_at_sql("url_state_current_000"),
            "ALTER TABLE url_state_current_000 "
            "ADD COLUMN IF NOT EXISTS url_score_updated_at TIMESTAMPTZ",
        )

    def test_selectdb_integration_migration_adds_priority_columns(self):
        sql = selectdb_migration.add_selectdb_column_sql(
            "url_state_current_003",
            "is_selectdb_selected",
            "BOOLEAN NOT NULL DEFAULT FALSE",
        )
        self.assertEqual(
            sql,
            "ALTER TABLE url_state_current_003 "
            "ADD COLUMN IF NOT EXISTS is_selectdb_selected BOOLEAN NOT NULL DEFAULT FALSE",
        )

        index_sql = selectdb_migration.create_selectdb_priority_index_sql(3)
        self.assertIn("idx_url_state_current_003_selectdb_priority", index_sql)
        self.assertIn("is_selectdb_selected DESC", index_sql)
        self.assertIn("selectdb_score DESC NULLS LAST", index_sql)
        self.assertIn("WHERE should_crawl = TRUE", index_sql)

    def test_selectdb_result_schema_has_html_path(self):
        schema_sql = selectdb_migration.SELECTDB_RESULT_SCHEMA_SQL
        self.assertIn("public.selected_url_crawl_results", schema_sql)
        self.assertIn("html_path TEXT", schema_sql)
        self.assertIn("crawl_synced_at TIMESTAMPTZ", schema_sql)


class _FakeResult:
    def fetchall(self):
        return [SimpleNamespace(domain_id=7, url="https://example.com/a")]


class _FakeOffererSession:
    def __init__(self):
        self.calls = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.calls.append((str(sql), params))
        return _FakeResult()

    def commit(self):
        self.committed = True


class _FakeOffererSessionFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


class GoldenDiscoveryOffererStrategyTest(unittest.TestCase):
    def test_strategy_prefers_refreshed_ranker_scores_in_sql(self):
        session = _FakeOffererSession()
        strategy = GoldenDiscoveryRankerV1Strategy(Session=_FakeOffererSessionFactory(session))

        selected = strategy.select_by_domain(
            shard_id=3,
            exclude_domain_ids={11, 22},
            per_domain_cap=2,
            max_domains=5,
        )

        self.assertEqual(selected, {7: ["https://example.com/a"]})
        self.assertTrue(session.committed)

        sql, params = session.calls[0]
        self.assertIn("FROM url_state_current_003", sql)
        self.assertIn(
            "MAX(CASE WHEN url_score_updated_at IS NOT NULL THEN url_score END)",
            sql,
        )
        self.assertIn("MAX(CASE WHEN is_selectdb_selected THEN 1 ELSE 0 END)", sql)
        self.assertIn("MAX(CASE WHEN is_selectdb_selected THEN selectdb_score END)", sql)
        self.assertIn("has_selectdb_selected DESC", sql)
        self.assertIn("best_selectdb_score DESC NULLS LAST", sql)
        self.assertNotIn("WHEN MAX(CASE", sql)
        self.assertIn("NOT EXISTS", sql)
        self.assertIn("FROM domain_state d", sql)
        self.assertIn("d.crawl_paused_until > NOW()", sql)
        self.assertIn("CASE WHEN is_selectdb_selected THEN 0 ELSE 1 END", sql)
        self.assertIn("selectdb_score DESC NULLS LAST", sql)
        self.assertIn(
            "CASE WHEN url_score_updated_at IS NULL THEN 1 ELSE 0 END",
            sql,
        )
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertEqual(params["exclude"], (11, 22))
        self.assertEqual(params["per_domain_cap"], 2)
        self.assertEqual(params["max_domains"], 5)


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return [
            ("https://example.com/a",),
            ("https://example.com/b",),
        ]


class _FakeRawConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _FakeSqlAlchemyConnection:
    def __init__(self, cursor):
        self.connection = _FakeRawConnection(cursor)


class _FakeScorerSession:
    def __init__(self, cursor):
        self._cursor = cursor

    def connection(self):
        return _FakeSqlAlchemyConnection(self._cursor)


class _FakeBeginContext:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeScorerSessionFactory:
    def __init__(self, cursor):
        self.session = _FakeScorerSession(cursor)

    def begin(self):
        return _FakeBeginContext(self.session)


class _FakeScorer:
    def score_many(self, urls):
        self.urls = urls
        return [0.25, 0.75]


class GoldenDiscoveryRankerServiceTest(unittest.TestCase):
    def test_scorer_claims_unscored_urls_and_updates_timestamp(self):
        cursor = _FakeCursor()
        fake_scorer = _FakeScorer()
        cfg = GoldenDiscoveryRankerConfig(
            total_shards=256,
            num_workers=4,
            worker_id=0,
            batch_size=1000,
            scan_interval_sec=60,
            max_batches_per_shard=1,
        )
        service = GoldenDiscoveryRankerService(
            cfg=cfg,
            Session=_FakeScorerSessionFactory(cursor),
            scorer=fake_scorer,
        )
        execute_values_calls = []

        def fake_execute_values(cur, sql, rows, page_size):
            execute_values_calls.append((cur, sql, rows, page_size))

        with patch.object(scorer_service, "execute_values", fake_execute_values):
            count = service._score_batch(5)

        self.assertEqual(count, 2)
        self.assertEqual(
            fake_scorer.urls,
            ["https://example.com/a", "https://example.com/b"],
        )

        select_sql, select_params = cursor.executed[0]
        self.assertIn("FROM url_state_current_005", select_sql)
        self.assertIn("url_score_updated_at IS NULL", select_sql)
        self.assertIn("ORDER BY first_seen ASC NULLS LAST", select_sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", select_sql)
        self.assertEqual(select_params, (1000,))

        _, update_sql, rows, page_size = execute_values_calls[0]
        self.assertIn("url_score_updated_at = CURRENT_TIMESTAMP", update_sql)
        self.assertEqual(
            rows,
            [
                ("https://example.com/a", 0.25),
                ("https://example.com/b", 0.75),
            ],
        )
        self.assertEqual(page_size, 2)


class _FakeInlineRanker:
    def score_many(self, urls):
        self.urls = urls
        return [0.42 + i / 100 for i, _ in enumerate(urls)]


class GoldenDiscoveryIngestInlineScoringTest(unittest.TestCase):
    def test_new_links_are_inserted_with_inline_ranker_scores_when_enabled(self):
        ranker = _FakeInlineRanker()
        db = IngestDB(Session=None, inline_ranker=ranker)
        execute_values_calls = []

        records = [
            (
                0,
                {
                    "url": "https://example.com/a",
                    "domain_id": 7,
                    "domain_score": 0.1,
                    "discovered_from": "https://example.com/",
                },
            ),
            (
                1,
                {
                    "url": "https://example.com/b",
                    "domain_id": 7,
                    "domain_score": 0.1,
                    "discovered_from": "https://example.com/",
                },
            ),
        ]

        def fake_execute_values(cur, sql, rows, page_size, fetch=False):
            materialized_rows = list(rows)
            execute_values_calls.append((sql, materialized_rows, page_size, fetch))
            if fetch:
                return [(row[0], True) for row in materialized_rows]
            return None

        with patch.object(ingest_db_ops, "execute_values", fake_execute_values):
            result = db._bulk_links(cur=object(), shard_id=3, items=records)

        self.assertEqual(result, [(0, True), (1, True)])
        self.assertEqual(ranker.urls, ["https://example.com/a", "https://example.com/b"])

        current_sql, current_rows, _, current_fetch = execute_values_calls[0]
        self.assertTrue(current_fetch)
        self.assertIn("url_score, url_score_updated_at", current_sql)
        self.assertEqual(current_rows[0][3], 0.42)
        self.assertEqual(current_rows[1][3], 0.43)
        self.assertIsNotNone(current_rows[0][4])
        self.assertIs(current_rows[0][4], current_rows[1][4])

        history_sql, history_rows, _, history_fetch = execute_values_calls[1]
        self.assertFalse(history_fetch)
        self.assertIn("url_score, url_score_updated_at", history_sql)
        self.assertEqual(history_rows, current_rows)

    def test_new_links_remain_unscored_when_inline_ranker_is_disabled(self):
        db = IngestDB(Session=None, inline_ranker=None)
        execute_values_calls = []
        records = [
            (
                0,
                {
                    "url": "https://example.com/a",
                    "domain_id": 7,
                    "domain_score": 0.1,
                },
            )
        ]

        def fake_execute_values(cur, sql, rows, page_size, fetch=False):
            materialized_rows = list(rows)
            execute_values_calls.append((sql, materialized_rows, page_size, fetch))
            if fetch:
                return [(row[0], True) for row in materialized_rows]
            return None

        with patch.object(ingest_db_ops, "execute_values", fake_execute_values):
            result = db._bulk_links(cur=object(), shard_id=3, items=records)

        self.assertEqual(result, [(0, True)])
        current_rows = execute_values_calls[0][1]
        self.assertEqual(current_rows[0][3], 0.0)
        self.assertIsNone(current_rows[0][4])


class SelectDBCrawlResultWriterTest(unittest.TestCase):
    def test_html_writer_stores_successful_html_by_path(self):
        with TemporaryDirectory() as tmp:
            writer = object.__new__(SelectDBCrawlResultWriter)
            writer.html_dir = Path(tmp)
            path = writer._write_html(
                url="https://example.com/a",
                content="<html>ok</html>",
                fetched_at="2026-05-09T01:02:03+00:00",
                shard_id=5,
                status="ok",
            )

            self.assertIsNotNone(path)
            html_path = Path(path)
            self.assertTrue(html_path.is_file())
            self.assertIn("20260509", str(html_path))
            self.assertIn("shard_005", str(html_path))
            self.assertEqual(html_path.read_text(encoding="utf-8"), "<html>ok</html>")

    def test_html_writer_does_not_write_failures(self):
        with TemporaryDirectory() as tmp:
            writer = object.__new__(SelectDBCrawlResultWriter)
            writer.html_dir = Path(tmp)
            path = writer._write_html(
                url="https://example.com/a",
                content="<html>fail</html>",
                fetched_at="2026-05-09T01:02:03+00:00",
                shard_id=5,
                status="fail",
            )
            self.assertIsNone(path)
            self.assertEqual(list(Path(tmp).glob("**/*")), [])

    def test_upsert_keeps_previous_html_path_when_excluded_path_is_null(self):
        sql = SelectDBCrawlResultWriter._upsert_sql()
        self.assertIn("public.selected_url_crawl_results", sql)
        self.assertIn(
            "html_path = COALESCE(EXCLUDED.html_path, selected_url_crawl_results.html_path)",
            sql,
        )


if __name__ == "__main__":
    unittest.main()
