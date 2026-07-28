"""Tests for config loading, the fixture X backend, and the read budget.

Nothing here touches the network. The budget tests in particular exist because
the failure mode they guard against costs real money.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from augury_signal.clients.budget import ReadBudget, ReadBudgetExceeded
from augury_signal.clients.x import FixtureXBackend, XClient
from augury_signal.config import load_market_config, repo_root
from augury_signal.models import MarketQuery, MarketStatus, Venue, market_id


class TestMarketConfig:
    def test_loads_the_shipped_config(self):
        config = load_market_config()
        assert config.markets
        assert all(t.market.venue in (Venue.KALSHI, Venue.POLYMARKET) for t in config.markets)

    def test_targets_are_claims_not_questions(self):
        """The stance model conditions on a claim; a question scores worse."""
        for tracked in load_market_config().markets:
            assert not tracked.market.target.strip().endswith("?")

    def test_lookup_by_id_and_by_ticker(self):
        config = load_market_config()
        first = config.markets[0].market
        assert config.resolve(first.market_id).market.ticker == first.ticker
        assert config.resolve(first.ticker).market.market_id == first.market_id

    def test_unknown_market_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="tracked markets are"):
            load_market_config().by_id("kalshi:NOPE")

    def test_rejects_mismatched_id(self, tmp_path):
        bad = tmp_path / "markets.yaml"
        bad.write_text(
            "version: 1\nmarkets:\n"
            "  - id: kalshi:WRONG\n    venue: kalshi\n    ticker: RIGHT\n"
            "    title: t\n    target: c\n    query:\n      keywords: [a]\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="does not match venue/ticker"):
            load_market_config(bad)

    def test_rejects_market_with_no_query_terms(self, tmp_path):
        bad = tmp_path / "markets.yaml"
        bad.write_text(
            "version: 1\nmarkets:\n"
            "  - venue: kalshi\n    ticker: T\n    title: t\n    target: c\n"
            "    query:\n      exclude: [x]\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="no query terms"):
            load_market_config(bad)

    def test_rejects_unknown_version(self, tmp_path):
        bad = tmp_path / "markets.yaml"
        bad.write_text("version: 99\nmarkets: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported config version"):
            load_market_config(bad)


class TestQueryBuilding:
    def test_quotes_phrases_and_negates_exclusions(self):
        query = MarketQuery(
            market_id="kalshi:X",
            keywords=["rate cut", "FOMC"],
            exclude=["fed up"],
        ).to_x_query(lang="en")
        assert '"rate cut"' in query
        assert "FOMC" in query
        assert '-"fed up"' in query
        assert "-is:retweet" in query
        assert "lang:en" in query

    def test_deduplicates_terms(self):
        q = MarketQuery(market_id="m", keywords=["Fed"], aliases=["Fed"]).all_terms()
        assert q == ["Fed"]

    def test_empty_query_is_an_error(self):
        with pytest.raises(ValueError, match="no positive terms"):
            MarketQuery(market_id="m").to_x_query()


class TestFixtureBackend:
    @property
    def fixtures_dir(self):
        return repo_root() / "apps" / "augury-signal" / "fixtures"

    def test_replays_the_fed_fixture(self):
        backend = FixtureXBackend(self.fixtures_dir)
        posts = backend.search("ignored", "kalshi:KXFEDDECISION-26JUL-C25")
        assert len(posts) >= 20
        assert all(p.source == "fixture" for p in posts)
        assert all(p.market_id == "kalshi:KXFEDDECISION-26JUL-C25" for p in posts)

    def test_anchoring_keeps_posts_recent(self):
        """Un-anchored fixtures decay to nothing and every test sees an empty signal."""
        backend = FixtureXBackend(self.fixtures_dir, anchor_to_now=True)
        posts = backend.search("q", "kalshi:KXFEDDECISION-26JUL-C25")
        newest = max(p.created_at for p in posts)
        assert datetime.now(UTC) - newest < timedelta(minutes=1)

    def test_anchoring_preserves_relative_spacing(self):
        backend = FixtureXBackend(self.fixtures_dir, anchor_to_now=True)
        posts = sorted(backend.search("q", "kalshi:KXFEDDECISION-26JUL-C25"), key=lambda p: p.created_at)
        span = posts[-1].created_at - posts[0].created_at
        assert timedelta(hours=40) < span < timedelta(hours=60)

    def test_unknown_market_returns_nothing(self):
        assert FixtureXBackend(self.fixtures_dir).search("q", "kalshi:NOPE") == []

    def test_max_results_returns_the_newest(self):
        backend = FixtureXBackend(self.fixtures_dir)
        posts = backend.search("q", "kalshi:KXFEDDECISION-26JUL-C25", max_results=5)
        assert len(posts) == 5
        all_posts = backend.search("q", "kalshi:KXFEDDECISION-26JUL-C25")
        assert min(p.created_at for p in posts) >= sorted(p.created_at for p in all_posts)[-5]

    def test_malformed_line_names_the_file_and_line(self, tmp_path):
        (tmp_path / "default.jsonl").write_text('{"post_id":"a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            FixtureXBackend(tmp_path).search("q", "kalshi:ANY")


class TestReadBudget:
    def _budget(self, tmp_path, limit=100):
        return ReadBudget(path=tmp_path / "budget.json", max_daily_reads=limit)

    def test_starts_empty(self, tmp_path):
        b = self._budget(tmp_path)
        assert b.used_today() == 0
        assert b.remaining() == 100

    def test_records_and_persists_across_instances(self, tmp_path):
        """An in-memory counter resets on restart and turns a crash loop into
        unbounded spending — the flaw in the Phase 1 prototype."""
        self._budget(tmp_path).record(30)
        assert self._budget(tmp_path).used_today() == 30

    def test_check_raises_before_exceeding(self, tmp_path):
        b = self._budget(tmp_path, limit=50)
        b.record(40)
        b.check(10)
        with pytest.raises(ReadBudgetExceeded, match="would exceed the budget"):
            b.check(11)

    def test_error_message_quotes_a_dollar_cost(self, tmp_path):
        b = self._budget(tmp_path, limit=1)
        with pytest.raises(ReadBudgetExceeded, match=r"\$"):
            b.check(500)

    def test_corrupt_ledger_is_not_read_as_zero(self, tmp_path):
        """Treating an unreadable ledger as unspent silently restores the budget."""
        path = tmp_path / "budget.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="refusing to assume"):
            ReadBudget(path=path, max_daily_reads=100).used_today()

    def test_ledger_is_valid_json(self, tmp_path):
        b = self._budget(tmp_path)
        b.record(5)
        assert isinstance(json.loads((tmp_path / "budget.json").read_text()), dict)

    def test_negative_inputs_rejected(self, tmp_path):
        b = self._budget(tmp_path)
        with pytest.raises(ValueError):
            b.record(-1)
        with pytest.raises(ValueError):
            b.check(-1)


class TestXClientSelection:
    class _Settings:
        x_live = False
        x_bearer_token = None
        max_daily_reads = 100

    def test_defaults_to_fixtures(self):
        client = XClient.from_settings(self._Settings())
        assert not client.is_live

    def test_live_without_token_warns_and_falls_back(self):
        settings = self._Settings()
        settings.x_live = True
        with pytest.warns(RuntimeWarning, match="falling back to fixture"):
            client = XClient.from_settings(settings)
        assert not client.is_live


def test_market_id_is_venue_prefixed():
    assert market_id(Venue.KALSHI, "ABC") == "kalshi:ABC"


def test_market_settled_invariant():
    from augury_signal.models import Market

    with pytest.raises(ValueError, match="settled exactly when"):
        Market(
            market_id="kalshi:X",
            venue=Venue.KALSHI,
            ticker="X",
            title="t",
            target="c",
            status=MarketStatus.SETTLED,
            resolved_outcome=None,
        )
