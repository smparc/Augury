# %% [markdown]
# # Augury — Phase 1 Prototype
#
# Quick, ugly, end-to-end pass: pull posts from X about one Kalshi market,
# score sentiment with VADER, pull the market's own price history, and plot
# them against each other. The only goal here is to see if there's *any*
# visible relationship before building five services around the idea.
#
# No database, no Dagster, no DeBERTa — those come in later phases. This
# notebook only needs a subset of `requirements.txt`:
# `requests pandas matplotlib seaborn vaderSentiment python-dotenv jupyter`.

# %%
import os
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

X_BEARER_TOKEN = os.environ["X_BEARER_TOKEN"]
MAX_DAILY_READS = int(os.environ.get("MAX_DAILY_READS", 2000))
KALSHI_BASE_URL = os.environ.get("KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2")

sns.set_theme(style="darkgrid")
_reads_used_this_session = 0  # rough running total, printed after every X call

# %% [markdown]
# ## 1. Find a Kalshi market to track
#
# Kalshi's market data is public — no key needed. Set `SERIES_TICKER` to a
# series you care about (e.g. `"KXFED"` for Fed rate decisions), or leave it
# `None` to browse whatever's currently open.

# %%
def list_kalshi_markets(series_ticker: str | None = None, status: str = "open", limit: int = 20) -> pd.DataFrame:
    params = {"status": status, "limit": limit}
    if series_ticker:
        params["series_ticker"] = series_ticker
    resp = requests.get(f"{KALSHI_BASE_URL}/markets", params=params, timeout=10)
    resp.raise_for_status()
    markets = resp.json()["markets"]
    df = pd.DataFrame(markets)
    cols = [c for c in ["ticker", "event_ticker", "title", "yes_bid", "yes_ask", "volume", "close_time"] if c in df.columns]
    return df[cols]

SERIES_TICKER = None  # e.g. "KXFED" — set this after browsing the output below once
list_kalshi_markets(series_ticker=SERIES_TICKER, limit=25)

# %% [markdown]
# ## 2. Pull that market's price history
#
# Copy a `ticker` from the table above. The series ticker is usually
# everything before the first `-` (e.g. `KXFED-26MAR19` → series `KXFED`) —
# sanity-check it against the `event_ticker` column if the split looks wrong.

# %%
TICKER = "REPLACE_ME"          # e.g. "KXFED-26MAR19"
SERIES_TICKER = "REPLACE_ME"   # e.g. "KXFED"

def get_kalshi_price_history(series_ticker: str, ticker: str, period_minutes: int = 60, lookback_days: int = 14) -> pd.DataFrame:
    """period_minutes must be 1, 60, or 1440 per Kalshi's candlesticks endpoint.
    Param names per docs.kalshi.com/api-reference/market/get-market-candlesticks —
    Kalshi's API does evolve, so double-check there if this 404s."""
    end_ts = int(time.time())
    start_ts = end_ts - lookback_days * 86400
    url = f"{KALSHI_BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_minutes}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    candles = resp.json()["candlesticks"]
    rows = [
        {
            "timestamp": pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
            "yes_price": float(c["price"]["close_dollars"]),
        }
        for c in candles
    ]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

price_df = get_kalshi_price_history(SERIES_TICKER, TICKER, period_minutes=60, lookback_days=14)
price_df.tail()

# %% [markdown]
# ## 3. Pull related X posts
#
# X is pay-per-use (~$0.005/read as of 2026), so this stays deliberately
# small. Keep `QUERY` to plain keywords — the rich operators from the x.com
# search box (`min_faves:`, `since:`/`until:`, `filter:blue_verified`, etc.)
# are silently dropped by this endpoint; they don't error, they just do
# nothing.

# %%
QUERY = "REPLACE_ME -is:retweet lang:en"   # e.g. '(Fed OR "interest rates") -is:retweet lang:en'

def search_x_posts(query: str, max_results: int = 100, start_time: datetime | None = None) -> pd.DataFrame:
    global _reads_used_this_session
    if _reads_used_this_session + max_results > MAX_DAILY_READS:
        raise RuntimeError(f"Would exceed MAX_DAILY_READS ({MAX_DAILY_READS}) — raise the budget or lower max_results.")

    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at",
    }
    if start_time:
        params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    resp = requests.get("https://api.x.com/2/tweets/search/recent", headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    posts = resp.json().get("data", [])

    _reads_used_this_session += len(posts)
    print(f"Pulled {len(posts)} posts — ~${len(posts) * 0.005:.2f} — {_reads_used_this_session}/{MAX_DAILY_READS} reads used this session")

    return pd.DataFrame(posts)

posts_df = search_x_posts(QUERY, max_results=100, start_time=datetime.now(timezone.utc) - timedelta(days=7))
posts_df.head()

# %% [markdown]
# ## 4. Score sentiment
#
# VADER only — a lexicon baseline, not the target-conditioned DeBERTa model
# from the full architecture. Good enough to see if there's a signal worth
# building the rest of the pipeline for.

# %%
analyzer = SentimentIntensityAnalyzer()
posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], utc=True)
posts_df["compound"] = posts_df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
posts_df[["created_at", "text", "compound"]].head()

# %% [markdown]
# ## 5. Align sentiment with price and plot
#
# Bucket posts into the same hourly windows as the candlesticks, average the
# compound score per bucket, then plot both series on twin axes.

# %%
sentiment_hourly = (
    posts_df.set_index("created_at")["compound"]
    .resample("1h")
    .mean()
    .rename("sentiment")
    .reset_index()
    .rename(columns={"created_at": "timestamp"})
)

merged = pd.merge_asof(
    price_df.sort_values("timestamp"),
    sentiment_hourly.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("2h"),
)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
ax1.plot(merged["timestamp"], merged["yes_price"], color="tab:blue", label="Kalshi YES price ($)")
ax2.plot(merged["timestamp"], merged["sentiment"], color="tab:orange", label="Mean X sentiment")
ax1.set_ylabel("YES price ($)", color="tab:blue")
ax2.set_ylabel("Mean sentiment (VADER compound)", color="tab:orange")
ax1.set_title(f"{TICKER}: price vs. X sentiment")
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Quick lagged-correlation gut check
#
# Not a real Granger causality test — no stationarity check, no lag-order
# selection, no multiple-comparisons correction. Those belong in the R
# analytics phase once there's enough history to make them meaningful. This
# is only a rough first look at whether sentiment and price move together at
# all, and in which direction.

# %%
clean = merged.dropna(subset=["yes_price", "sentiment"])
for lag_hours in [-2, -1, 0, 1, 2]:
    corr = clean["sentiment"].shift(lag_hours).corr(clean["yes_price"])
    direction = "sentiment leads" if lag_hours > 0 else "price leads" if lag_hours < 0 else "same hour"
    print(f"lag {lag_hours:+d}h ({direction}): r = {corr:.3f}")
