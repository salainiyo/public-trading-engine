# Public Trading Engine

[![CI](https://github.com/salainiyo/public-trading-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/salainiyo/public-trading-engine/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A single-strategy, single-broker trading bot for a funded-account challenge.
It watches H1 XAUUSD, generates signals from a rule-based strategy, manages
positions and challenge rules (drawdowns, targets, valid trading days), and
exposes a local web dashboard for monitoring and control.

## Strategy: Supertrend + RSI

- **Market / timeframe:** XAUUSD, H1.
- **Regime filter:** close versus the ATR(10) lower band, computed as
  `(H + L) / 2 - 3 * ATR(10)`. Above the band is an uptrend regime; below it
  is a downtrend regime.
- **Long:** close above the previous 20-bar high, price above the ATR band,
  and RSI(14) > 60.
- **Short:** close below the previous 20-bar low, price below the ATR band,
  and RSI(14) < 40.
- **Entry:** market order at the next bar's open (signals are evaluated on
  closed bars only).
- **Risk:** fixed $25 stop loss, RR 3 (target $75), 1% risk per trade,
  maximum 2 concurrent positions. No add-ons, no trailing stop, no news
  filter, and no Friday-flat rule.

The strategy is named "supertrend" for historical reasons, but the code does
not implement a true trailing Supertrend with flip dots. It is an ATR-band
regime filter: a static lower-band test on ATR(10) that decides which side of
the market is tradable, combined with RSI momentum and a 20-bar breakout.

## Firm rulebooks

All three firms trade the same strategy with the same execution and risk
settings above.

| Firm | Fee | Profit targets (phase 1 / phase 2) | Daily drawdown | Max drawdown | Split |
| --- | --- | --- | --- | --- | --- |
| GoatFunded | $17 | 8% / 6% | 4% | 10% | 80% |
| AquaFunded | $39 | 8% / 5% | 5% | 8% | 90% |
| FundingPips | $36 | 8% / 5% | 5% | 10% | 80% |

Common rules: 3 valid trading days per phase, 14-day payout cycle, and
payouts retain the remaining equity in the account.

## Architecture

```text
Broker API (TradeLocker)
        │  quotes / bars / orders
        ▼
   main.py ── bar pipeline ──► engine/signals.py (Supertrend RSI)
        │                        engine/indicators.py (ATR10, RSI14, 20-bar extremes)
        │
        ├──► engine/strategy_runner.py ── challenge state, sizing, SL/TP, payouts
        ├──► engine/reconciler.py ─────── broker ↔ local trade state
        ├──► parity/ ──────────────────── live vs backtest trade comparison
        └──► events/websocket_broadcaster.py ──► dashboard/index.html (localhost:8765)

Persistence: database/trade_store.py, database/treasury_store.py (SQLite under bot/data/)
Credentials: config/credentials.py (AES-GCM), config/tokens.py, config/loader.py
```

## Quick start

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your broker credentials
python main.py
```

The dashboard is then available at http://localhost:8765/.

## Configuration

All runtime configuration lives in `.env` (see `.env.example`) and in the
farm definitions at `engine/market.py:GOAT_FIRM_5K_PROFILE`,
`AQUA_FIRM_5K_PROFILE`, `FP_FIRM_5K_PROFILE` and `engine/market.py:FARMS`.
Farms are the only place strategy, stop mode, RR, risk and concurrency are
configured. Broker credentials use the `GOAT_*` variable names for
compatibility; they apply to all three firm farms.

## Tests

```bash
python3 -m pytest tests/unit/ tests/integration/ -v
```

CI runs the same command on every push and pull request to `main`.

## Standalone backtest

The archived research backtest is kept as a standalone script. It needs
`numpy`, `pandas`, and `numba`.

```bash
python3 scripts/supertrend_rsi_propfirm.py --data path/to/XAU_H1.csv --slice test --paths 1000
```

## Docker (optional)

```bash
docker compose up -d --build
```

The dashboard is exposed only on localhost port 8765.

## Known differences from the backtest

- The backtest models spread and slippage; live trading uses real broker
  fills and commission.
- Both live and backtest enter at the next bar's open; live fills depend on
  the broker's execution speed and session availability.

## Roadmap

- Optional broker adapters beyond the current single broker.
- Automated strategy-report generation from the standalone backtest.
- Historical parity dashboards for live-vs-backtest drift.

See [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md).
