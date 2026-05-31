# Broker Integration Plan

## Goals
- Support DeGiro now, IBKR soon, others later without restructuring
- Single database, broker-agnostic schema
- Clean separation: auth / data sync / execution / display
- Consistent with existing repo conventions (flat scripts, `get_db`, `config.py`, `utils.py`)

---

## File structure

```
brokers/
    __init__.py
    base.py          # Abstract BrokerClient — interface every broker must implement
    degiro.py        # DeGiro implementation (absorbs degiro_orders.py)
    ibkr.py          # IBKR implementation (stub for now)

broker_sync.py       # Pulls from any broker → broker.db  (mirrors create_*.py style)
broker_trade.py      # Execution logic — broker-agnostic, calls brokers/*.py
pages/10_Broker.py   # Streamlit dashboard
```

`degiro_orders.py` → deleted once `brokers/degiro.py` is live.

---

## Abstract interface  (`brokers/base.py`)

Every broker implements these methods:

```python
class BrokerClient(ABC):
    def connect(self) -> None: ...
    def get_portfolio(self) -> list[dict]: ...        # positions
    def get_cash(self) -> list[dict]: ...             # balances per currency
    def get_open_orders(self) -> list[dict]: ...
    def get_trades(self, from_date: date, to_date: date) -> list[dict]: ...
    def get_orders_history(self, from_date: date, to_date: date) -> list[dict]: ...
    def place_order(self, isin: str, action: str, order_type: str,
                    qty: int, price: float | None) -> str: ...   # returns order_id
    def cancel_order(self, order_id: str) -> bool: ...
```

Credentials always come from `.env` (loaded once in `connect()`). Never passed as arguments.

---

## Database:  `broker.db`

All brokers share one DB, distinguished by a `broker` column (`"degiro"`, `"ibkr"`, etc.).

```sql
-- Point-in-time position snapshots
CREATE TABLE positions (
    broker       TEXT,
    snapshot_date DATE,
    isin         TEXT,
    name         TEXT,
    qty          REAL,
    avg_cost     REAL,
    current_price REAL,
    value        REAL,
    unrealised_pnl REAL,
    currency     TEXT,
    PRIMARY KEY (broker, snapshot_date, isin)
);

-- Executed fills
CREATE TABLE trades (
    broker       TEXT,
    trade_date   DATE,
    isin         TEXT,
    name         TEXT,
    action       TEXT,   -- BUY / SELL
    qty          REAL,
    price        REAL,
    fee          REAL,
    currency     TEXT,
    order_id     TEXT,
    PRIMARY KEY (broker, order_id)
);

-- Cash balances
CREATE TABLE cash (
    broker       TEXT,
    snapshot_date DATE,
    currency     TEXT,
    amount       REAL,
    PRIMARY KEY (broker, snapshot_date, currency)
);

-- Orders (open + historical)
CREATE TABLE orders (
    broker       TEXT,
    created_date DATE,
    isin         TEXT,
    action       TEXT,
    order_type   TEXT,   -- LIMIT / MARKET / STOP_LIMIT
    qty          REAL,
    limit_price  REAL,
    status       TEXT,   -- OPEN / FILLED / CANCELLED / EXPIRED
    order_id     TEXT,
    PRIMARY KEY (broker, order_id)
);
```

---

## broker_sync.py

```
python broker_sync.py --broker degiro          # snapshot positions + cash today
python broker_sync.py --broker degiro --trades # pull trade history (last 30d)
python broker_sync.py --broker all             # all connected brokers
```

Mirrors `create_factors.py --date` style. Safe to re-run (idempotent).

---

## broker_trade.py

```
python broker_trade.py --broker degiro --action buy  --isin US4581401001 --qty 5 --price 19.50
python broker_trade.py --broker degiro --action sell --isin US4581401001 --qty 5
python broker_trade.py --broker degiro --cancel --order-id <id>
python broker_trade.py --broker degiro --list-open
```

Always does a dry-run check first and prints fee/margin estimate. Requires `--confirm` to submit.

---

## pages/10_Broker.py

Sections:
1. **Positions** — current holdings table with unrealised P&L, overlaid with model scores from `models.db` where ISIN matches
2. **Cash** — balance per currency
3. **Open orders** — live order book with cancel button
4. **Trade history** — fills over selected date range
5. **P&L curve** — cumulative realised + unrealised over time (from `positions` snapshots)

---

## Integration with existing pipeline

- `positions.isin` joins directly to `universe.db companies` and `models.db` — model scores on your actual holdings appear automatically
- `broker_sync.py` can be added as a step in `daily_update.py`
- `config.py` gets `BROKER_DB` path constant

---

## Build order

1. `brokers/base.py` — abstract class
2. `brokers/degiro.py` — port `degiro_orders.py` logic in
3. `broker.db` schema (add to `create_databases.py`)
4. `broker_sync.py` — positions + cash + trades
5. `broker_trade.py` — CLI execution
6. `pages/10_Broker.py` — dashboard
7. `brokers/ibkr.py` — when ready
