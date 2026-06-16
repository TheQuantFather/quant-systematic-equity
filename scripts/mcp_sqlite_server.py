"""Read-only MCP server over the project's SQLite databases.

Exposes schema-discovery and SELECT-only query tools for every ``data/*.db``
file so an MCP client (e.g. Claude Code) can explore factors/models/returns
without any risk of writing. Read-only is enforced two ways:

1. Each connection is opened with the SQLite URI ``mode=ro`` (kernel refuses
   any write at the file level — safe even while another process, such as the
   EDGAR pull, is writing to the same DB).
2. A SQL guard rejects any statement that is not a single read
   (``SELECT`` / ``WITH`` / ``EXPLAIN`` / ``PRAGMA``).

Run as a stdio server (registered via ``.mcp.json``); never print to stdout —
that channel carries the MCP protocol. All diagnostics go to stderr.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# data/ lives one level up from scripts/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_ROWS = 1000  # cap result size so a careless query can't dump millions of rows

mcp = FastMCP("quant-sqlite")


def _databases() -> dict[str, Path]:
    """Discover available .db files keyed by stem (e.g. 'factors')."""
    return {p.stem: p for p in sorted(DATA_DIR.glob("*.db"))}


def _connect(database: str) -> sqlite3.Connection:
    """Open one database read-only. Validates the name against discovered DBs
    (prevents path traversal — only known data/*.db are reachable)."""
    dbs = _databases()
    if database not in dbs:
        raise ValueError(
            f"Unknown database '{database}'. Available: {', '.join(sorted(dbs))}"
        )
    conn = sqlite3.connect(f"file:{dbs[database]}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _is_read_only(sql: str) -> bool:
    """True only for a single read statement."""
    stripped = sql.strip().rstrip(";").lstrip()
    if ";" in stripped:  # reject multi-statement payloads
        return False
    head = stripped.split(None, 1)[0].upper() if stripped else ""
    return head in {"SELECT", "WITH", "EXPLAIN", "PRAGMA"}


@mcp.tool()
def list_databases() -> str:
    """List the available SQLite databases with size and table count."""
    out = []
    for name, path in _databases().items():
        with _connect(name) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
        out.append(
            {"database": name, "size_mb": round(path.stat().st_size / 1e6, 1),
             "tables": n}
        )
    return json.dumps(out, indent=2)


@mcp.tool()
def list_tables(database: str) -> str:
    """List tables (with row counts) in the given database."""
    with _connect(database) as conn:
        names = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        rows = []
        for t in names:
            cnt = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            rows.append({"table": t, "rows": cnt})
    return json.dumps(rows, indent=2)


@mcp.tool()
def describe_table(database: str, table: str) -> str:
    """Show columns (name, type, pk flag) for one table."""
    with _connect(database) as conn:
        valid = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if table not in valid:
            raise ValueError(f"Unknown table '{table}' in '{database}'.")
        cols = [
            {"name": r["name"], "type": r["type"], "pk": bool(r["pk"])}
            for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        ]
    return json.dumps(cols, indent=2)


@mcp.tool()
def read_query(database: str, sql: str) -> str:
    """Run a read-only SQL query (SELECT/WITH/EXPLAIN/PRAGMA) and return rows
    as JSON. Results are capped at the first 1000 rows."""
    if not _is_read_only(sql):
        raise ValueError(
            "Only single read-only statements are allowed "
            "(SELECT / WITH / EXPLAIN / PRAGMA)."
        )
    with _connect(database) as conn:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        data = [dict(r) for r in rows]
    truncated = len(data) == MAX_ROWS
    return json.dumps(
        {"row_count": len(data), "truncated": truncated, "rows": data},
        indent=2, default=str,
    )


if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"data dir not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    mcp.run()
