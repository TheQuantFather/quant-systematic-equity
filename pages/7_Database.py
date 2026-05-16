"""
7_Database.py — Interactive database explorer with table browser and SQL editor.
Read-only: only SELECT statements are permitted.
"""

import streamlit as st
import sqlite3
import pandas as pd
import zlib, io as _io
import numpy as np

st.set_page_config(page_title="Database Explorer", layout="wide")
st.title("Database Explorer")

from config import (
    UNIVERSE_DB, RETURNS_DB, FACTORS_DB, MODELS_DB, CONSTITUENTS_DB, RISK_DB, BARRA_DB,
)
from utils import get_db, inject_css
inject_css()

DATABASES = {
    "universe":     UNIVERSE_DB,
    "returns":      RETURNS_DB,
    "factors":      FACTORS_DB,
    "models":       MODELS_DB,
    "constituents": CONSTITUENTS_DB,
    "risk":         RISK_DB,
    "barra":        BARRA_DB,
}

_VALID_TABLES: dict[str, set[str]] = {}   # populated lazily below

# Blob columns that should not be shown raw
# matrix_blob: numpy .npy format (Ledoit-Wolf N×N)
# cov_blob:    raw float32 bytes (Barra K×K factor covariance)
BLOB_COLS = {"matrix_blob", "cov_blob"}


@st.cache_data
def get_table_names(db_path: str) -> list[str]:
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data
def get_schema(db_path: str, table: str) -> pd.DataFrame:
    _assert_valid_table(db_path, table)
    with get_db(db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return pd.DataFrame(rows, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])


@st.cache_data
def get_row_count(db_path: str, table: str) -> int:
    _assert_valid_table(db_path, table)
    with get_db(db_path) as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return n


def _assert_valid_table(db_path: str, table: str) -> None:
    """Raise ValueError if `table` is not a known table name in this database."""
    if db_path not in _VALID_TABLES:
        _VALID_TABLES[db_path] = set(get_table_names(db_path))
    if table not in _VALID_TABLES[db_path]:
        raise ValueError(f"Unknown table: {table!r}")


def run_query(db_path: str, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """Run a SELECT query. Returns (df, None) on success, (None, error_msg) on failure."""
    stripped = sql.strip().lstrip(";").strip()
    if not stripped.upper().startswith("SELECT") and not stripped.upper().startswith("PRAGMA"):
        return None, "Only SELECT and PRAGMA statements are allowed."
    try:
        with get_db(db_path) as conn:
            df = pd.read_sql_query(sql, conn)
        # Replace blob columns with human-readable summary
        for col in BLOB_COLS:
            if col in df.columns:
                def _blob_summary(b, _col=col):
                    if not isinstance(b, (bytes, memoryview)):
                        return b
                    raw = zlib.decompress(b)
                    if _col == "cov_blob":
                        K = int(len(raw) / 4 ** 0.5)
                        return f"[{K}×{K} float32, {len(b):,} bytes compressed]"
                    else:
                        arr = np.load(_io.BytesIO(raw))
                        n = arr.shape[0]
                        return f"[{n}×{n} float32, {len(b):,} bytes compressed]"
                df[col] = df[col].apply(_blob_summary)
        return df, None
    except Exception as e:
        return None, str(e)



# ---------------------------------------------------------------------------
# Sidebar — pick database and table
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Database")
    db_name = st.selectbox("Database", list(DATABASES.keys()))
    db_path = str(DATABASES[db_name])

    tables = get_table_names(db_path)
    table  = st.selectbox("Table", tables)

    st.divider()
    st.markdown(f"**Path:** `{db_path}`")
    if table:
        n = get_row_count(db_path, table)
        st.markdown(f"**Rows:** {n:,}")

# ---------------------------------------------------------------------------
# Main area — tabs: Browse | Schema | SQL
# ---------------------------------------------------------------------------

tabs = st.tabs(["Browse", "Schema", "SQL"])
tab_browse, tab_schema, tab_sql = tabs

# --- Browse tab ---
with tab_browse:
    if not table:
        st.info("Select a table in the sidebar.")
    else:
        schema = get_schema(db_path, table)
        col_names = schema["name"].tolist()

        with st.expander("Filters", expanded=False):
            filter_col = st.selectbox("Filter column", ["(none)"] + col_names, key="f_col")
            filter_val = st.text_input("Contains (case-insensitive)", key="f_val")

        limit  = st.select_slider("Rows to show", [50, 100, 250, 500, 1000, 5000], value=250)
        offset = st.number_input("Offset (skip rows)", min_value=0, value=0, step=limit)

        if filter_col != "(none)" and filter_val.strip():
            # Column name validated against schema (from selectbox); value is parameterized.
            sql    = f"SELECT * FROM {table} WHERE CAST({filter_col} AS TEXT) LIKE ? LIMIT {limit} OFFSET {offset}"
            params = (f"%{filter_val.strip()}%",)
        else:
            sql    = f"SELECT * FROM {table} LIMIT {limit} OFFSET {offset}"
            params = ()

        try:
            _assert_valid_table(db_path, table)
            with get_db(db_path) as conn:
                df = pd.read_sql_query(sql, conn, params=params)
            err = None
        except Exception as e:
            df, err = None, str(e)

        if err:
            st.error(err)
        else:
            display_sql = sql.replace("?", repr(params[0])) if params else sql
            st.caption(f"`{display_sql}`")
            st.dataframe(df, use_container_width=True, hide_index=True)

# --- Schema tab ---
with tab_schema:
    if not table:
        st.info("Select a table in the sidebar.")
    else:
        schema = get_schema(db_path, table)
        display = schema[["name", "type", "notnull", "pk"]].copy()
        display.columns = ["Column", "Type", "Not Null", "Primary Key"]
        display["Not Null"]    = display["Not Null"].map({0: "", 1: "✓"})
        display["Primary Key"] = display["Primary Key"].map({0: "", 1: "✓"})
        st.dataframe(display, use_container_width=False, hide_index=True, width=500)

        # Row count per table for this database
        st.markdown("**All tables in this database**")
        summary = []
        for t in tables:
            n = get_row_count(db_path, t)
            sch = get_schema(db_path, t)
            summary.append({"Table": t, "Rows": f"{n:,}", "Columns": len(sch)})
        st.dataframe(pd.DataFrame(summary), use_container_width=False, hide_index=True, width=380)

# --- SQL tab ---
with tab_sql:
    st.caption("Read-only — only SELECT and PRAGMA statements are permitted.")

    default_sql = f"SELECT *\nFROM {table}\nLIMIT 100" if table else "SELECT * FROM ??? LIMIT 100"

    sql_input = st.text_area(
        "Query",
        value=default_sql,
        height=180,
        key="sql_input",
        help="Write any SELECT query across this database.",
    )

    col_run, col_dl = st.columns([1, 5])
    run = col_run.button("Run", type="primary", use_container_width=True)

    if run:
        with st.spinner("Running…"):
            result, err = run_query(db_path, sql_input)
        if err:
            st.error(err)
        else:
            st.success(f"{len(result):,} rows returned")
            st.dataframe(result, use_container_width=True, hide_index=True)
            csv = result.to_csv(index=False)
            col_dl.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{db_name}_query.csv",
                mime="text/csv",
                use_container_width=False,
            )
