import os
import glob
import json
from typing import Any, Dict, List, Optional, Tuple

import duckdb


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection.

    If IVORY_USE_TABLE_INDEX=1 is set, use a persistent on-disk cache DB
    at duckdb_cache.db; otherwise use an in-memory connection.
    """
    use_cache = _env_flag("IVORY_USE_TABLE_INDEX", "0")
    if use_cache:
        # Place cache at repo root if possible
        db_path = os.path.abspath(os.path.join(os.getcwd(), "duckdb_cache.db"))
        conn = duckdb.connect(db_path)
    else:
        conn = duckdb.connect(":memory:")

    # Attempt to load optional extensions (best-effort)
    try:
        conn.execute("LOAD sqlite_scanner;")
    except Exception:
        # sqlite scanner may not be available in all builds
        pass
    try:
        conn.execute("LOAD httpfs;")
    except Exception:
        pass

    return conn


def _dataset_parquet_glob(dataset_name: str) -> str:
    """Return a glob that matches parquet files for the dataset."""
    # datasets/<name>/**/data.parquet
    safe = dataset_name.replace("/", "__").replace("\\", "__")
    base = os.path.join("datasets", safe)
    return os.path.join(base, "**", "*.parquet")


def _label_sqlite_paths(dataset_name: str) -> List[str]:
    safe = dataset_name.replace("/", "__").replace("\\", "__")
    labels_dir = os.path.join("datasets", safe, "labels")
    return sorted(glob.glob(os.path.join(labels_dir, "*.sqlite")))


def _quoted_ident(name: str) -> str:
    # Basic identifier quoting for view/table names
    return '"' + name.replace('"', '""') + '"'


def create_dataset_view(conn: duckdb.DuckDBPyConnection, dataset_name: str) -> str:
    """
    Create (or replace) a DuckDB view for a dataset that reads Parquet files directly.

    Returns the view name. The view includes all columns from Parquet and a
    helper column `__filename` indicating the source file. Stable row IDs are not
    enforced here; see notes below.
    """
    parquet_glob = _dataset_parquet_glob(dataset_name)
    safe_view = dataset_name.replace("/", "_").replace("\\", "_")
    view_name = "t_" + safe_view

    # Use filename=true to expose source path which we can use as part of a composite key
    sql = f"""
    CREATE OR REPLACE VIEW {_quoted_ident(view_name)} AS
    SELECT *
    FROM read_parquet('{parquet_glob}', filename=true)
    """
    conn.execute(sql)
    return view_name


# --- Simple, safe JSON query compilation ---

ALLOWED_OPS = {"eq", "neq", "lt", "lte", "gt", "gte", "contains", "in"}


def _compile_where(
    where: List[Dict[str, Any]],
    params: List[Any],
    any_text_columns: Optional[List[str]] = None,
) -> str:
    clauses: List[str] = []
    for cond in where:
        col = cond.get("column")
        op = cond.get("op")
        val = cond.get("value")
        if not col or op not in ALLOWED_OPS:
            continue
        # Special case: search across any text-like column when column is "*"
        if op == "contains" and str(col) == "*" and any_text_columns:
            # Build an OR group across all candidate text columns
            or_parts: List[str] = []
            for c in any_text_columns:
                ident_c = _quoted_ident(c)
                or_parts.append(f"STRPOS(LOWER(CAST({ident_c} AS TEXT)), LOWER(?)) > 0")
            if or_parts:
                clauses.append("(" + " OR ".join(or_parts) + ")")
                # Need one parameter per placeholder in the OR group
                params.extend([str(val)] * len(or_parts))
            continue

        ident = _quoted_ident(col)
        if op == "eq":
            clauses.append(f"{ident} = ?")
            params.append(val)
        elif op == "neq":
            clauses.append(f"{ident} <> ?")
            params.append(val)
        elif op == "lt":
            clauses.append(f"{ident} < ?")
            params.append(val)
        elif op == "lte":
            clauses.append(f"{ident} <= ?")
            params.append(val)
        elif op == "gt":
            clauses.append(f"{ident} > ?")
            params.append(val)
        elif op == "gte":
            clauses.append(f"{ident} >= ?")
            params.append(val)
        elif op == "contains":
            clauses.append(f"STRPOS(LOWER(CAST({ident} AS TEXT)), LOWER(?)) > 0")
            params.append(str(val))
        elif op == "in":
            if not isinstance(val, list) or len(val) == 0:
                # skip invalid IN
                continue
            placeholders = ",".join(["?"] * len(val))
            clauses.append(f"{ident} IN ({placeholders})")
            params.extend(val)
    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


def execute_query(
    dataset_name: str,
    select: Optional[List[str]] = None,
    where: Optional[List[Dict[str, Any]]] = None,
    order_by: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    return_total: bool = False,
) -> Dict[str, Any]:
    """
    Execute a safe, simple analytical query over the dataset view.

    - Reads Parquet via DuckDB.
    - Does NOT persist results.
    - Joins labels (if present) by text when a `text` column exists (temporary until stable row ids are added).
    """
    conn = get_duckdb_connection()
    view = create_dataset_view(conn, dataset_name)

    # Detect labels and build LEFT JOINs when label files exist.
    # For now, join by `text` column if present in the view; otherwise skip.
    # Each label file is expected to have table `labels(text TEXT PRIMARY KEY, value TEXT)`.
    label_paths = _label_sqlite_paths(dataset_name)
    join_sql_parts: List[str] = []
    # Inspect view columns to see if __row_id is present
    has_row_id = False
    try:
        info_cur = conn.execute(f"PRAGMA table_info('{view}')")
        cols_info = info_cur.fetchall()
        view_cols = {row[1] for row in cols_info} if cols_info and len(cols_info[0]) > 1 else set()
        has_row_id = "__row_id" in view_cols
    except Exception:
        pass

    alias_main = "v"  # safe SQL alias for the dataset view

    for idx, path in enumerate(label_paths):
        alias = f"lbl_{idx}"
        try:
            conn.execute(f"ATTACH DATABASE '{path}' AS {alias} (TYPE SQLITE)")
        except Exception:
            continue

        # Try join by row_id first if view has it
        joined = False
        if has_row_id:
            try:
                test_sql = (
                    f"SELECT 1 FROM {_quoted_ident(view)} AS {alias_main} "
                    f"LEFT JOIN {alias}.labels AS {alias} ON {alias}.row_id = {alias_main}.__row_id LIMIT 1"
                )
                conn.execute(test_sql)
                join_sql_parts.append(
                    f"LEFT JOIN {alias}.labels AS {alias} ON {alias}.row_id = {alias_main}.__row_id"
                )
                joined = True
            except Exception:
                joined = False

        if not joined:
            # Fallback to joining by text if both sides have text
            try:
                test_sql = (
                    f"SELECT 1 FROM {_quoted_ident(view)} AS {alias_main} "
                    f"LEFT JOIN {alias}.labels AS {alias} ON {alias}.text = {alias_main}.text LIMIT 1"
                )
                conn.execute(test_sql)
                join_sql_parts.append(
                    f"LEFT JOIN {alias}.labels AS {alias} ON {alias}.text = {alias_main}.text"
                )
            except Exception:
                # skip if cannot join
                pass

    params: List[Any] = []
    sel = select or ["*"]
    safe_sel = ", ".join(_quoted_ident(c) if c != "*" else "*" for c in sel)
    base = f"SELECT {safe_sel} FROM {_quoted_ident(view)} AS {alias_main} "
    joins = ("\n" + "\n".join(join_sql_parts) + "\n") if join_sql_parts else ""
    # Inspect view columns and collect text-like columns for broad search
    any_text_columns: List[str] = []
    try:
        info_cur = conn.execute(f"PRAGMA table_info('{view}')")
        cols_info = info_cur.fetchall()
        # DuckDB PRAGMA table_info returns: [cid, name, type, notnull, dflt_value, pk]
        for row in cols_info:
            try:
                name = row[1]
                typ = (row[2] or "").upper()
                # Include typical textual types; otherwise allow casting anyway
                if "CHAR" in typ or "TEXT" in typ or "STRING" in typ or typ == "VARCHAR":
                    any_text_columns.append(name)
            except Exception:
                continue
        # Fallback: if none detected, include all columns so CAST will handle non-text
        if not any_text_columns and cols_info:
            any_text_columns = [row[1] for row in cols_info if len(row) > 1]
    except Exception:
        any_text_columns = []

    where_sql = _compile_where(where or [], params, any_text_columns=any_text_columns)

    order_sql = ""
    if order_by and isinstance(order_by, dict):
        col = order_by.get("column")
        direction = order_by.get("direction", "asc").lower()
        if col:
            direction_sql = "DESC" if direction == "desc" else "ASC"
            order_sql = f" ORDER BY {_quoted_ident(col)} {direction_sql}"

    limit_sql = ""
    if isinstance(limit, int) and limit > 0:
        limit_sql = f" LIMIT {int(limit)}"
    if isinstance(offset, int) and offset > 0:
        limit_sql += f" OFFSET {int(offset)}"

    sql = base + joins + where_sql + order_sql + limit_sql
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    data = [dict(zip(cols, r)) for r in rows]
    result: Dict[str, Any] = {"columns": cols, "rows": data}

    if return_total:
        # Compute total count with the same WHERE (no limit/offset)
        count_params: List[Any] = []
        count_where = _compile_where(where or [], count_params, any_text_columns=any_text_columns)
        count_sql = f"SELECT COUNT(*) AS total FROM {_quoted_ident(view)} AS {alias_main} " + count_where
        try:
            count_row = conn.execute(count_sql, count_params).fetchone()
            result["total"] = int(count_row[0]) if count_row else 0
        except Exception:
            result["total"] = None

    return result


def preview_schema(dataset_name: str, limit: int = 5) -> Dict[str, Any]:
    """Return a tiny sample of the dataset rows and column names/types."""
    conn = get_duckdb_connection()
    view = create_dataset_view(conn, dataset_name)
    cur = conn.execute(f"SELECT * FROM {_quoted_ident(view)} LIMIT {int(limit)}")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return {"columns": cols, "sample": rows}
