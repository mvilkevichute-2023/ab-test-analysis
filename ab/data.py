"""Загрузка данных и выполнение SQL-запросов из папки sql/ через DuckDB.

DuckDB понимает почти тот же SQL, что и PostgreSQL, поэтому запросы
можно без изменений запускать и в «боевой» базе.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE exposures AS SELECT * FROM read_csv_auto('{DATA_DIR / 'exposures.csv'}')"
    )
    con.execute(
        f"CREATE TABLE orders AS SELECT * FROM read_csv_auto('{DATA_DIR / 'orders.csv'}')"
    )
    user_metrics_sql = (SQL_DIR / "02_user_metrics.sql").read_text(encoding="utf-8")
    con.execute(f"CREATE TABLE user_metrics AS {user_metrics_sql.rstrip().rstrip(';')}")
    return con


def query(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / name).read_text(encoding="utf-8")).df()
