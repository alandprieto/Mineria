import sqlite3
from contextlib import contextmanager

import pandas as pd

from .config import DB_PATH

TABLES = [
    "Country",
    "League",
    "Player",
    "Team",
    "Player_Attributes",
    "Team_Attributes",
    "Match",
]


@contextmanager
def connect(path=DB_PATH):
    conn = sqlite3.connect(str(path))
    try:
        yield conn
    finally:
        conn.close()


def load_table(table_name, path=DB_PATH):
    if table_name not in TABLES:
        raise ValueError(f"Tabla desconocida: {table_name}")
    with connect(path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def load_all(path=DB_PATH):
    return {table: load_table(table, path) for table in TABLES}