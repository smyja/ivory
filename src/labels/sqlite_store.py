import os
import sqlite3
from datetime import datetime
from typing import Optional, Tuple


def _dataset_labels_dir(dataset_name: str) -> str:
    safe = dataset_name.replace("/", "__").replace("\\", "__")
    return os.path.join("datasets", safe, "labels")


def _label_db_path(dataset_name: str, label_name: str) -> str:
    return os.path.join(_dataset_labels_dir(dataset_name), f"{label_name}.labels.sqlite")


def ensure_label_db(dataset_name: str, label_name: str) -> str:
    path = _label_db_path(dataset_name, label_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        c = conn.cursor()
        # Table with both row_id and text; row_id preferred unique key.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                row_id TEXT,
                text TEXT,
                value TEXT,
                updated_at TEXT
            );
            """
        )
        # Ensure indexes (SQLite lacks IF NOT EXISTS for UNIQUE constraints on existing tables, use indexes)
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_labels_row_id ON labels(row_id)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_labels_text ON labels(text)")
        conn.commit()
    return path


def upsert_label_by_text(dataset_name: str, label_name: str, text: str, value: str) -> None:
    path = ensure_label_db(dataset_name, label_name)
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(path) as conn:
        c = conn.cursor()
        # Upsert by text
        c.execute(
            "INSERT INTO labels(text, value, updated_at) VALUES (?, ?, ?)\n"
            "ON CONFLICT(text) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (text, value, ts),
        )
        conn.commit()


def get_label_by_text(dataset_name: str, label_name: str, text: str) -> Optional[Tuple[str, str, str]]:
    path = ensure_label_db(dataset_name, label_name)
    with sqlite3.connect(path) as conn:
        c = conn.cursor()
        row = c.execute("SELECT text, value, updated_at FROM labels WHERE text = ?", (text,)).fetchone()
        return row


def upsert_label_by_row_id(
    dataset_name: str, label_name: str, row_id: str, value: str, text: Optional[str] = None
) -> None:
    path = ensure_label_db(dataset_name, label_name)
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(path) as conn:
        c = conn.cursor()
        # Upsert by row_id; also update text if provided
        if text is not None:
            c.execute(
                "INSERT INTO labels(row_id, text, value, updated_at) VALUES (?, ?, ?, ?)\n"
                "ON CONFLICT(row_id) DO UPDATE SET text=excluded.text, value=excluded.value, updated_at=excluded.updated_at",
                (row_id, text, value, ts),
            )
        else:
            c.execute(
                "INSERT INTO labels(row_id, value, updated_at) VALUES (?, ?, ?)\n"
                "ON CONFLICT(row_id) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (row_id, value, ts),
            )
        conn.commit()
