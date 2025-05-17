import sqlite3
from datetime import datetime
from typing import List, Dict


class DatabaseManager:
    """Simple SQLite-based storage for detection results."""

    def __init__(self, db_path: str = "detections.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                account TEXT,
                from_addr TEXT,
                subject TEXT,
                date TEXT,
                body TEXT,
                is_phishing INTEGER,
                confidence REAL,
                timestamp TEXT
            )
            """
        )
        self.conn.commit()

    def message_exists(self, message_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM messages WHERE message_id=?", (message_id,))
        return cur.fetchone() is not None

    def save_detection(
        self,
        message_id: str,
        email_info: Dict,
        is_phishing: bool,
        confidence: float,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO messages
            (message_id, account, from_addr, subject, date, body, is_phishing, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                email_info.get("account", ""),
                email_info.get("from", ""),
                email_info.get("subject", ""),
                email_info.get("date", "") if not isinstance(email_info.get("date"), datetime) else email_info["date"].isoformat(),
                email_info.get("body", ""),
                1 if is_phishing else 0,
                confidence,
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()

    def get_detections(self, limit: int = 100) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT message_id, account, from_addr, subject, date, body, is_phishing, confidence, timestamp
            FROM messages ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "message_id": row[0],
                    "account": row[1],
                    "from": row[2],
                    "subject": row[3],
                    "date": row[4],
                    "body": row[5],
                    "is_phishing": bool(row[6]),
                    "confidence": row[7],
                    "timestamp": row[8],
                }
            )
        return results

    def close(self) -> None:
        self.conn.close()
