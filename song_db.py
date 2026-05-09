#!/usr/bin/env python3
"""SQLite database wrapper for the music intelligence agent."""
import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime


MUSIC_DIR = Path.home() / "Documents" / "music" / "music data"


class SongDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    title TEXT,
                    artist TEXT,
                    status TEXT DEFAULT 'pending',
                    analyzed_at TEXT,
                    document_path TEXT,
                    youtube_url TEXT,
                    youtube_views INTEGER,
                    raw_analysis_json TEXT
                );

                CREATE TABLE IF NOT EXISTS discovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    source_artist TEXT,
                    source_song TEXT,
                    discovered_at TEXT DEFAULT (datetime('now')),
                    downloaded_file TEXT
                );

                CREATE TABLE IF NOT EXISTS song_features (
                    song_id          INTEGER PRIMARY KEY REFERENCES songs(id),
                    bpm              REAL,
                    key              TEXT,
                    chords           TEXT,
                    genres           TEXT,
                    mood             TEXT,
                    energy           TEXT,
                    instruments      TEXT,
                    similar_artists  TEXT,
                    embedding        BLOB,
                    extracted_at     TEXT
                );
            """)

            # Migrate: add quality/retry columns if they don't exist yet
            for col_def in [
                "ALTER TABLE songs ADD COLUMN retry_count INTEGER DEFAULT 0",
                "ALTER TABLE songs ADD COLUMN quality_score INTEGER",
                "ALTER TABLE songs ADD COLUMN quality_issues TEXT",
            ]:
                try:
                    conn.execute(col_def)
                except Exception:
                    pass  # column already exists

    def add_song(self, file_path: str, title: str = None, artist: str = None):
        """Insert a song with status=pending. Idempotent via INSERT OR IGNORE."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO songs (file_path, title, artist) VALUES (?, ?, ?)",
                (str(file_path), title, artist)
            )

    def get_pending_songs(self) -> list[str]:
        """
        Return file paths to analyze, smart-ordered by priority:
        1. needs_retry songs (retry_count < 3)
        2. pending billboard chart songs
        3. pending discovered songs (any source)
        4. remaining pending songs (FIFO)
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.file_path
                   FROM songs s
                   WHERE s.status IN ('pending', 'needs_retry')
                     AND (s.retry_count IS NULL OR s.retry_count < 3)
                   ORDER BY
                     CASE s.status WHEN 'needs_retry' THEN 0 ELSE 1 END,
                     CASE WHEN EXISTS(
                       SELECT 1 FROM discovery_log dl
                       WHERE dl.downloaded_file = s.file_path
                         AND dl.source_artist = 'billboard'
                     ) THEN 0 ELSE 1 END,
                     CASE WHEN EXISTS(
                       SELECT 1 FROM discovery_log dl
                       WHERE dl.downloaded_file = s.file_path
                     ) THEN 0 ELSE 1 END,
                     s.id"""
            ).fetchall()
        return [row["file_path"] for row in rows]

    def mark_analyzing(self, file_path: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE songs SET status = 'analyzing' WHERE file_path = ?",
                (str(file_path),)
            )

    def mark_done(self, file_path: str, doc_path: str, views: int = None):
        with self._conn() as conn:
            conn.execute(
                """UPDATE songs SET status = 'done', analyzed_at = ?,
                   document_path = ?, youtube_views = ?
                   WHERE file_path = ?""",
                (datetime.now().isoformat(), str(doc_path), views, str(file_path))
            )

    def mark_error(self, file_path: str, error: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE songs SET status = 'error', raw_analysis_json = ? WHERE file_path = ?",
                (error, str(file_path))
            )

    def get_analyzed_artists(self) -> list[str]:
        """Return distinct artists whose songs have been analyzed."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT artist FROM songs WHERE status = 'done' AND artist IS NOT NULL"
            ).fetchall()
        return [row["artist"] for row in rows if row["artist"]]

    def scan_library(self, music_dir: Path = None):
        """Scan music data dir and insert all WAV files as pending."""
        if music_dir is None:
            music_dir = MUSIC_DIR
        count = 0
        for wav in sorted(music_dir.glob("*.wav")):
            self.add_song(str(wav))
            count += 1
        return count

    def discovery_count(self) -> int:
        """Return number of songs that were auto-discovered."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM discovery_log WHERE downloaded_file IS NOT NULL"
            ).fetchone()
        return row["cnt"]

    def log_discovery(self, query: str, source_artist: str, source_song: str, downloaded_file: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO discovery_log (query, source_artist, source_song, downloaded_file) VALUES (?, ?, ?, ?)",
                (query, source_artist, source_song, downloaded_file)
            )

    def stats(self) -> dict:
        """Return summary statistics."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM songs GROUP BY status"
            ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def get_song_id(self, file_path: str) -> int | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM songs WHERE file_path = ?", (str(file_path),)
            ).fetchone()
        return row["id"] if row else None

    def upsert_features(self, song_id: int, features: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO songs (id, file_path) VALUES (?, ?)",
                (song_id, f"__placeholder_{song_id}__")
            )
            conn.execute(
                """INSERT INTO song_features
                   (song_id, bpm, key, chords, genres, mood, energy, instruments,
                    similar_artists, embedding, extracted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(song_id) DO UPDATE SET
                     bpm=excluded.bpm, key=excluded.key, chords=excluded.chords,
                     genres=excluded.genres, mood=excluded.mood, energy=excluded.energy,
                     instruments=excluded.instruments, similar_artists=excluded.similar_artists,
                     embedding=excluded.embedding, extracted_at=excluded.extracted_at""",
                (
                    song_id,
                    features.get("bpm"),
                    features.get("key"),
                    json.dumps(features.get("chords") or []),
                    json.dumps(features.get("genres") or []),
                    features.get("mood"),
                    features.get("energy"),
                    json.dumps(features.get("instruments") or []),
                    json.dumps(features.get("similar_artists") or []),
                    features.get("embedding"),
                    datetime.now().isoformat(),
                )
            )

    def get_songs_without_features(self) -> list[dict]:
        """Return done songs that don't have features extracted yet."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.id, s.file_path, s.document_path, s.artist, s.title
                   FROM songs s
                   LEFT JOIN song_features sf ON s.id = sf.song_id
                   WHERE s.status = 'done' AND sf.song_id IS NULL"""
            ).fetchall()
        return [dict(r) for r in rows]

    def query_songs(self, bpm_min=None, bpm_max=None, key=None, genre=None,
                    mood=None, energy=None, limit=20) -> list[dict]:
        """Structured query over song_features JOIN songs."""
        clauses = []
        params = []
        if bpm_min is not None:
            clauses.append("sf.bpm >= ?"); params.append(bpm_min)
        if bpm_max is not None:
            clauses.append("sf.bpm <= ?"); params.append(bpm_max)
        if key:
            clauses.append("LOWER(sf.key) LIKE ?"); params.append(f"%{key.lower()}%")
        if mood:
            clauses.append("LOWER(sf.mood) LIKE ?"); params.append(f"%{mood.lower()}%")
        if energy:
            clauses.append("LOWER(sf.energy) = ?"); params.append(energy.lower())
        if genre:
            clauses.append("LOWER(sf.genres) LIKE ?"); params.append(f"%{genre.lower()}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT s.artist, s.title, s.file_path, s.document_path,
                           sf.bpm, sf.key, sf.mood, sf.energy, sf.genres, sf.similar_artists
                    FROM songs s INNER JOIN song_features sf ON s.id = sf.song_id
                    {where}
                    ORDER BY sf.bpm
                    LIMIT ?""",
                params + [limit]
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["genres"] = json.loads(d["genres"]) if d["genres"] else []
            d["similar_artists"] = json.loads(d["similar_artists"]) if d["similar_artists"] else []
            result.append(d)
        return result

    def get_all_embeddings(self) -> list[dict]:
        """Return all songs with stored CLAP embeddings."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.id as song_id, s.file_path, s.artist, s.title, sf.embedding
                   FROM songs s JOIN song_features sf ON s.id = sf.song_id
                   WHERE sf.embedding IS NOT NULL"""
            ).fetchall()
        return [dict(r) for r in rows]

    def features_stats(self) -> dict:
        """Return aggregate stats across all song_features rows."""
        with self._conn() as conn:
            bpm_row = conn.execute(
                "SELECT MIN(bpm) as bpm_min, MAX(bpm) as bpm_max, AVG(bpm) as bpm_avg FROM song_features WHERE bpm IS NOT NULL"
            ).fetchone()
            key_rows = conn.execute(
                "SELECT key, COUNT(*) as cnt FROM song_features WHERE key IS NOT NULL GROUP BY key ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) as cnt FROM song_features").fetchone()["cnt"]
        return {
            "total_with_features": total,
            "bpm_min": bpm_row["bpm_min"],
            "bpm_max": bpm_row["bpm_max"],
            "bpm_avg": round(bpm_row["bpm_avg"], 1) if bpm_row["bpm_avg"] else None,
            "top_keys": [{"key": r["key"], "count": r["cnt"]} for r in key_rows],
        }


    def get_song_retry_info(self, file_path: str) -> dict:
        """Return {retry_count, quality_score, quality_issues} for a song."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT retry_count, quality_score, quality_issues FROM songs WHERE file_path = ?",
                (str(file_path),)
            ).fetchone()
        if not row:
            return {"retry_count": 0, "quality_score": None, "quality_issues": []}
        return {
            "retry_count": row["retry_count"] or 0,
            "quality_score": row["quality_score"],
            "quality_issues": json.loads(row["quality_issues"] or "[]"),
        }

    def update_quality(self, file_path: str, score: int, issues: list):
        """Store quality score on a good (no retry needed) done song."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE songs SET quality_score = ?, quality_issues = ? WHERE file_path = ?",
                (score, json.dumps(issues), str(file_path))
            )

    def mark_needs_retry(self, file_path: str, score: int, issues: list):
        """Set status='needs_retry' and store quality info."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE songs SET status = 'needs_retry', quality_score = ?,
                   quality_issues = ? WHERE file_path = ?""",
                (score, json.dumps(issues), str(file_path))
            )

    def mark_abandoned(self, file_path: str, score: int, issues: list):
        """Set status='abandoned' after max retries exhausted."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE songs SET status = 'abandoned', quality_score = ?,
                   quality_issues = ? WHERE file_path = ?""",
                (score, json.dumps(issues), str(file_path))
            )

    def increment_retry_count(self, file_path: str):
        """Increment retry_count by 1 before re-analyzing a needs_retry song."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE songs SET retry_count = COALESCE(retry_count, 0) + 1 WHERE file_path = ?",
                (str(file_path),)
            )

    def promote_errors_to_retry(self):
        """Move all status='error' songs to 'needs_retry' so they get re-tried."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE songs SET status = 'needs_retry' WHERE status = 'error'"
            )


if __name__ == "__main__":
    db_path = Path(__file__).parent / "agent.db"
    db = SongDB(db_path)
    scanned = db.scan_library()
    stats = db.stats()
    print(f"Scanned library: {scanned} WAV files found")
    print(f"Database stats: {stats}")
    pending = db.get_pending_songs()
    print(f"Pending songs: {len(pending)}")
    if pending:
        print(f"First pending: {Path(pending[0]).name}")
