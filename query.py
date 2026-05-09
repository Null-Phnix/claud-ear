#!/usr/bin/env python3
"""
CLI for querying the music library.

Usage:
  uv run python query.py --bpm 130-150
  uv run python query.py --key "C minor" --genre trap
  uv run python query.py --mood dark --energy high --limit 20
  uv run python query.py --similar path/to/song.wav --n 5
  uv run python query.py --insights
"""
import argparse
import json
from pathlib import Path


def print_table(rows: list[dict]):
    if not rows:
        print("No results.")
        return
    headers = ["artist", "title", "bpm", "key", "energy", "mood"]
    col_w = {h: max(len(h), max((len(str(r.get(h) or "")) for r in rows), default=0)) for h in headers}
    sep = "  "
    header_line = sep.join(h.ljust(col_w[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(sep.join(str(r.get(h) or "").ljust(col_w[h]) for h in headers))
    print(f"\n{len(rows)} result(s)")


def cmd_query(db, args):
    bpm_min = bpm_max = None
    if args.bpm:
        parts = args.bpm.split("-")
        if len(parts) == 2:
            bpm_min, bpm_max = float(parts[0]), float(parts[1])
        else:
            bpm_min = bpm_max = float(parts[0])
    rows = db.query_songs(
        bpm_min=bpm_min, bpm_max=bpm_max,
        key=args.key, genre=args.genre,
        mood=args.mood, energy=args.energy,
        limit=args.limit,
    )
    print_table(rows)


def cmd_similar(db, wav_path: str, n: int):
    from extractor import compute_embedding, cosine_similarity

    print(f"Computing embedding for: {wav_path}")
    query_embed = compute_embedding(wav_path, max_duration=30.0)
    if query_embed is None:
        print("ERROR: could not compute embedding.")
        return

    all_embeds = db.get_all_embeddings()
    if not all_embeds:
        print("No embeddings stored yet. Run: uv run python extractor.py")
        return

    scores = []
    for row in all_embeds:
        if row["embedding"]:
            sim = cosine_similarity(query_embed, row["embedding"])
            scores.append((sim, row))

    scores.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop {n} similar songs:")
    for sim, row in scores[:n]:
        print(f"  {sim:.3f}  {row.get('artist','?')} — {row.get('title','?')}")
        print(f"         {row.get('file_path','')}")


def cmd_insights(db):
    stats = db.features_stats()
    db_stats = db.stats()

    print("=== Library Insights ===\n")
    print(f"Songs analyzed: {db_stats.get('done', 0)}")
    print(f"Songs with features extracted: {stats['total_with_features']}")
    print(f"Pending: {db_stats.get('pending', 0)}  Errors: {db_stats.get('error', 0)}\n")

    if stats["bpm_avg"]:
        print(f"BPM range: {stats['bpm_min']} – {stats['bpm_max']}  (avg {stats['bpm_avg']})\n")

    if stats["top_keys"]:
        print("Top keys:")
        for k in stats["top_keys"]:
            bar = "█" * k["count"]
            print(f"  {k['key']:20s} {bar} {k['count']}")

    # Genre breakdown from JSON columns
    from collections import Counter
    import sqlite3
    conn = sqlite3.connect(str(Path(__file__).parent / "agent.db"))
    rows = conn.execute("SELECT genres FROM song_features WHERE genres IS NOT NULL").fetchall()
    conn.close()
    genre_counter: Counter = Counter()
    for (genres_json,) in rows:
        for g in json.loads(genres_json or "[]"):
            genre_counter[g] += 1
    if genre_counter:
        print("\nGenre breakdown:")
        for genre, cnt in genre_counter.most_common(8):
            bar = "█" * cnt
            print(f"  {genre:20s} {bar} {cnt}")

    # Top similar artists mentioned across all analyses
    artist_counter: Counter = Counter()
    conn = sqlite3.connect(str(Path(__file__).parent / "agent.db"))
    rows = conn.execute("SELECT similar_artists FROM song_features WHERE similar_artists IS NOT NULL").fetchall()
    conn.close()
    for (artists_json,) in rows:
        for a in json.loads(artists_json or "[]"):
            artist_counter[a.strip()] += 1
    if artist_counter:
        print("\nMost mentioned similar artists:")
        for artist, cnt in artist_counter.most_common(10):
            print(f"  {cnt:3d}x  {artist}")


def main():
    from song_db import SongDB
    db = SongDB(Path(__file__).parent / "agent.db")

    parser = argparse.ArgumentParser(description="Query the music intelligence library")
    parser.add_argument("--bpm", help="BPM range, e.g. 130-150 or 140")
    parser.add_argument("--key", help="Key, e.g. 'C minor'")
    parser.add_argument("--genre", help="Genre keyword, e.g. 'trap'")
    parser.add_argument("--mood", help="Mood keyword, e.g. 'dark'")
    parser.add_argument("--energy", choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--similar", metavar="WAV_PATH", help="Find songs similar to this WAV")
    parser.add_argument("--n", type=int, default=5, help="Number of similar songs to return")
    parser.add_argument("--insights", action="store_true", help="Show library-wide stats")
    args = parser.parse_args()

    if args.insights:
        cmd_insights(db)
    elif args.similar:
        cmd_similar(db, args.similar, args.n)
    else:
        cmd_query(db, args)


if __name__ == "__main__":
    main()
