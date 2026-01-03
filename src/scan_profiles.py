import argparse
import json
import os
import sys
import hashlib
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List, Dict, Tuple, Set
from tqdm.auto import tqdm

RESULT_NAME = "embeddings.json.gz"
ALT_VIDEO_RESULT_NAME = "video_embeddings.json.gz"
RESULT_NAMES: Set[str] = {RESULT_NAME, ALT_VIDEO_RESULT_NAME}

IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VID_EXT = {".mp4"}
MEDIA_EXT = IMG_EXT | VID_EXT

AUD_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
TRANSCRIBABLE_EXT = VID_EXT | AUD_EXT

VIDEO_OUT_DEFAULT = "work_queue.jsonl"
AUDIO_OUT_DEFAULT = "audio_work_queue.jsonl"


@dataclass(frozen=True)
class PostScan:
    profile: str
    post_id: str
    post_dir: str   # ABSOLUTE
    names: List[str]


def _listdir_names(path: str) -> List[str]:
    try:
        return [e.name for e in os.scandir(path)]
    except (FileNotFoundError, PermissionError, OSError):
        return []


def _iter_dirs(path: str) -> Iterator[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    yield e
    except (FileNotFoundError, PermissionError, OSError):
        return


def _media_files(names: List[str], allowed_exts: Set[str]) -> List[str]:
    """Return filenames in `names` with extensions in `allowed_exts`, ignoring result artifacts."""
    out: List[str] = []
    ignore = RESULT_NAMES  # lockfiles removed
    for n in names:
        if n in ignore:
            continue
        ext = os.path.splitext(n)[1].lower()
        if ext in allowed_exts:
            out.append(n)
    return sorted(out)


def _job_key(post_dir: str, filename: str) -> str:
    """Stable idempotency key per job (blake2b/64-bit hex)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(post_dir.encode("utf-8"))
    h.update(b"\x00")
    h.update(filename.encode("utf-8"))
    return h.hexdigest()


def _shard_ix(job_key_hex: str, shards: int) -> int:
    """Map a job_key hex string to a shard index [0..shards-1]."""
    if shards <= 1:
        return 0
    return int(job_key_hex, 16) % shards


def scan_profile(profile_dir: str) -> Tuple[str, List[PostScan], Dict[str, int]]:
    profile_name = os.path.basename(profile_dir.rstrip("/"))
    posts: List[PostScan] = []
    stats = {
        "posts_total": 0,
        "queued_seen": 0,
    }

    for post_entry in _iter_dirs(profile_dir):
        stats["posts_total"] += 1
        post_dir = post_entry.path  # ABSOLUTE
        post_id = post_entry.name
        names = _listdir_names(post_dir)

        posts.append(PostScan(
            profile=profile_name,
            post_id=post_id,
            post_dir=post_dir,
            names=names,
        ))
        stats["queued_seen"] += 1

    return profile_name, posts, stats


def build_worklists(
    base_profiles: str,
    max_posts: int,
    workers: int,
    show_progress: bool,
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict[str, int]]]:
    """
    Returns (video_rows, audio_rows, profile_stats).
    Each row uses the minimal schema with ABS post_dir, plus job_key.
    """
    profile_dirs = [e.path for e in _iter_dirs(base_profiles)]
    total_profiles = len(profile_dirs)
    if total_profiles == 0:
        return [], [], {}

    workers = max(1, min(int(workers), 16))

    video_rows: List[Dict] = []
    audio_rows: List[Dict] = []
    profile_stats: Dict[str, Dict[str, int]] = {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = ex.map(scan_profile, profile_dirs)
        for profile_name, posts, stats in tqdm(
            it, total=total_profiles, disable=not show_progress,
            desc="Scanning profiles", unit="profile"
        ):
            for p in posts:
                names_set = set(p.names)

                # A post is "done" for VIDEO if it already has a result file.
                video_done = any(name in RESULT_NAMES for name in names_set)
                if not video_done:
                    media_files = _media_files(p.names, MEDIA_EXT)  # images + .mp4
                    for fname in media_files:
                        jk = _job_key(p.post_dir, fname)
                        video_rows.append({
                            "profile": p.profile,
                            "post_id": p.post_id,
                            "post_dir": p.post_dir,  # ABS
                            "filename": fname,
                            "job_key": jk,
                        })

                transcribables = _media_files(p.names, TRANSCRIBABLE_EXT)
                for fname in transcribables:
                    stem, _ = os.path.splitext(fname)
                    if (stem + ".txt") not in names_set:
                        jk = _job_key(p.post_dir, fname)
                        audio_rows.append({
                            "profile": p.profile,
                            "post_id": p.post_id,
                            "post_dir": p.post_dir,  # ABS
                            "filename": fname,
                            "job_key": jk,
                        })

                # Independent caps per queue
                if max_posts and len(video_rows) >= max_posts and len(audio_rows) >= max_posts:
                    pass  # keep scanning to complete stats for this profile

            profile_stats[profile_name] = stats

            if max_posts and len(video_rows) >= max_posts and len(audio_rows) >= max_posts:
                break

    if max_posts:
        video_rows = video_rows[:max_posts]
        audio_rows = audio_rows[:max_posts]

    return video_rows, audio_rows, profile_stats


def delete_video_results(base_profiles: str, show_progress: bool) -> Tuple[int, int]:
    targets: List[str] = []
    for root, _, files in os.walk(base_profiles, topdown=True, followlinks=False):
        for name in files:
            if name in RESULT_NAMES:
                targets.append(os.path.join(root, name))
    deleted_ok = 0
    delete_errors = 0
    for path in tqdm(targets, disable=not show_progress, desc="Deleting video results", unit="file"):
        try:
            os.remove(path)
            deleted_ok += 1
        except OSError:
            delete_errors += 1
    return deleted_ok, delete_errors


def delete_audio_transcripts(base_profiles: str, show_progress: bool) -> Tuple[int, int]:
    """
    Delete <stem>.txt only when a matching transcribable <stem>.<ext> exists.
    """
    delete_list: List[str] = []
    for root, _, files in os.walk(base_profiles, topdown=True, followlinks=False):
        names_set = set(files)
        stems = {os.path.splitext(f)[0] for f in files if os.path.splitext(f)[1].lower() in TRANSCRIBABLE_EXT}
        for s in stems:
            cand = s + ".txt"
            if cand in names_set:
                delete_list.append(os.path.join(root, cand))

    deleted_ok = 0
    delete_errors = 0
    for path in tqdm(delete_list, disable=not show_progress, desc="Deleting transcripts", unit="file"):
        try:
            os.remove(path)
            deleted_ok += 1
        except OSError:
            delete_errors += 1
    return deleted_ok, delete_errors


def _write_sharded_jsonl(base_path: str, rows: List[Dict], shards: int) -> None:
    """
    Write rows to a single JSONL file or N sharded files using deterministic hashing of job_key.
    When shards == 1, preserves original single-file behavior.
    """
    shards = max(1, int(shards))
    if shards == 1:
        with open(base_path, "w", encoding="utf-8") as f:
            for r in rows:
                # ensure job_key present even if upstream changes
                r = dict(r)
                r.setdefault("job_key", _job_key(r["post_dir"], r["filename"]))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    prefix, ext = os.path.splitext(base_path)
    with ExitStack() as stack:
        files = [stack.enter_context(open(f"{prefix}-{i:02d}{ext}", "w", encoding="utf-8")) for i in range(shards)]
        for r in rows:
            r = dict(r)
            r.setdefault("job_key", _job_key(r["post_dir"], r["filename"]))
            i = _shard_ix(r["job_key"], shards)
            files[i].write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build per-file video and audio worklists (lock-free).")
    ap.add_argument("profiles_root", help="ABS path to the PROFILES root directory")

    ap.add_argument("--max-posts", type=int, default=0,
                    help="Cap EACH queue independently (0 = unlimited).")
    default_workers = min(max(1, (os.cpu_count() or 2)), 8)
    ap.add_argument("--workers", type=int, default=default_workers,
                    help=f"Thread workers (default: {default_workers}, clamped to 1..16)")
    ap.add_argument("--summary", action="store_true", help="Print a summary to stderr.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    ap.add_argument("--video-out", default=VIDEO_OUT_DEFAULT, help=f"Path to write video JSONL (default: {VIDEO_OUT_DEFAULT})")
    ap.add_argument("--audio-out", default=AUDIO_OUT_DEFAULT, help=f"Path to write audio JSONL (default: {AUDIO_OUT_DEFAULT})")
    ap.add_argument("--delete-video", action="store_true", help=f"Delete {', '.join(sorted(RESULT_NAMES))} before scanning.")
    ap.add_argument("--delete-audio", action="store_true", help="Delete transcript <stem>.txt files before scanning.")
    ap.add_argument("--shards", type=int, default=1,
                    help="Number of output shards per queue. 1 = single file (default).")

    args = ap.parse_args()

    root = os.path.abspath(args.profiles_root)
    if not os.path.isdir(root):
        print(f"ERROR: Not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    if args.delete_video:
        dv_ok, dv_err = delete_video_results(root, show_progress=(not args.no_progress))
        print(f"[DELETE-VIDEO] removed={dv_ok} errors={dv_err}", file=sys.stderr)
    if args.delete_audio:
        da_ok, da_err = delete_audio_transcripts(root, show_progress=(not args.no_progress))
        print(f"[DELETE-AUDIO] removed={da_ok} errors={da_err}", file=sys.stderr)

    video_rows, audio_rows, stats = build_worklists(
        base_profiles=root,
        max_posts=args.max_posts,
        workers=args.workers,
        show_progress=(not args.no_progress),
    )

    _write_sharded_jsonl(args.video_out, video_rows, args.shards)
    _write_sharded_jsonl(args.audio_out, audio_rows, args.shards)

    if args.summary:
        total_profiles = len(stats)
        posts_total = sum(s.get("posts_total", 0) for s in stats.values())
        print(
            f"[SUMMARY] profiles={total_profiles} posts_total={posts_total} "
            f"video_rows={len(video_rows)} audio_rows={len(audio_rows)} shards={args.shards}",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()
