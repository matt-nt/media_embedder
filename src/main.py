from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from types import SimpleNamespace
import gzip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor
import tempfile

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from video import DaliVideoDecoder
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

LOG = logging.getLogger("barebones")
_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_level_name, format="%(asctime)s %(levelname)s %(message)s")
LOG.setLevel(getattr(logging, _level_name, logging.DEBUG))
LOG.debug("Logger initialized at %s", _level_name)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
prefer_gpu = True

SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USER = os.getenv("SFTP_USER")
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
REMOTE_BASE_PATH = os.getenv("REMOTE_BASE_PATH")
SFTP_PKEY = os.getenv("SFTP_PKEY")

if not (SFTP_HOST and SFTP_USER and REMOTE_BASE_PATH):
    raise ValueError("Missing env: SFTP_HOST, SFTP_USER, REMOTE_BASE_PATH")

IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
RESULT_NAME = "embeddings.json.gz"
SSH_KEY_SECRET_PATH = "/run/secrets/ssh_private_key"

def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f}{u}"
        f /= 1024

def _to_json_gz_bytes(obj: dict) -> bytes:
    """
    Serialize `obj` to UTF-8 JSON and gzip it (deterministic mtime=0).
    """
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    buf = io.BytesIO()
    # mtime=0 => stable gzip headers; helpful for dedup/caching
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    """Parse an environment variable as float with a safe fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        LOG.warning("Invalid float for %s=%s; using default %.2f", name, raw, default)
        return default


def _int_env(name: str, default: int) -> int:
    """Parse an environment variable as int with a safe fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        LOG.warning("Invalid int for %s=%s; using default %d", name, raw, default)
        return default


def _strip_quotes(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.strip().strip('"').strip("'")


@dataclass
class WorkItem:
    post_id: str
    rel: PurePosixPath  # path relative to REMOTE_BASE_PATH
    filename: Optional[PurePosixPath] = None
    profile: Optional[str] = None


@dataclass
class WorkQueueEntry:

    post_id: str
    post_dir: PurePosixPath
    filename: Optional[PurePosixPath]
    profile: Optional[str]

    def to_work_item(self, remote_base: PurePosixPath) -> WorkItem:
        """Convert the queue entry into the normalized :class:`WorkItem`."""

        rel = _normalize_rel(remote_base, self.post_dir)
        return WorkItem(
            post_id=self.post_id,
            rel=rel,
            filename=self.filename,
            profile=self.profile,
        )


@dataclass
class Downloaded:
    item: WorkItem
    local_dir: Path


@dataclass
class UploadTask:
    remote_rel: PurePosixPath
    payload: bytes


# --------------------------
# Speech gating helpers
# --------------------------


def _merge_spans(
    spans: List[Tuple[float, float]],
    max_gap: float = 1.5,
    max_window: float = 60.0,
) -> List[Tuple[float, float]]:

    if not spans:
        return []

    spans = sorted(spans)
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if end < start:
            start, end = end, start
        if start < cur_start:
            start = cur_start
        if end < start:
            continue

        proposed_end = max(cur_end, end)
        within_gap = start - cur_end <= max_gap
        within_window = (proposed_end - cur_start) <= max_window
        if within_gap and within_window:
            cur_end = proposed_end
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))
    return merged


class SpeechGate:
    """Inclusive speech gate that favours recall over precision."""

    def __init__(
        self,
        *,
        pad: float,
        merge_within: float,
        min_len: float,
        backend: str = "ffmpeg",
    ) -> None:
        self.pad = float(pad)
        self.merge_within = float(merge_within)
        self.min_len = float(min_len)
        self.backend = backend

    def detect(self, wav_path: Path) -> List[Tuple[float, float]]:
        """Detect speech segments and apply post-processing."""

        spans: List[Tuple[float, float]] = []
        try:
            spans = self._detect_ffmpeg_energy(wav_path)
        except FileNotFoundError:
            LOG.warning("ffmpeg not available; falling back to full audio for %s", wav_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.warning("ffmpeg detection failed for %s: %s", wav_path, exc)

        if not spans:
            dur = self._duration_via_ffprobe(wav_path)
            if dur > 0:
                return [(0.0, dur)]
            return []

        return self._post(spans, wav_path)

    def _duration_via_ffprobe(self, wav_path: Path) -> float:
        """Measure clip duration using ffprobe (fallback to 0 on failure)."""

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ]
        try:
            probe = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            LOG.warning("ffprobe not available; treating %s duration as 0", wav_path)
            return 0.0

        output = (probe.stdout or "").strip()
        try:
            return max(float(output), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _detect_ffmpeg_energy(self, wav_path: Path) -> List[Tuple[float, float]]:
        """Detect non-silent spans via ffmpeg's silencedetect filter."""

        noise_db = os.getenv("VAD_FFMPEG_NOISE_DB", "-45dB")
        min_sil = os.getenv("VAD_FFMPEG_MIN_SIL", "0.50")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(wav_path),
            "-af",
            f"silencedetect=noise={noise_db}:d={min_sil}",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stderr = proc.stderr or ""
        if proc.returncode not in {0, 255}:
            LOG.debug(
                "ffmpeg silencedetect returned code %s for %s. stderr=%s",
                proc.returncode,
                wav_path,
                stderr.strip(),
            )

        silences: List[Tuple[float, float]] = []
        current_start: Optional[float] = None
        for line in stderr.splitlines():
            line = line.strip()
            if "silence_start" in line:
                try:
                    value = line.split("silence_start:", 1)[1].split()[0]
                    current_start = float(value)
                except (IndexError, ValueError):  # pragma: no cover - parsing guard
                    current_start = None
            elif "silence_end" in line:
                try:
                    value = line.split("silence_end:", 1)[1].split("|")[0].strip()
                    end_val = float(value)
                except (IndexError, ValueError):  # pragma: no cover - parsing guard
                    current_start = None
                    continue
                if current_start is None:
                    current_start = 0.0
                silences.append((current_start, end_val))
                current_start = None

        duration = self._duration_via_ffprobe(wav_path)
        if current_start is not None:
            silences.append((current_start, duration))

        silences = [(max(0.0, s), max(0.0, e)) for s, e in silences if e >= s]
        silences.sort()

        spans: List[Tuple[float, float]] = []
        cursor = 0.0
        for sil_start, sil_end in silences:
            sil_start = min(sil_start, duration)
            sil_end = min(sil_end, duration)
            if sil_start > cursor:
                spans.append((cursor, sil_start))
            cursor = max(cursor, sil_end)

        if duration > cursor:
            spans.append((cursor, duration))

        return [(s, e) for s, e in spans if e - s > 0.0]

    def _post(self, spans: List[Tuple[float, float]], wav_path: Path) -> List[Tuple[float, float]]:
        if not spans:
            return []
        spans.sort()
        merged: List[Tuple[float, float]] = []
        gap = max(0.6, self.merge_within)
        for start, end in spans:
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        duration = self._duration_via_ffprobe(wav_path)
        pad = max(0.35, self.pad)
        padded: List[Tuple[float, float]] = []
        for start, end in merged:
            padded_start = max(0.0, start - pad)
            padded_end = end + pad
            if duration > 0.0:
                padded_end = min(duration, padded_end)
            padded.append((padded_start, padded_end))
        return padded

# --------------------------
# Remote I/O (rsync / ssh)
# --------------------------

class RemoteIO:
    def __init__(self) -> None:
        self.host = SFTP_HOST
        self.user = SFTP_USER
        self.port = SFTP_PORT
        self.base = PurePosixPath(REMOTE_BASE_PATH)
        self.pkey = _strip_quotes(SFTP_PKEY)

        self._ssh_parts = [
            "ssh",
            "-T",
            "-p",
            str(self.port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        control_path = os.path.join(
            tempfile.gettempdir(), f"media-embedder-ssh-{os.getpid()}"
        )
        # Clean up any stale control socket from a previous run that reused the pid.
        try:
            os.unlink(control_path)
        except FileNotFoundError:
            pass
        self._ssh_parts += [
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60s",
            "-o",
            f"ControlPath={control_path}",
        ]
        if self.pkey:
            self._ssh_parts += ["-i", self.pkey]
        secret = _strip_quotes(SSH_KEY_SECRET_PATH)
        if os.path.exists(secret):
            self._ssh_parts += ["-i", secret]
        self._rsync_base = ["rsync", "-aq", "--no-motd", "-e", shlex.join(self._ssh_parts)]

        LOG.debug("RemoteIO: host=%s user=%s port=%s base=%s pkey=%s secret=%s",
                  self.host, self.user, self.port, self.base, bool(self.pkey), os.path.exists(secret))

    @property
    def remote_spec(self) -> str:
        return f"{self.user}@{self.host}"

    def download_worklist(self, remote_name: str, dst: Path) -> None:
        src = f"{self.remote_spec}:{(self.base/remote_name).as_posix()}"
        cmd = [*self._rsync_base, src, str(dst)]
        LOG.debug("rsync worklist: %s", shlex.join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)

    def download_dir(self, rel: PurePosixPath, dst: Path) -> None:
        src = f"{self.remote_spec}:{(self.base/rel).as_posix().rstrip('/')}/"
        dst.mkdir(parents=True, exist_ok=True)
        cmd = [*self._rsync_base, src, str(dst)]
        LOG.debug("rsync dir: %s", shlex.join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)

    def upload_bytes(self, rel: PurePosixPath, data: bytes) -> None:
        remote_path = (self.base / rel).as_posix()
        remote_dir = str(PurePosixPath(remote_path).parent)
        tmp_path = remote_path + ".tmp"
        ssh = [*self._ssh_parts, f"{self.user}@{self.host}"]
        command = (
            f"mkdir -p {shlex.quote(remote_dir)}"
            f" && cat > {shlex.quote(tmp_path)}"
            f" && mv {shlex.quote(tmp_path)} {shlex.quote(remote_path)}"
        )
        LOG.debug("ssh upload pipeline: %s", shlex.join([*ssh, command]))
        subprocess.run([*ssh, command], input=data, check=True, capture_output=True)


# --------------------------
# Worklist
# --------------------------

def _parse_worklist(path: Path) -> List[WorkQueueEntry]:
    LOG.debug("Parsing worklist: %s", path)
    text = path.read_text(encoding="utf-8").lstrip()
    if not text:
        LOG.debug("Worklist empty")
        return []

    def _iter_rows() -> Iterable[dict]:
        if text.startswith("["):
            yield from json.loads(text)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)

    entries: List[WorkQueueEntry] = []
    for row in _iter_rows():
        post_id = str(row.get("post_id", "")).strip()
        post_dir_raw = row.get("post_dir") or row.get("relative_path")
        profile_raw = row.get("profile")
        filename_raw = row.get("filename")

        profile = str(profile_raw).strip() if profile_raw is not None else ""
        profile = profile or None

        filename: Optional[PurePosixPath] = None
        if filename_raw is not None:
            name = str(filename_raw).strip()
            if name:
                candidate = PurePosixPath(name.strip("/"))
                if any(part in {"", ".", ".."} for part in candidate.parts):
                    LOG.warning("Skipping unsafe filename for %s: %s", post_id or "<missing>", name)
                else:
                    filename = candidate

        if not post_id or not post_dir_raw:
            LOG.warning("Skipping invalid row: %s", row)
            continue

        post_dir_str = str(post_dir_raw).strip()
        if not post_dir_str:
            LOG.warning("Skipping row with empty post_dir: %s", row)
            continue

        candidate_dir = PurePosixPath(post_dir_str)
        if any(part == ".." for part in candidate_dir.parts):
            LOG.warning("Skipping unsafe post_dir for %s: %s", post_id, post_dir_str)
            continue

        # Drop empty/"." components but preserve the remainder of the path so the
        # remote base can be normalized later on.
        cleaned_parts = [part for part in candidate_dir.parts if part not in {"", ".", "/"}]
        if not cleaned_parts:
            LOG.warning("Skipping row with no usable path components: %s", row)
            continue

        post_dir = PurePosixPath(*cleaned_parts)
        entries.append(
            WorkQueueEntry(
                post_id=post_id,
                post_dir=post_dir,
                filename=filename,
                profile=profile,
            )
        )

    LOG.debug("Parsed %d items", len(entries))
    if entries:
        LOG.debug("First item: id=%s post_dir=%s", entries[0].post_id, entries[0].post_dir)
    return entries


# --------------------------
# Embedding
# --------------------------

@dataclass
class FrameBatchEntry:
    """Container for RGB frame payloads passed to the embedder."""

    data: Union[np.ndarray, torch.Tensor]
    on_cuda: bool = False


BatchResult = Tuple[bool, Union[np.ndarray, Exception]]


@dataclass
class BatchRequest:
    post_id: str
    media_name: str
    frames: List[FrameBatchEntry]
    result_queue: "queue.Queue[BatchResult]"
    total: int = field(init=False)
    sent: int = 0
    buffers: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.total = len(self.frames)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.sent)


class BatchScheduler:
    """Accumulates frame requests across posts and runs batched inference."""

    _STOP = object()

    def __init__(
        self,
        embedder: "Embedder",
        *,
        flush_interval: Optional[float] = None,
        min_batch_hint: Optional[int] = None,
    ) -> None:
        self.embedder = embedder
        self.max_batch = max(1, int(embedder.batch))
        fi = flush_interval
        if fi is None:
            fi = float(os.getenv("BATCH_FLUSH_SEC", "0.10"))
        self.flush_interval = max(0.0, float(fi))
        requested_default = getattr(embedder, "requested_batch", embedder.batch)
        try:
            requested_default = int(requested_default)
        except (TypeError, ValueError):
            requested_default = self.max_batch
        default_min_target = min(self.max_batch, requested_default)
        min_target = _int_env("BATCH_MIN_TARGET", default_min_target)
        requested_min = max(1, min(self.max_batch, min_target))
        self._requested_min_batch = requested_min
        if min_batch_hint is not None:
            try:
                hint = int(min_batch_hint)
            except (TypeError, ValueError):
                hint = requested_min
            hint = max(1, min(self.max_batch, hint))
            effective_min = min(requested_min, hint)
        else:
            hint = None
            effective_min = requested_min
        self._min_batch_hint = hint
        self.min_batch = max(1, effective_min)
        default_idle_grace = self.flush_interval * 4 if self.flush_interval > 0 else 0.0
        idle_grace = _float_env("BATCH_IDLE_GRACE", default_idle_grace)
        if self.flush_interval > 0 and idle_grace < self.flush_interval:
            idle_grace = self.flush_interval
        self.idle_grace = max(0.0, idle_grace)
        self._queue: "queue.Queue[Union[BatchRequest, object]]" = queue.Queue()
        self._batch_count = 0
        self._batch_total = 0
        self._last_log = time.monotonic()
        self._primed = False
        self._thread = threading.Thread(target=self._loop, name="batcher", daemon=True)
        self._thread.start()
        LOG.info(
            "BatchScheduler thresholds: requested_min=%d effective_min=%d max_batch=%d flush_interval=%.3fs idle_grace=%.3fs",
            self._requested_min_batch,
            self.min_batch,
            self.max_batch,
            self.flush_interval,
            self.idle_grace,
        )

    def submit(
        self,
        frames: Sequence[FrameBatchEntry],
        *,
        post_id: str,
        media_name: str,
    ) -> "queue.Queue[BatchResult]":
        result_q: "queue.Queue[BatchResult]" = queue.Queue(maxsize=1)
        req = BatchRequest(
            post_id=post_id,
            media_name=media_name,
            frames=list(frames),
            result_queue=result_q,
        )
        if req.total == 0:
            result_q.put((True, np.zeros((0, 1), dtype=np.float32)))
            return result_q
        self._queue.put(req)
        return result_q

    def close(self) -> None:
        self._queue.put(self._STOP)
        self._thread.join()

    def _loop(self) -> None:
        pending: Deque[BatchRequest] = deque()
        pending_total = 0
        shutdown = False
        last_enqueue_time: Optional[float] = None

        while True:
            item: Optional[Union[BatchRequest, object]] = None
            timed_out = False
            if not shutdown:
                timeout = self.flush_interval if pending_total else None
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    item = None
                    timed_out = True
                if item is self._STOP:
                    shutdown = True
                elif isinstance(item, BatchRequest):
                    pending.append(item)
                    pending_total += item.remaining
                    last_enqueue_time = time.monotonic()
                    item = None
                elif item is None and not pending_total:
                    if shutdown and not pending:
                        break
                    continue

            if not pending and shutdown:
                break

            force_flush = shutdown
            force_flush_idle = False
            if not force_flush and timed_out and pending_total:
                if pending_total >= self.min_batch:
                    force_flush = True
                else:
                    if not self._primed:
                        if self.idle_grace > 0.0 and last_enqueue_time is not None:
                            now = time.monotonic()
                            since_enqueue = now - last_enqueue_time
                            if since_enqueue < self.idle_grace:
                                LOG.debug(
                                    "BatchScheduler priming pending=%d min_batch=%d waiting_for_more=%.3fs",
                                    pending_total,
                                    self.min_batch,
                                    self.idle_grace - since_enqueue,
                                )
                                timed_out = False
                                continue
                    force_flush = True
                    force_flush_idle = True
                    LOG.debug(
                        "BatchScheduler fast-path dispatching starved queue pending=%d min_batch=%d",
                        pending_total,
                        self.min_batch,
                    )

            while pending:
                should_dispatch = pending_total >= self.max_batch
                if not should_dispatch and pending_total >= self.min_batch:
                    should_dispatch = True
                if not should_dispatch and not force_flush:
                    break
                take = min(self.max_batch, pending_total)
                batch: List[FrameBatchEntry] = []
                ownership: List[Tuple[BatchRequest, int, int]] = []

                while len(batch) < take and pending:
                    req = pending[0]
                    remaining = req.remaining
                    if remaining == 0:
                        pending.popleft()
                        continue
                    consume = min(remaining, take - len(batch))
                    start = req.sent
                    batch.extend(req.frames[start : start + consume])
                    ownership.append((req, start, consume))
                    req.sent += consume
                    if req.remaining == 0:
                        pending.popleft()
                    pending_total -= consume

                if not batch:
                    break

                LOG.debug(
                    "Dispatching batch size=%d pending=%d", len(batch), pending_total
                )
                try:
                    embeddings = self.embedder.embed_images(batch)
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOG.exception("Batch inference failed: %s", exc)
                    for req, _, _ in ownership:
                        req.result_queue.put((False, exc))
                    continue

                if not self._primed:
                    self._primed = True

                self._batch_count += 1
                self._batch_total += len(batch)
                now = time.monotonic()
                if self._batch_count and now - self._last_log >= 5.0:
                    avg = self._batch_total / self._batch_count
                    LOG.info("BatchScheduler avg batch=%.2f over %d batches", avg, self._batch_count)
                    self._last_log = now

                offset = 0
                for req, start, count in ownership:
                    part = embeddings[offset : offset + count]
                    offset += count
                    req.buffers.append(part)
                    if req.remaining > 0:
                        continue
                    if req.buffers:
                        if len(req.buffers) == 1:
                            final = req.buffers[0]
                        else:
                            final = np.concatenate(req.buffers, axis=0)
                    else:
                        final = np.zeros((0, 1), dtype=np.float32)
                    req.result_queue.put((True, final))
                if pending_total == 0:
                    last_enqueue_time = None
                if shutdown:
                    force_flush = True
                elif force_flush_idle:
                    force_flush = pending_total > 0
                else:
                    force_flush = False
                timed_out = False

            if shutdown and not pending and pending_total == 0:
                break


@dataclass
class _DecodedItem:
    post_id: str
    name: str
    kind: str
    payload: object
    from_gpu: bool = False


class PrefetchingQueue:
    """Minimal prefetch queue that batches decoded items before embedding."""

    def __init__(
        self,
        embedder,
        *,
        result_queue: "queue.Queue",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.embedder = embedder
        self.result_queue = result_queue
        self.log = logger or LOG
        self._start_threads()

    def _start_threads(self) -> None:  # pragma: no cover - patched in tests
        return

    def _max_batch(self) -> int:
        getter = getattr(self.embedder, "_max_batch", None)
        if getter is None:
            return 1
        try:
            return max(1, int(getter()))
        except Exception:  # pragma: no cover - defensive
            return 1

    def _process_slice(self, slice_items: Sequence[_DecodedItem], active_posts: Dict[str, dict]) -> None:
        if not slice_items:
            return
        batch_size = self._max_batch()
        for start in range(0, len(slice_items), batch_size):
            batch = list(slice_items[start : start + batch_size])
            if not batch:
                continue
            payloads = [item.payload for item in batch]
            use_gpu = all(
                getattr(item, "from_gpu", False)
                or (torch.is_tensor(item.payload) and getattr(item.payload, "is_cuda", False))
                for item in batch
            ) and hasattr(self.embedder, "_forward_gpu_tensors")
            if use_gpu:
                outputs = self.embedder._forward_gpu_tensors(payloads)  # type: ignore[attr-defined]
            else:
                outputs = self.embedder._embed_many(payloads)
            self._attach_embeddings(batch, outputs, active_posts)

    def _attach_embeddings(
        self,
        batch: Sequence[_DecodedItem],
        outputs: Sequence[object],
        active_posts: Dict[str, dict],
    ) -> None:
        for item, embedding in zip(batch, outputs):
            post_state = active_posts.get(item.post_id)
            if not post_state:
                continue
            vector = embedding
            if torch.is_tensor(vector):
                vector = vector.detach().cpu().tolist()
            elif hasattr(vector, "tolist"):
                vector = vector.tolist()
            else:
                vector = list(vector)  # type: ignore[arg-type]

            if item.kind == "image":
                post_state.setdefault("files", []).append(
                    {"name": item.name, "type": "image", "embedding": vector}
                )
                if "pending_images" in post_state:
                    post_state["pending_images"] = max(0, int(post_state["pending_images"]) - 1)
            elif item.kind == "video":
                frames = post_state.setdefault("videos", {}).setdefault(item.name, [])
                frames.append({"index": len(frames), "embedding": vector})


class RemoteUploader:
    """Upload helper that stages payloads on a remote host via SSH."""

    def __init__(
        self,
        *,
        host: str,
        user: str,
        remote_base: PurePosixPath,
        port: int = 22,
        ssh_args: Optional[Sequence[str]] = None,
    ) -> None:
        self.host = host
        self.user = user
        self.remote_base = remote_base
        self.port = int(port)
        self.ssh_args = list(ssh_args or [])

    def _ssh_command(self, remote_cmd: str) -> List[str]:
        return [
            "ssh",
            "-p",
            str(self.port),
            *self.ssh_args,
            f"{self.user}@{self.host}",
            remote_cmd,
        ]

    def _run_ssh(self, remote_cmd: str, *, input_data: Optional[bytes] = None) -> None:
        cmd = self._ssh_command(remote_cmd)
        subprocess.run(cmd, input=input_data, check=True)

    def upload(self, remote_path: PurePosixPath, payload: bytes) -> None:
        full_path = self.remote_base / remote_path
        remote_dir = full_path.parent
        tmp_path = full_path.with_name(full_path.name + ".tmp")
        commands = [
            f"mkdir -p {shlex.quote(remote_dir.as_posix())}",
            f"cat > {shlex.quote(tmp_path.as_posix())}",
            f"mv {shlex.quote(tmp_path.as_posix())} {shlex.quote(full_path.as_posix())}",
        ]
        remote_cmd = " && ".join(commands)
        self._run_ssh(remote_cmd, input_data=payload)


class Embedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        fp16: bool,
        batch_size: int,
        *,
        auto_batch: bool = False,
    ) -> None:
        self.device = device
        self.fast = _bool_env("USE_FAST_PROCESSOR", True)
        self.auto_batch = auto_batch
        self.requested_batch = batch_size
        self.original_batch_arg = "auto" if auto_batch else str(batch_size)
        self.batch = max(1, batch_size)

        dtype = torch.float16 if (fp16 and device == "cuda") else None
        LOG.debug(
            "Loading model=%s device=%s fp16=%s batch=%d use_fast=%s auto_batch=%s",
            model_name,
            device,
            dtype == torch.float16,
            self.batch,
            self.fast,
            auto_batch,
        )
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=self.fast)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self._image_config = self._build_image_config()
        self._transfer_stream = (
            torch.cuda.Stream() if device == "cuda" and torch.cuda.is_available() else None
        )
        self._measure_gpu = _bool_env("MEASURE_GPU_TIME", False)
        self._legacy_preproc = _bool_env("LEGACY_PREPROC", False)
        self._throughput_window: Deque[Tuple[float, int]] = deque()
        self._throughput_images = 0
        self._total_images = 0
        self._last_throughput_log = time.monotonic()
        self._gpu_monitor_stop: Optional[threading.Event] = None
        self._gpu_monitor_thread: Optional[threading.Thread] = None
        self._gpu_monitor_failed = False
        LOG.info("Loaded %s on %s (fp16=%s)", model_name, device, dtype == torch.float16)

        if device == "cuda" and shutil.which("nvidia-smi") is not None:
            try:
                dev_index = torch.cuda.current_device()
            except Exception:
                dev_index = 0
            self._gpu_monitor_stop = threading.Event()
            self._gpu_monitor_thread = threading.Thread(
                target=self._gpu_monitor_loop,
                name="gpu-monitor",
                args=(dev_index,),
                daemon=True,
            )
            self._gpu_monitor_thread.start()

        if self.auto_batch:
            try:
                tuned = self._tune_batch_size(sample=None)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.warning(
                    "Auto batch-size warmup failed (%s); keeping batch=%d",
                    exc,
                    self.batch,
                )
            else:
                if tuned > 0:
                    self.batch = tuned
                LOG.info(
                    "Auto batch-size tuned to %d (requested=%s)",
                    self.batch,
                    self.original_batch_arg,
                )

    def _infer_processor_resolution(self) -> tuple[int, int]:
        default = (384, 384)
        proc = getattr(self.processor, "image_processor", None)
        if proc is None:
            return default

        def _normalize(size) -> Optional[tuple[int, int]]:
            if size is None:
                return None
            if isinstance(size, dict):
                h = size.get("height") or size.get("shortest_edge") or size.get("shortest_size")
                w = size.get("width") or size.get("shortest_edge") or size.get("shortest_size")
                if h and w:
                    return int(h), int(w)
            if isinstance(size, (list, tuple)):
                if len(size) == 2:
                    return int(size[0]), int(size[1])
                if len(size) == 1:
                    val = int(size[0])
                    return val, val
            if isinstance(size, int):
                val = int(size)
                return val, val
            if hasattr(size, "height") and hasattr(size, "width"):
                try:
                    return int(size.height), int(size.width)
                except Exception:  # pragma: no cover - defensive
                    return None
            return None

        for attr in ("crop_size", "size", "image_size", "resample_size"):
            dims = _normalize(getattr(proc, attr, None))
            if dims:
                return dims
        return default

    def _synthetic_sample(self) -> np.ndarray:
        h, w = self._infer_processor_resolution()
        h = max(1, int(h))
        w = max(1, int(w))
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _gpu_monitor_loop(self, device_index: int) -> None:
        if self._gpu_monitor_stop is None:
            return
        while not self._gpu_monitor_stop.wait(10.0):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(device_index),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                output = result.stdout.strip()
                if not output:
                    continue
                parts = [p.strip() for p in output.split(",") if p.strip()]
                if len(parts) >= 4:
                    gpu_util, mem_util, mem_used, mem_total = parts[:4]
                    LOG.info(
                        "GPU UTIL: gpu=%s%% mem=%s%% (%s/%s MiB)",
                        gpu_util,
                        mem_util,
                        mem_used,
                        mem_total,
                    )
                else:
                    LOG.info("GPU UTIL: %s", output)
            except Exception as exc:
                if not self._gpu_monitor_failed:
                    LOG.debug("GPU util probe failed: %s", exc)
                    self._gpu_monitor_failed = True
                return

    def _tune_batch_size(self, sample: Optional[List[np.ndarray]]) -> int:
        if self.device != "cuda" or not torch.cuda.is_available():
            LOG.info(
                "Auto batch-size requested but CUDA unavailable; keeping batch=%d",
                self.batch,
            )
            return self.batch

        base_frames: List[np.ndarray] = []
        if sample:
            for frame in sample:
                arr = np.asarray(frame)
                if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim != 3 or arr.shape[-1] != 3:
                    continue
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                base_frames.append(arr)

        if not base_frames:
            base_frames = [self._synthetic_sample()]

        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        reserve_bytes = max(int(total_mem * 0.08), 256 * 1024 * 1024)
        max_tune = max(1, _int_env("BATCH_TUNE_MAX", 2048))

        LOG.info(
            "Starting auto batch-size warmup (requested=%s, initial=%d, free=%.1f MiB)",
            self.original_batch_arg,
            self.batch,
            free_mem / (1024 ** 2),
        )

        def _make_batch(bs: int) -> List[np.ndarray]:
            return [base_frames[i % len(base_frames)] for i in range(bs)]

        oom_errors = (RuntimeError,)
        if hasattr(torch.cuda, "OutOfMemoryError"):
            oom_errors = (RuntimeError, torch.cuda.OutOfMemoryError)

        def _try_batch(bs: int) -> bool:
            if bs <= 0:
                return False
            LOG.debug("Warmup trial batch=%d", bs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                free, total = torch.cuda.mem_get_info()
                reserve = max(int(total * 0.08), reserve_bytes)
                if free <= reserve:
                    LOG.debug(
                        "Skipping batch=%d: free memory %.1f MiB below reserve %.1f MiB",
                        bs,
                        free / (1024 ** 2),
                        reserve / (1024 ** 2),
                    )
                    return False

            inputs = None
            outputs = None
            try:
                batch_images = _make_batch(bs)
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                return True
            except oom_errors as err:
                message = str(err).lower()
                if "out of memory" in message:
                    LOG.debug("CUDA OOM for batch=%d: %s", bs, err)
                    return False
                raise
            finally:
                if outputs is not None:
                    del outputs
                if inputs is not None:
                    del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        best = 0
        failure = None
        goal = max(self.batch, self.requested_batch, 1)

        if self.auto_batch:
            free_after_reserve = max(free_mem - reserve_bytes, 0)
            approx_frame_bytes = 0
            if base_frames:
                # Rough heuristic: processor normalizes frames to float32 tensors.
                approx_frame_bytes = base_frames[0].nbytes * 4
            mem_cap = 0
            if approx_frame_bytes > 0:
                mem_cap = int(free_after_reserve // approx_frame_bytes)
            upper_bound = max(goal, mem_cap)
            if upper_bound <= 0:
                upper_bound = goal
            max_cap = min(max(upper_bound, goal), max_tune)
            LOG.info(
                "Auto batch-size warmup upper bound=%d (goal=%d, free=%.1f MiB, reserve=%.1f MiB)",
                max_cap,
                goal,
                free_mem / (1024 ** 2),
                reserve_bytes / (1024 ** 2),
            )
        else:
            max_cap = min(max(goal * 4, goal), max_tune)

        candidate = 1

        while candidate <= max_cap:
            if _try_batch(candidate):
                best = candidate
                candidate *= 2
            else:
                failure = candidate
                break

        if best == 0:
            best = 1

        if failure is None:
            selected = max(best, 1)
            if torch.cuda.is_available():
                free_after, total_after = torch.cuda.mem_get_info()
                free_mb = free_after / (1024 ** 2)
                total_mb = total_after / (1024 ** 2)
            else:
                free_mb = total_mb = 0.0
            LOG.info(
                "Auto batch-size result=%d (free≈%.1f MiB, total≈%.1f MiB)",
                selected,
                free_mb,
                total_mb,
            )
            return selected

        lo = best + 1
        hi = failure - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _try_batch(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        selected = max(best, 1)
        if torch.cuda.is_available():
            free_after, total_after = torch.cuda.mem_get_info()
            free_mb = free_after / (1024 ** 2)
            total_mb = total_after / (1024 ** 2)
        else:
            free_mb = total_mb = 0.0
        LOG.info(
            "Auto batch-size result=%d (free≈%.1f MiB, total≈%.1f MiB)",
            selected,
            free_mb,
            total_mb,
        )
        return selected

    def _build_image_config(self) -> Optional[dict]:
        proc = getattr(self.processor, "image_processor", None)
        if proc is None:
            return None

        def _to_hw(size_obj) -> Optional[Tuple[int, int]]:
            if size_obj is None:
                return None
            if isinstance(size_obj, dict):
                h = (
                    size_obj.get("height")
                    or size_obj.get("shortest_edge")
                    or size_obj.get("shortest_size")
                )
                w = (
                    size_obj.get("width")
                    or size_obj.get("shortest_edge")
                    or size_obj.get("shortest_size")
                )
                if h and w:
                    return int(h), int(w)
            if isinstance(size_obj, (list, tuple)):
                if len(size_obj) == 2:
                    return int(size_obj[0]), int(size_obj[1])
                if len(size_obj) == 1:
                    val = int(size_obj[0])
                    return val, val
            if isinstance(size_obj, int):
                return int(size_obj), int(size_obj)
            if hasattr(size_obj, "height") and hasattr(size_obj, "width"):
                try:
                    return int(size_obj.height), int(size_obj.width)
                except Exception:
                    return None
            return None

        resize_shorter = None
        size_raw = getattr(proc, "size", None)
        if isinstance(size_raw, dict):
            for key in ("shortest_edge", "shortest_size", "shortest_side"):
                if key in size_raw:
                    resize_shorter = int(size_raw[key])
                    break

        resize_hw = _to_hw(size_raw)
        crop_hw = _to_hw(getattr(proc, "crop_size", None))

        resample = getattr(proc, "resample", None)
        mode = "bilinear"
        align_corners = False
        if resample is not None:
            try:
                from PIL import Image as _PILImage

                if resample == _PILImage.Resampling.NEAREST:
                    mode = "nearest"
                    align_corners = None
                elif resample == _PILImage.Resampling.BICUBIC:
                    mode = "bicubic"
                elif resample == _PILImage.Resampling.BILINEAR:
                    mode = "bilinear"
                elif resample == _PILImage.Resampling.LANCZOS:
                    mode = "bicubic"
                else:
                    mode = "bilinear"
            except Exception:
                mode = "bilinear"
        else:
            mode = "bilinear"

        mean = torch.tensor(
            getattr(proc, "image_mean", [0.5, 0.5, 0.5]),
            device=self.device,
            dtype=self.model_dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            getattr(proc, "image_std", [0.5, 0.5, 0.5]),
            device=self.device,
            dtype=self.model_dtype,
        ).view(1, 3, 1, 1)

        return {
            "do_resize": bool(getattr(proc, "do_resize", False)),
            "resize_shorter": resize_shorter,
            "resize_hw": resize_hw,
            "resize_mode": mode,
            "resize_align_corners": align_corners,
            "do_center_crop": bool(getattr(proc, "do_center_crop", False)),
            "crop_hw": crop_hw,
            "do_rescale": bool(getattr(proc, "do_rescale", False)),
            "rescale_factor": float(getattr(proc, "rescale_factor", 1.0)),
            "do_normalize": bool(getattr(proc, "do_normalize", False)),
            "mean": mean,
            "std": std,
        }

    def _unwrap_payload(self, item: Union[np.ndarray, torch.Tensor, FrameBatchEntry]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(item, FrameBatchEntry):
            return item.data
        return item

    def _to_chw_tensor(self, payload: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(payload):
            t = payload
        else:
            arr = np.asarray(payload)
            t = torch.from_numpy(arr)

        if t.ndim != 3:
            raise ValueError(f"Expected 3D tensor/array, got shape={tuple(t.shape)}")

        if t.shape[0] == 3:
            chw = t
        elif t.shape[-1] == 3:
            chw = t.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported channel layout: {tuple(t.shape)}")

        if chw.device.type == "cuda":
            return chw.contiguous()

        chw = chw.contiguous()
        if chw.device.type == "cpu":
            try:
                chw = chw.pin_memory()
            except RuntimeError:
                pass
        return chw

    def _compute_resize_dims(self, h: int, w: int, config: dict) -> Tuple[int, int]:
        resize_shorter = config.get("resize_shorter")
        resize_hw = config.get("resize_hw")
        if resize_shorter:
            short = min(h, w)
            if short <= 0:
                return max(1, h), max(1, w)
            scale = resize_shorter / short
            if scale != 1.0:
                new_h = max(1, int(round(h * scale)))
                new_w = max(1, int(round(w * scale)))
                return new_h, new_w
        if resize_hw:
            return max(1, int(resize_hw[0])), max(1, int(resize_hw[1]))
        return max(1, int(h)), max(1, int(w))

    def _center_crop(self, tensor: torch.Tensor, crop_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
        if not crop_hw:
            return tensor
        crop_h, crop_w = crop_hw
        if crop_h <= 0 or crop_w <= 0:
            return tensor

        squeeze = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze = True

        if tensor.ndim != 4:
            return tensor.squeeze(0) if squeeze else tensor

        h, w = tensor.shape[-2:]
        if crop_h > h or crop_w > w:
            pad_top = max((crop_h - h) // 2, 0)
            pad_bottom = max(crop_h - h - pad_top, 0)
            pad_left = max((crop_w - w) // 2, 0)
            pad_right = max(crop_w - w - pad_left, 0)
            if pad_top or pad_bottom or pad_left or pad_right:
                tensor = F.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                )
            h, w = tensor.shape[-2:]

        top = max((h - crop_h) // 2, 0)
        left = max((w - crop_w) // 2, 0)
        tensor = tensor[:, :, top : top + crop_h, left : left + crop_w]
        if squeeze:
            tensor = tensor.squeeze(0)
        return tensor

    def _apply_image_transforms_single(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._image_config is None:
            return tensor.to(self.model_dtype)

        cfg = self._image_config
        if tensor.dtype != torch.float32 and tensor.dtype != torch.float16:
            tensor = tensor.float()

        if cfg.get("do_rescale"):
            tensor = tensor * cfg["rescale_factor"]
        else:
            tensor = tensor / 255.0

        if cfg.get("do_resize"):
            h, w = tensor.shape[-2:]
            target_h, target_w = self._compute_resize_dims(h, w, cfg)
            if (target_h, target_w) != (h, w):
                mode = cfg.get("resize_mode", "bilinear")
                align = cfg.get("resize_align_corners")
                kwargs = {}
                if mode in {"bilinear", "bicubic"}:
                    kwargs["align_corners"] = bool(align is True)
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(target_h, target_w),
                    mode=mode,
                    **kwargs,
                ).squeeze(0)

        if cfg.get("do_center_crop") and cfg.get("crop_hw"):
            tensor = self._center_crop(tensor, cfg["crop_hw"])

        tensor = tensor.to(self.model_dtype)

        if cfg.get("do_normalize"):
            tensor = (tensor - cfg["mean"]) / cfg["std"]

        return tensor

    def _apply_image_transforms_batch(
        self,
        batch: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self._image_config is None:
            return batch.to(self.model_dtype)

        cfg = self._image_config
        if batch.dtype not in (torch.float16, torch.float32):
            batch = batch.float()

        if cfg.get("do_rescale"):
            batch = batch * cfg["rescale_factor"]
        else:
            batch = batch / 255.0

        if cfg.get("do_resize"):
            mode = cfg.get("resize_mode", "bilinear")
            kwargs = {}
            if mode in {"bilinear", "bicubic"}:
                kwargs["align_corners"] = bool(cfg.get("resize_align_corners"))
            if target_hw is None:
                h, w = batch.shape[-2:]
                target_hw = self._compute_resize_dims(h, w, cfg)
            tgt_h, tgt_w = target_hw
            if (tgt_h, tgt_w) != batch.shape[-2:]:
                batch = F.interpolate(batch, size=(tgt_h, tgt_w), mode=mode, **kwargs)

        if cfg.get("do_center_crop") and cfg.get("crop_hw"):
            batch = self._center_crop(batch, cfg["crop_hw"])

        batch = batch.to(self.model_dtype)

        if cfg.get("do_normalize"):
            batch = (batch - cfg["mean"]) / cfg["std"]

        return batch

    def _stack_group(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.empty((0, 3, 0, 0), device=self.device, dtype=self.model_dtype)
        max_h = max(int(t.shape[-2]) for t in tensors)
        max_w = max(int(t.shape[-1]) for t in tensors)
        same_shape = all(
            int(t.shape[-2]) == max_h and int(t.shape[-1]) == max_w for t in tensors
        )
        if same_shape:
            return torch.stack([t.contiguous() for t in tensors], dim=0)

        stacked: List[torch.Tensor] = []
        for tensor in tensors:
            h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
            if h == max_h and w == max_w:
                stacked.append(tensor.contiguous())
                continue
            pad_h = max_h - h
            pad_w = max_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            padded = F.pad(
                tensor.unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            ).squeeze(0)
            stacked.append(padded.contiguous())
        return torch.stack(stacked, dim=0)

    def _preprocess_batch_legacy(
        self, chunk: Sequence[Union[np.ndarray, torch.Tensor, FrameBatchEntry]]
    ) -> torch.Tensor:
        processed: List[torch.Tensor] = []
        for item in chunk:
            payload = self._unwrap_payload(item)
            tensor = self._to_chw_tensor(payload)
            stream = self._transfer_stream
            if stream is not None:
                with torch.cuda.stream(stream):
                    if tensor.device.type != self.device:
                        tensor = tensor.to(self.device, non_blocking=True)
                    tensor = self._apply_image_transforms_single(tensor)
            else:
                if tensor.device.type != self.device:
                    tensor = tensor.to(self.device, non_blocking=True)
                tensor = self._apply_image_transforms_single(tensor)
            processed.append(tensor)

        if self._transfer_stream is not None:
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

        return torch.stack(processed, dim=0)

    def _preprocess_batch(
        self, chunk: Sequence[Union[np.ndarray, torch.Tensor, FrameBatchEntry]]
    ) -> torch.Tensor:
        if not chunk:
            return torch.empty((0, 3, 0, 0), device=self.device, dtype=self.model_dtype)

        if self._image_config is None:
            images: List[np.ndarray] = []
            for item in chunk:
                payload = self._unwrap_payload(item)
                if torch.is_tensor(payload):
                    t = payload.detach().cpu()
                    if t.ndim == 3 and t.shape[0] == 3:
                        images.append(t.permute(1, 2, 0).numpy())
                    else:
                        images.append(np.array(t.numpy()))
                else:
                    arr = np.asarray(payload)
                    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    images.append(arr)
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model_dtype)
            return pixel_values

        if self._legacy_preproc:
            LOG.debug("LEGACY_PREPROC=1 → using per-image preprocessing path")
            return self._preprocess_batch_legacy(chunk)

        tensors: List[torch.Tensor] = []
        shapes: List[Tuple[int, int]] = []
        for item in chunk:
            payload = self._unwrap_payload(item)
            tensor = self._to_chw_tensor(payload)
            shapes.append((int(tensor.shape[-2]), int(tensor.shape[-1])))
            tensors.append(tensor)

        stream = self._transfer_stream
        device_tensors: List[torch.Tensor] = []
        for tensor in tensors:
            if stream is not None:
                with torch.cuda.stream(stream):
                    if tensor.device.type != self.device:
                        tensor = tensor.to(self.device, non_blocking=True)
                    tensor = tensor.contiguous()
                device_tensors.append(tensor)
            else:
                if tensor.device.type != self.device:
                    tensor = tensor.to(self.device, non_blocking=True)
                device_tensors.append(tensor.contiguous())

        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)

        cfg = self._image_config
        if cfg.get("do_resize"):
            groups: Dict[Tuple[int, int], List[int]] = {}
            for idx, (h, w) in enumerate(shapes):
                target_hw = self._compute_resize_dims(h, w, cfg)
                groups.setdefault(target_hw, []).append(idx)
        else:
            groups = {(-1, -1): list(range(len(device_tensors)))}

        outputs: Optional[torch.Tensor] = None
        for target_hw, indices in groups.items():
            subset = [device_tensors[i] for i in indices]
            stacked = self._stack_group(subset)
            transformed = self._apply_image_transforms_batch(
                stacked, target_hw=None if target_hw == (-1, -1) else target_hw
            )
            if outputs is None:
                out_shape = (len(chunk),) + tuple(transformed.shape[1:])
                outputs = torch.empty(out_shape, device=transformed.device, dtype=transformed.dtype)
            for offset, original_idx in enumerate(indices):
                outputs[original_idx] = transformed[offset]

        if outputs is None:
            return torch.empty((0, 3, 0, 0), device=self.device, dtype=self.model_dtype)
        return outputs

    def _reshape_pixel_batch(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """Normalize pixel batch layout before running the vision tower."""

        if pixel_values.ndim < 3:
            raise ValueError(
                f"Expected at least 3 dims for pixel batch, got shape={tuple(pixel_values.shape)}"
            )

        # Allow passing a single image tensor without an explicit batch dimension.
        if pixel_values.ndim == 3:
            return pixel_values.unsqueeze(0), None

        # Handle batched inputs with multi-view crops (B, V, C, H, W).
        if pixel_values.ndim == 5:
            batch, views = int(pixel_values.shape[0]), int(pixel_values.shape[1])
            if batch <= 0 or views <= 0:
                reshaped = pixel_values.reshape(-1, *pixel_values.shape[-3:])
                return reshaped, None
            reshaped = pixel_values.reshape(batch * views, *pixel_values.shape[-3:])
            return reshaped, (batch, views)

        if pixel_values.ndim != 4:
            # Collapse extra singleton dimensions if present (e.g. [B,1,C,H,W]).
            squeezed = pixel_values
            while squeezed.ndim > 4:
                if squeezed.shape[1] != 1:
                    raise ValueError(
                        f"Unsupported pixel batch shape (cannot squeeze to 4D): {tuple(pixel_values.shape)}"
                    )
                squeezed = squeezed.squeeze(1)
            if squeezed.ndim != 4:
                raise ValueError(
                    f"Expected 4D tensor after squeeze, got shape={tuple(squeezed.shape)}"
                )
            return squeezed, None

        return pixel_values, None

    def _forward_features(
        self, pixel_values: torch.Tensor, *, measure_gpu_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[float]]:
        pixel_values, multi_view = self._reshape_pixel_batch(pixel_values)

        def _run() -> torch.Tensor:
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if self.device == "cuda"
                else contextlib.nullcontext()
            )
            with torch.inference_mode():
                with autocast_ctx:
                    feats = self.model.get_image_features(pixel_values=pixel_values)
            if multi_view is None:
                return feats
            batch, views = multi_view
            if feats.ndim < 2:
                return feats
            feats = feats.view(batch, views, -1).mean(dim=1)
            return feats

        if not measure_gpu_time:
            feats = _run()
            return feats, None

        if not torch.cuda.is_available():
            raise RuntimeError("GPU timing requested but CUDA is unavailable")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        feats = _run()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        return feats, float(elapsed)

    def close(self) -> None:
        if self._gpu_monitor_stop is not None:
            self._gpu_monitor_stop.set()
        if self._gpu_monitor_thread is not None:
            try:
                self._gpu_monitor_thread.join(timeout=1.0)
            except RuntimeError:
                pass
        self._gpu_monitor_thread = None
        self._gpu_monitor_stop = None

    @torch.no_grad()
    def embed_images(
        self, rgbs: List[Union[np.ndarray, torch.Tensor, FrameBatchEntry]]
    ) -> np.ndarray:
        if not rgbs:
            return np.zeros((0, 1), dtype=np.float32)
        outs: List[torch.Tensor] = []
        for i in range(0, len(rgbs), self.batch):
            chunk = rgbs[i : i + self.batch]
            LOG.debug("Embedding batch %d..%d (size=%d)", i, i + len(chunk) - 1, len(chunk))
            pixel_values = self._preprocess_batch(chunk)
            feats, gpu_time = self._forward_features(
                pixel_values, measure_gpu_time=self._measure_gpu
            )
            if gpu_time is not None:
                LOG.debug(
                    "GPU batch time=%.4fs size=%d", gpu_time, int(pixel_values.shape[0])
                )
            feats = F.normalize(feats, p=2, dim=-1)
            outs.append(feats.detach().cpu())
            processed = len(chunk)
            now = time.monotonic()
            self._total_images += processed
            self._throughput_window.append((now, processed))
            self._throughput_images += processed
            while self._throughput_window and now - self._throughput_window[0][0] > 5.0:
                _, removed = self._throughput_window.popleft()
                self._throughput_images = max(0, self._throughput_images - removed)
            if now - self._last_throughput_log >= 5.0 and self._throughput_images > 0:
                window_start = self._throughput_window[0][0]
                window_sec = max(now - window_start, 1e-6)
                imgs = self._throughput_images
                rate = imgs / window_sec
                LOG.info(
                    "THROUGHPUT: %d imgs in %.3fs ⇒ %.1f img/s | batch=%d",
                    imgs,
                    window_sec,
                    rate,
                    processed,
                )
                self._last_throughput_log = now
        stacked = torch.cat(outs, dim=0).numpy()
        LOG.debug("Embedded %d images → shape=%s", len(rgbs), stacked.shape)
        return stacked


# --------------------------
# Decoding
# --------------------------

class MalformedVideoError(Exception):
    """Raised when the video cannot be opened/decoded at all."""


class Decoder:
    def __init__(self, stride_sec: float, prefer_gpu: bool) -> None:
        self.stride = float(stride_sec)
        self.prefer_gpu = prefer_gpu and torch.cuda.is_available()
        LOG.debug("Decoder init: stride=%.3f prefer_gpu=%s cuda=%s",
                  self.stride, self.prefer_gpu, torch.cuda.is_available())

    def frames_from_video(self, path: str) -> Iterator[FrameBatchEntry]:
        LOG.debug("Decoding video: %s (prefer_gpu=%s)", path, self.prefer_gpu)
        init_errors: List[str] = []

        # Try GPU first (if requested), then CPU
        dec = None
        if self.prefer_gpu:
            try:
                dec = DaliVideoDecoder(path, stride_sec=self.stride, use_gpu=True, device_id=torch.cuda.current_device())
                mode = "gpu"
            except Exception as e:
                init_errors.append(f"gpu_init: {e}")
                LOG.debug("GPU decoder failed (%s), falling back to CPU", e)

        if dec is None:
            try:
                dec = DaliVideoDecoder(path, stride_sec=self.stride, use_gpu=False)
                mode = "cpu"
            except Exception as e:
                init_errors.append(f"cpu_init: {e}")
                # Both inits failed → treat as malformed/unreadable
                reason = "; ".join(init_errors) or "unknown"
                raise MalformedVideoError(reason)

        def _iter_frames() -> Iterator[FrameBatchEntry]:
            count = 0
            try:
                for f in dec.frames():
                    if isinstance(f.payload, np.ndarray):
                        arr = f.payload
                        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                            arr = np.transpose(arr, (1, 2, 0))
                        entry = FrameBatchEntry(data=arr, on_cuda=False)
                    elif torch.is_tensor(f.payload):
                        t = f.payload
                        if t.ndim != 3:
                            raise ValueError(f"Unexpected tensor frame ndim={t.ndim}")
                        if t.shape[0] == 3:
                            tensor = t.contiguous()
                        elif t.shape[-1] == 3:
                            tensor = t.permute(2, 0, 1).contiguous()
                        else:
                            raise ValueError(f"Unexpected tensor frame shape: {tuple(t.shape)}")
                        entry = FrameBatchEntry(
                            data=tensor,
                            on_cuda=bool(tensor.is_cuda or f.device == "gpu"),
                        )
                    else:
                        arr = np.array(f.payload)
                        entry = FrameBatchEntry(data=arr, on_cuda=False)

                    count += 1
                    yield entry
            finally:
                LOG.debug("Decoded %d frames (%s)", count, mode)
                try:
                    dec.close()
                except Exception:
                    pass

        return _iter_frames()


# --------------------------
# Minimal pipeline
# --------------------------

def _normalize_rel(base: PurePosixPath, rel: PurePosixPath) -> PurePosixPath:
    """Strip accidental leading copies of the remote base from a worklist relative_path."""
    base_parts = tuple(p for p in base.parts if p != "/")
    parts = tuple(p for p in rel.parts if p != "/")
    while len(parts) >= len(base_parts) and tuple(parts[: len(base_parts)]) == base_parts:
        parts = parts[len(base_parts) :]
    return PurePosixPath(*parts)


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.remote = RemoteIO()
        self.workdir = Path(args.workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.down_q: "queue.Queue[Optional[WorkItem]]" = queue.Queue()
        self.proc_q: "queue.Queue[Optional[Downloaded]]" = queue.Queue()
        self.up_q: "queue.Queue[Optional[UploadTask]]" = queue.Queue()
        self.t0 = time.monotonic()
        self.stats = {
            "processed": 0,
            "images": 0,
            "videos": 0,
            "frames": 0,
            "bytes_out": 0,
            "errors": 0,
        }
        self.stats_lock = threading.Lock()
        self._progress_cond = threading.Condition(self.stats_lock)
        self.compute_only = _bool_env("COMPUTE_ONLY", False)

        LOG.debug("Pipeline init: workdir=%s. Auto batch=%s. Batch size=%d", self.workdir, getattr(args, "auto_batch", False), args.batch_size)

        # Model + decoder
        if not torch.cuda.is_available():
            raise RuntimeError("Missing CUDA")
        device = "cuda"
        self.embedder = Embedder(
            args.model,
            device,
            _bool_env("FP16", True),
            args.batch_size,
            auto_batch=getattr(args, "auto_batch", False),
        )
        self.decoder = Decoder(args.stride_sec, prefer_gpu=(device == "cuda"))
        self.batch_coordinator: Optional[BatchScheduler] = None
        default_workers = max(1, (os.cpu_count() or 4) - 1)
        raw_max_workers = getattr(args, "max_workers", None)
        if raw_max_workers is None:
            env_default = _int_env("MAX_WORKERS", _int_env("PROC_WORKERS", default_workers))
        else:
            env_default = raw_max_workers
        try:
            worker_count = max(1, int(env_default))
        except (TypeError, ValueError):
            worker_count = default_workers
        self.worker_count = worker_count
        self.batch_coordinator = BatchScheduler(
            self.embedder, min_batch_hint=self.worker_count
        )
        self.proc_workers = [
            threading.Thread(target=self._process_worker, name=f"processor-{i}", daemon=True)
            for i in range(self.worker_count)
        ]

        self.download_worker_count = max(
            1,
            int(
                getattr(
                    args,
                    "download_workers",
                    _int_env("DOWNLOAD_WORKERS", max(2, min(4, self.worker_count))),
                )
            ),
        )
        self.upload_worker_count = max(
            1,
            int(
                getattr(
                    args,
                    "upload_workers",
                    _int_env("UPLOAD_WORKERS", max(2, min(4, self.worker_count))),
                )
            ),
        )
        LOG.debug(
            "Pipeline init: workdir=%s device=%s max_workers=%d dl_workers=%d ul_workers=%d",
            self.workdir,
            device,
            self.worker_count,
            self.download_worker_count,
            self.upload_worker_count,
        )

        self.download_threads = [
            threading.Thread(target=self._downloader, name=f"downloader-{i}", daemon=True)
            for i in range(self.download_worker_count)
        ]
        self.upload_threads = [
            threading.Thread(target=self._uploader, name=f"uploader-{i}", daemon=True)
            for i in range(self.upload_worker_count)
        ]

    # ---------- Threads ----------
    def _close_batch_coordinator(self) -> None:
        if self.batch_coordinator is not None:
            self.batch_coordinator.close()
            self.batch_coordinator = None

    def _run_compute_only(self) -> None:
        if self.batch_coordinator is None:
            hint = getattr(self, "worker_count", None)
            self.batch_coordinator = BatchScheduler(
                self.embedder, min_batch_hint=hint
            )
        iterations = max(1, _int_env("COMPUTE_ONLY_ITERS", 50))
        batch = max(1, self.embedder.batch)
        LOG.info("COMPUTE_ONLY mode active: iterations=%d batch=%d", iterations, batch)
        sample = self.embedder._synthetic_sample()
        total = 0
        t0 = time.monotonic()
        for idx in range(iterations):
            frames = [FrameBatchEntry(data=sample, on_cuda=False) for _ in range(batch)]
            request = self.batch_coordinator.submit(
                frames,
                post_id=f"compute-only-{idx}",
                media_name="synthetic",
            )
            ok, payload = request.get()
            if not ok:
                raise RuntimeError(f"Synthetic batch failed: {payload}")
            total += batch
        elapsed = max(time.monotonic() - t0, 1e-6)
        LOG.info(
            "COMPUTE_ONLY throughput: %d images in %.3fs ⇒ %.2f img/s",
            total,
            elapsed,
            total / elapsed,
        )

    def _downloader(self) -> None:
        while True:
            item = self.down_q.get()
            if item is None:
                LOG.debug("Downloader exit signal")
                self.down_q.task_done()
                break
            try:
                local_dir = self.workdir / "input" / item.post_id
                if local_dir.exists():
                    removed = 0
                    for p in local_dir.glob("**/*"):
                        try:
                            if p.is_file():
                                p.unlink()
                                removed += 1
                        except Exception:
                            pass
                    LOG.debug("Cleaned %s (removed %d files)", local_dir, removed)
                LOG.debug("Downloading %s → %s", item.rel, local_dir)
                self.remote.download_dir(item.rel, local_dir)
                try:
                    listing = sorted([e.name for e in local_dir.iterdir()])
                    LOG.debug("Post %s contents: %s", item.post_id, listing[:20])
                except Exception:
                    pass
                self.proc_q.put(Downloaded(item=item, local_dir=local_dir))
                LOG.debug("Downloaded %s", item.rel)
            except Exception as e:
                LOG.error("Download failed for %s: %s", item.rel, e)
            finally:
                self.down_q.task_done()

    def _uploader(self) -> None:
        while True:
            task = self.up_q.get()
            if task is None:
                LOG.debug("Uploader exit signal")
                self.up_q.task_done()
                break
            try:
                LOG.debug("Uploading %s (bytes=%d)", task.remote_rel, len(task.payload))
                self.remote.upload_bytes(task.remote_rel, task.payload)
                LOG.info("Uploaded %s", task.remote_rel)
            except Exception as e:
                LOG.error("Upload failed for %s: %s", task.remote_rel, e)
            finally:
                self.up_q.task_done()

    def _process_worker(self) -> None:
        while True:
            dl = self.proc_q.get()
            if dl is None:
                LOG.debug("Processor exit signal")
                self.proc_q.task_done()
                break
            try:
                self._process_one(dl)
            except Exception as exc:
                LOG.exception("Processing failed for %s: %s", getattr(dl.item, "post_id", "?"), exc)
                self._inc_stat("errors")
            finally:
                self.proc_q.task_done()
                self._record_processed()

    def _inc_stat(self, key: str, delta: int = 1) -> None:
        with self.stats_lock:
            self.stats[key] += delta

    def _record_processed(self) -> None:
        with self._progress_cond:
            self.stats["processed"] += 1
            self._progress_cond.notify_all()

    # ---------- Helpers ----------
    @staticmethod
    def _resolve_expected_media(dir_: Path, filename: Optional[PurePosixPath]) -> Optional[Path]:
        if filename is None:
            return None
        parts = [part for part in filename.parts if part]
        if not parts:
            return None
        if any(part in {".", ".."} for part in parts):
            LOG.warning("Ignoring unsafe filename path: %s", filename)
            return None
        candidate = dir_.joinpath(*parts)
        if candidate.exists() and candidate.is_file():
            return candidate
        LOG.warning("Expected media not found at %s; falling back to scan", candidate)
        return None

    @staticmethod
    def _find_single_media(dir_: Path) -> Optional[Path]:
        for entry in sorted(dir_.glob("*")):
            if entry.is_file() and entry.suffix.lower() in (IMG_EXT | VID_EXT):
                return entry
        return None

    def _process_one(self, dl: Downloaded) -> None:
        expected = dl.item.filename
        if expected is not None:
            media = self._resolve_expected_media(dl.local_dir, expected)
        else:
            media = self._find_single_media(dl.local_dir)
        rel_out = dl.item.rel / RESULT_NAME
        result: Dict[str, object] = {"id": dl.item.post_id, "files": []}
        if dl.item.profile:
            result["profile"] = dl.item.profile

        if media is None:
            try:
                listing = sorted([e.name for e in dl.local_dir.iterdir()])
            except Exception:
                listing = []
            if expected is not None:
                LOG.warning(
                    "[%s] Expected media %s missing in %s. Contents=%s",
                    dl.item.post_id,
                    expected,
                    dl.local_dir,
                    listing[:50],
                )
                result["files"] = [{
                    "name": expected.as_posix(),
                    "error": "missing",
                }]
            else:
                LOG.warning("[%s] No media in %s. Contents=%s", dl.item.post_id, dl.local_dir, listing[:50])
            payload = _to_json_gz_bytes(result)
            self.stats["bytes_out"] += len(payload)
            self.up_q.put(UploadTask(rel_out, payload))
            return

        LOG.debug("[%s] Found media: %s", dl.item.post_id, media)

        if media.suffix.lower() in IMG_EXT:
            try:
                rgb = np.array(Image.open(media).convert("RGB"))
                LOG.debug("[%s] Image size=%s", dl.item.post_id, getattr(media.stat(), 'st_size', None))
                request = self.batch_coordinator.submit(
                    [FrameBatchEntry(data=rgb, on_cuda=False)],
                    post_id=dl.item.post_id,
                    media_name=media.name,
                )
                ok, payload = request.get()
                if not ok:
                    raise payload  # type: ignore[misc]
                embedding = payload[0]
                self._inc_stat("images")
                result["files"] = [
                    {"name": media.name, "type": "image", "embedding": embedding.tolist()}
                ]
            except UnidentifiedImageError:
                self._inc_stat("images")
                self._inc_stat("errors")
                LOG.debug("[%s] PIL failed to open image", dl.item.post_id)
                result["files"] = [{"name": media.name, "type": "image", "error": "decode_failed"}]
            except Exception as exc:
                self._inc_stat("images")
                self._inc_stat("errors")
                LOG.error("[%s] Image embedding failed: %s", dl.item.post_id, exc)
                result["files"] = [{"name": media.name, "type": "image", "error": "embed_failed"}]
        else:  # video
            LOG.debug("[%s] Decoding video frames (stride=%.3fs)", dl.item.post_id, self.decoder.stride)
            self._inc_stat("videos")
            try:
                frames_iter = self.decoder.frames_from_video(str(media))
            except MalformedVideoError as e:
                LOG.warning("[%s] Video unreadable/malformed: %s", dl.item.post_id, e)
                result["files"] = [{
                    "name": media.name,
                    "type": "video",
                    "error": "malformed_video",
                    "message": str(e),
                }]
                payload = _to_json_gz_bytes(result)
                self._inc_stat("bytes_out", len(payload))
                self.up_q.put(UploadTask(rel_out, payload))
                return
            except Exception as exc:
                self._inc_stat("errors")
                LOG.exception("[%s] Video decode failed: %s", dl.item.post_id, exc)
                result["files"] = [{
                    "name": media.name,
                    "type": "video",
                    "error": "decode_failed",
                    "message": str(exc),
                }]
                payload = _to_json_gz_bytes(result)
                self._inc_stat("bytes_out", len(payload))
                self.up_q.put(UploadTask(rel_out, payload))
                return

            chunk_limit = 1
            if self.batch_coordinator is not None:
                chunk_limit = max(1, int(getattr(self.batch_coordinator, "max_batch", 1)))

            total_frames = 0
            gpu_frames = 0
            frames_out: List[Dict[str, object]] = []
            chunk: List[FrameBatchEntry] = []
            chunk_indices: List[int] = []
            embed_error: Optional[Exception] = None

            def _submit_chunk() -> bool:
                nonlocal chunk, chunk_indices, embed_error
                if not chunk:
                    return True
                if self.batch_coordinator is None:
                    raise RuntimeError("Batch coordinator unavailable for video embedding")
                to_send = list(chunk)
                indices = list(chunk_indices)
                chunk = []
                chunk_indices = []
                request = self.batch_coordinator.submit(
                    to_send,
                    post_id=dl.item.post_id,
                    media_name=media.name,
                )
                ok, payload = request.get()
                if not ok:
                    embed_error = payload  # type: ignore[assignment]
                    return False
                for idx, embedding in zip(indices, payload):
                    frames_out.append({"index": idx, "embedding": embedding.tolist()})
                return True

            try:
                with contextlib.ExitStack() as stack:
                    iterator = iter(frames_iter)
                    if hasattr(frames_iter, "close"):
                        stack.enter_context(contextlib.closing(frames_iter))
                    for frame in iterator:
                        if isinstance(frame, FrameBatchEntry) and frame.on_cuda:
                            gpu_frames += 1
                        chunk.append(frame)
                        chunk_indices.append(total_frames)
                        total_frames += 1
                        if len(chunk) >= chunk_limit:
                            if not _submit_chunk():
                                break
                    else:
                        if embed_error is None:
                            _submit_chunk()
            except MalformedVideoError as e:
                LOG.warning("[%s] Video unreadable/malformed: %s", dl.item.post_id, e)
                result["files"] = [{
                    "name": media.name,
                    "type": "video",
                    "error": "malformed_video",
                    "message": str(e),
                }]
                payload = _to_json_gz_bytes(result)
                self._inc_stat("bytes_out", len(payload))
                self.up_q.put(UploadTask(rel_out, payload))
                return
            except Exception as exc:
                self._inc_stat("errors")
                LOG.exception("[%s] Video decode failed: %s", dl.item.post_id, exc)
                result["files"] = [{
                    "name": media.name,
                    "type": "video",
                    "error": "decode_failed",
                    "message": str(exc),
                }]
                payload = _to_json_gz_bytes(result)
                self._inc_stat("bytes_out", len(payload))
                self.up_q.put(UploadTask(rel_out, payload))
                return

            self._inc_stat("frames", total_frames)
            if total_frames:
                LOG.debug(
                    "[%s] Decoded frames on CUDA: %d/%d",
                    dl.item.post_id,
                    gpu_frames,
                    total_frames,
                )

            if embed_error is not None:
                self._inc_stat("errors")
                LOG.error("[%s] Frame embedding failed: %s", dl.item.post_id, embed_error)
                result["files"] = [{
                    "name": media.name,
                    "type": "video",
                    "error": "embed_failed",
                    "message": str(embed_error),
                }]
            elif total_frames == 0:
                LOG.debug("[%s] No frames decoded", dl.item.post_id)
                self._inc_stat("errors")
                result["files"] = [{"name": media.name, "type": "video", "frames": []}]
            else:
                frames_out.sort(key=lambda entry: int(entry.get("index", 0)))
                result["files"] = [{"name": media.name, "type": "video", "frames": frames_out}]

        payload = _to_json_gz_bytes(result)
        self._inc_stat("bytes_out", len(payload))
        self.up_q.put(UploadTask(rel_out, payload))

    # ---------- Public ----------
    def run(self, worklist_path: Path) -> None:
        try:
            if self.compute_only:
                self._run_compute_only()
                return

            # If worklist is a bare name, try to pull it from remote into workdir
            if not worklist_path.exists():
                remote_name = worklist_path.name
                local_copy = self.workdir / remote_name
                LOG.info(
                    "Fetching worklist %s from %s:%s",
                    remote_name,
                    self.remote.remote_spec,
                    self.remote.base,
                )
                self.remote.download_worklist(remote_name, local_copy)
                worklist_path = local_copy

            entries = _parse_worklist(worklist_path)

            # Normalize any accidental base duplication in worklist relative paths
            items: List[WorkItem] = [entry.to_work_item(self.remote.base) for entry in entries]

            if not items:
                LOG.info("Nothing to do (empty worklist)")
                return

            LOG.info("Starting pipeline: %d posts", len(items))
            for thread in self.download_threads:
                thread.start()
            for thread in self.upload_threads:
                thread.start()
            for worker in self.proc_workers:
                worker.start()

            for it in items:
                self.down_q.put(it)
            for _ in self.download_threads:
                self.down_q.put(None)

            processed = 0
            total = len(items)
            while processed < total:
                with self._progress_cond:
                    self._progress_cond.wait(timeout=0.5)
                    snapshot = dict(self.stats)

                processed = snapshot.get("processed", processed)
                now = time.monotonic()
                elapsed = now - self.t0
                avg_sec = (elapsed / processed) if processed else 0.0
                ips = (processed / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, total - processed)
                eta_sec = remaining * avg_sec

                LOG.info(
                    "Processed %d/%d | %.2fs/item | %.2f it/s | ETA %s | frames=%d imgs=%d vids=%d out=%s",
                    processed,
                    total,
                    avg_sec,
                    ips,
                    _fmt_duration(eta_sec),
                    snapshot.get("frames", 0),
                    snapshot.get("images", 0),
                    snapshot.get("videos", 0),
                    _fmt_bytes(snapshot.get("bytes_out", 0)),
                )
                LOG.debug(
                    "Queues: down=%d proc=%d up=%d (elapsed=%s total_out=%s errors=%d)",
                    self.down_q.qsize(),
                    self.proc_q.qsize(),
                    self.up_q.qsize(),
                    _fmt_duration(elapsed),
                    _fmt_bytes(snapshot.get("bytes_out", 0)),
                    snapshot.get("errors", 0),
                )

                if processed >= total:
                    break
                if not any(thread.is_alive() for thread in self.download_threads) and self.proc_q.empty():
                    LOG.warning(
                        "Downloader finished with %d/%d posts processed; exiting early.",
                        processed,
                        total,
                    )
                    break

            self.down_q.join()
            for thread in self.download_threads:
                thread.join()
            self.proc_q.join()
            for _ in self.proc_workers:
                self.proc_q.put(None)
            for worker in self.proc_workers:
                worker.join()
            self._close_batch_coordinator()
            self.up_q.join()
            for _ in self.upload_threads:
                self.up_q.put(None)
            for thread in self.upload_threads:
                thread.join()
            LOG.info("Done.")
        finally:
            self._close_batch_coordinator()
            self.embedder.close()


# --------------------------
# CLI
# --------------------------

def _default_workdir() -> Path:
    return Path(os.getenv("WORKDIR", "/tmp/embedder")).resolve()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Bare-bones media embedder")
    p.add_argument("--worklist", type=Path, default=REMOTE_BASE_PATH + "/work_queue.jsonl", help="Path or basename of work_queue.jsonl")
    p.add_argument("--workdir", type=Path, default=_default_workdir())
    default_batch = os.getenv("BATCH_SIZE", "auto")
    default_proc_workers = max(1, (os.cpu_count() or 4) - 1)
    default_max_workers = _int_env(
        "MAX_WORKERS",
        _int_env("PROC_WORKERS", default_proc_workers),
    )
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "google/siglip2-so400m-patch14-384"))
    p.add_argument(
        "--batch-size",
        default=default_batch,
        help="Batch size (int) or 'auto' to probe the largest safe value",
    )
    p.add_argument("--stride-sec", type=float, default=float(os.getenv("STRIDE_SEC", "1.0")))
    p.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers,
        help="Maximum number of concurrent processor workers feeding the GPU",
    )
    p.add_argument(
        "--download-workers",
        type=int,
        default=_int_env(
            "DOWNLOAD_WORKERS", max(2, min(4, default_max_workers))
        ),
        help="Number of downloader threads",
    )
    p.add_argument(
        "--upload-workers",
        type=int,
        default=_int_env(
            "UPLOAD_WORKERS", max(2, min(4, default_max_workers))
        ),
        help="Number of uploader threads",
    )
    p.add_argument(
        "--vad-min-len",
        dest="vad_min_len",
        type=float,
        default=_float_env("VAD_MIN_SPEECH", 0.0),
        help="Minimum VAD span length (ignored by inclusive post-processing)",
    )
    p.add_argument(
        "--vad-merge-within",
        dest="vad_merge_within",
        type=float,
        default=_float_env("VAD_MERGE_WITHIN", 0.60),
        help="Merge VAD spans when separated by ≤ this gap",
    )
    p.add_argument(
        "--vad-pad",
        dest="vad_pad",
        type=float,
        default=_float_env("VAD_PAD", 0.35),
        help="Padding (s) applied around detected spans",
    )
    p.add_argument(
        "--gate-pass-ratio",
        dest="gate_pass_ratio",
        type=float,
        default=_float_env("GATE_PASS_RATIO", 0.60),
        help="Bypass gate when kept/full duration ratio drops below this",
    )
    p.add_argument(
        "--gate-max-spans",
        dest="gate_max_spans",
        type=int,
        default=_int_env("GATE_MAX_SPANS", 500),
        help="Bypass gate when span count exceeds this threshold",
    )
    p.add_argument(
        "--vad-ffmpeg-noise-db",
        dest="vad_ffmpeg_noise_db",
        default=os.getenv("VAD_FFMPEG_NOISE_DB", "-45dB"),
        help="Noise floor passed to ffmpeg silencedetect (more negative = more inclusive)",
    )
    p.add_argument(
        "--vad-ffmpeg-min-sil",
        dest="vad_ffmpeg_min_sil",
        default=os.getenv("VAD_FFMPEG_MIN_SIL", "0.50"),
        help="Minimum silence duration for ffmpeg silencedetect",
    )
    args = p.parse_args(argv)

    if args.max_workers < 1:
        LOG.warning("max_workers=%s invalid; forcing to 1", args.max_workers)
        args.max_workers = 1
    if args.download_workers < 1:
        LOG.warning("download_workers=%s invalid; forcing to 1", args.download_workers)
        args.download_workers = 1
    if args.upload_workers < 1:
        LOG.warning("upload_workers=%s invalid; forcing to 1", args.upload_workers)
        args.upload_workers = 1

    raw_batch = str(args.batch_size).strip()
    auto_batch = raw_batch.lower() == "auto"
    if auto_batch:
        fallback = default_batch if str(default_batch).lower() != "auto" else "32"
        try:
            parsed_batch = int(fallback)
        except ValueError:
            parsed_batch = 32
        args.batch_size = max(1, parsed_batch)
    else:
        try:
            args.batch_size = max(1, int(raw_batch))
        except ValueError as exc:
            raise SystemExit(f"--batch-size must be an integer or 'auto': {exc}")
    args.auto_batch = auto_batch

    LOG.debug("Args: %s", args)
    Pipeline(args).run(args.worklist)


def run_in_notebook(
    *,
    worklist: Optional[Union[str, Path]] = None,
    workdir: Optional[Union[str, Path]] = None,
    model: Optional[str] = None,
    batch_size: Union[int, str, None] = None,
    stride_sec: Optional[float] = None,
    download_workers: Optional[int] = None,
    upload_workers: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> None:
    env_remote = REMOTE_BASE_PATH
    env_model = os.getenv("MODEL_NAME", "google/siglip2-so400m-patch14-384")
    env_batch = os.getenv("BATCH_SIZE", "auto")
    env_stride = float(os.getenv("STRIDE_SEC", "1.0"))

    worklist = Path(worklist) if worklist is not None else Path(env_remote) / "work_queue.jsonl"
    workdir = Path(workdir) if workdir is not None else _default_workdir()
    model = model or env_model
    raw_batch = str(env_batch if batch_size is None else batch_size).strip()

    # Reproduce the "auto" semantics from main()
    auto_batch = raw_batch.lower() == "auto"
    if auto_batch:
        fallback = env_batch if str(env_batch).lower() != "auto" else "32"
        try:
            parsed_batch = int(fallback)
        except (TypeError, ValueError):
            parsed_batch = 32
        batch_size_val = max(1, parsed_batch)
    else:
        try:
            batch_size_val = max(1, int(raw_batch))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--batch-size must be an integer or 'auto': {exc}")

    stride_sec = float(stride_sec) if stride_sec is not None else env_stride

    default_proc_workers = max(1, (os.cpu_count() or 4) - 1)
    max_workers_env = _int_env("MAX_WORKERS", _int_env("PROC_WORKERS", default_proc_workers))
    max_workers_val = max_workers if max_workers is not None else max_workers_env
    try:
        max_workers_val = max(1, int(max_workers_val))
    except (TypeError, ValueError):
        max_workers_val = default_proc_workers

    download_workers_default = _int_env(
        "DOWNLOAD_WORKERS", max(2, min(4, max_workers_val))
    )
    upload_workers_default = _int_env(
        "UPLOAD_WORKERS", max(2, min(4, max_workers_val))
    )

    download_workers_val = (
        download_workers if download_workers is not None else download_workers_default
    )
    upload_workers_val = (
        upload_workers if upload_workers is not None else upload_workers_default
    )
    try:
        download_workers_val = max(1, int(download_workers_val))
    except (TypeError, ValueError):
        download_workers_val = max(2, min(4, max_workers_val))
    try:
        upload_workers_val = max(1, int(upload_workers_val))
    except (TypeError, ValueError):
        upload_workers_val = max(2, min(4, max_workers_val))

    # Build the same shape object `Pipeline` expects
    args = SimpleNamespace(
        worklist=worklist,
        workdir=workdir,
        model=model,
        batch_size=batch_size_val,
        stride_sec=stride_sec,
        auto_batch=auto_batch,
        download_workers=download_workers_val,
        upload_workers=upload_workers_val,
        max_workers=max_workers_val,
    )

    LOG.debug("Args (notebook): %s", args)
    Pipeline(args).run(args.worklist)


if __name__ == "__main__":
    main()
