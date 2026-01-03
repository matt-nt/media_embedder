"""Video decoding utilities using NVIDIA DALI"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack

import nvidia.dali.fn as dali_fn
import nvidia.dali.types as dali_types
from nvidia.dali.pipeline import Pipeline

LOG = logging.getLogger(__name__)


@dataclass
class DecodedFrame:
    """Single decoded video frame."""
    index: int
    timestamp: float
    payload: Union[np.ndarray, torch.Tensor]  # CHW(uint8) for torch, HWC(uint8) for numpy
    device: str  # "cpu" or "gpu"


class _VideoPipeline(Pipeline):
    """Thin DALI Pipeline wrapper that yields single, optionally-resized video frames with timestamps."""

    def __init__(
        self,
        *,
        filenames: list[str],
        labels: list[int],
        step: int,
        device: Literal["cpu", "gpu"],
        resize_shorter: Optional[int] = None,  # if set, resize shorter side to this
        resize_to: Optional[Tuple[int, int]] = None,  # (H, W) exact size if provided
        batch_size: int = 1,
        num_threads: int = 2,
        device_id: int = 0,
        seed: int = 42,
        prefetch_queue_depth: int = 2,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_threads=max(1, int(num_threads)),
            device_id=device_id if device == "gpu" else 0,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        self._filenames = filenames
        self._labels = labels
        self._step = max(1, int(step))
        self._device = device
        self._resize_shorter = resize_shorter
        self._resize_to = resize_to

    def define_graph(self):
        # Reader returns: video (N,T,H,W,C), timestamps (N,T), labels (N)
        # We use sequence_length=1 so T==1.
        video, timestamps, _ = dali_fn.readers.video(
            enable_timestamps=True,
            device=self._device,
            filenames=self._filenames,
            labels=self._labels,
            sequence_length=1,
            random_shuffle=False,
            initial_fill=16,
            enable_frame_num=False,
            file_list_include_preceding_frame=False,
            step=self._step,
            dtype=dali_types.UINT8,
            name="Reader",
        )

        # Optional resize inside DALI (keeps data on the same device)
        if self._resize_to is not None:
            h, w = self._resize_to
            video = dali_fn.resize(video, size=(h, w), device=self._device)
        elif self._resize_shorter is not None:
            video = dali_fn.resize(video, resize_shorter=int(self._resize_shorter), device=self._device)

        return video, timestamps  # pipeline outputs (frames, timestamps)


class DaliVideoDecoder:

    def __init__(
        self,
        path: str,
        *,
        stride_sec: float = 1.0,
        use_gpu: bool = True,
        num_threads: int = 2,
        device_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        max_frames_per_video: Optional[int] = 120,
        max_duration_sec: Optional[float] = 300.0,
        resize_shorter: Optional[int] = 384,
        resize_to: Optional[Tuple[int, int]] = None,
        prefetch_queue_depth: int = 2,
        seed: int = 42,
    ) -> None:
        self.path = path
        self.stride_sec = float(stride_sec)
        self.use_gpu = bool(use_gpu)
        self.num_threads = max(1, int(num_threads))
        self.device_id = int(device_id if device_id is not None else 0)
        self.log = logger or LOG

        self.max_frames_per_video = max_frames_per_video
        self.max_duration_sec = max_duration_sec
        self.resize_shorter = resize_shorter
        self.resize_to = resize_to
        self.prefetch_queue_depth = prefetch_queue_depth
        self.seed = seed

        self._pipe: Optional[_VideoPipeline] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._step: int = 1

        self._initialize_pipeline()

    # --------------------------
    # Public properties
    # --------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def step(self) -> int:
        return self._step

    # --------------------------
    # Context management
    # --------------------------
    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.release_outputs()
            except Exception:
                pass
            self._pipe = None

    def __enter__(self) -> "DaliVideoDecoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------------------------
    # Iteration
    # --------------------------
    def frames(self, max_samples: Optional[int] = None) -> Iterator[DecodedFrame]:
        """
        Yields DecodedFrame with:
          - index: approximate frame index derived from timestamp and stream fps
          - timestamp: seconds (float) from DALI timestamps
          - payload: torch.Tensor (CHW, uint8, GPU) when CUDA+use_gpu; else numpy.ndarray (HWC, uint8)
          - device: "gpu" or "cpu"
        """
        if self._pipe is None:
            return

        produced = 0
        effective_cap = max_samples
        fps = getattr(self, "_fps", 0.0)
        step = max(1, getattr(self, "_step", 1))
        max_duration = getattr(self, "max_duration_sec", None)

        if max_duration is not None and fps > 0:
            samples_per_sec = fps / step
            by_duration = int(math.ceil(max_duration * samples_per_sec))
            effective_cap = by_duration if effective_cap is None else min(effective_cap, by_duration)

        max_frames = getattr(self, "max_frames_per_video", None)
        if max_frames is not None:
            effective_cap = max_frames if effective_cap is None else min(effective_cap, max_frames)

        while True:
            if effective_cap is not None and produced >= effective_cap:
                break
            try:
                frames_out, timestamps_out = self._pipe.run()
            except StopIteration:
                self._pipe.reset()
                break
            except RuntimeError as exc:
                self.log.debug("[DALI] pipeline run failed for %s: %s", self.path, exc)
                raise

            try:
                ts = self._extract_index(timestamps_out)
                frame_idx = int(round(ts * fps)) if fps > 0 else produced
                payload, device = self._extract_payload(frames_out)
                yield DecodedFrame(index=frame_idx, timestamp=ts, payload=payload, device=device)
                produced += 1
            finally:
                try:
                    self._pipe.release_outputs()
                except Exception:
                    pass

    # --------------------------
    # Internal helpers
    # --------------------------
    def _initialize_pipeline(self) -> None:
        # Build with step=1 to read metadata
        first_step = 1
        self._pipe = self._build_pipeline(first_step)
        self._pipe.build()
        meta = self._pipe.reader_meta("Reader")

        fps = float(
            meta.get("avg_frame_rate")
            or meta.get("average_fps")
            or meta.get("frame_rate")
            or 0.0
        )
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(meta.get("total_frame_count") or meta.get("num_frames") or 0)

        self._fps = fps
        self._total_frames = total_frames

        # Compute stride in frames
        stride_frames = max(1, int(round(self._fps * max(self.stride_sec, 0.0))))

        # Rebuild with the real step so we don't decode unused frames
        if stride_frames != first_step:
            self._pipe = self._build_pipeline(stride_frames)
            self._pipe.build()
            self._step = stride_frames
        else:
            self._step = first_step

    def _build_pipeline(self, step: int) -> _VideoPipeline:
        device = "gpu" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        if self.use_gpu and device != "gpu":
            raise RuntimeError("CUDA is required for GPU video decoding but is unavailable")

        device_id = self.device_id if device == "gpu" else 0

        filenames = [self.path]
        labels = [0] * len(filenames)

        return _VideoPipeline(
            filenames=filenames,
            labels=labels,
            step=step,
            device=device,
            resize_shorter=self.resize_shorter,
            resize_to=self.resize_to,
            batch_size=1,
            num_threads=self.num_threads,
            device_id=device_id,
            seed=self.seed,
            prefetch_queue_depth=self.prefetch_queue_depth,
        )

    @staticmethod
    def _extract_timestamp(timestamps) -> float:
        """Extract a scalar seconds timestamp from DALI timestamps output."""
        tl_cpu = timestamps.as_cpu()
        if hasattr(tl_cpu, "as_array"):
            arr = tl_cpu.as_array()
        elif hasattr(tl_cpu, "numpy"):
            arr = tl_cpu.numpy()
        else:
            arr = np.array(tl_cpu)

        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(arr[0])

    def _extract_index(self, timestamps) -> float:
        """Return the fractional timestamp for the current frame."""
        return self._extract_timestamp(timestamps)

    def _extract_payload(self, frames) -> Tuple[Union[torch.Tensor, np.ndarray], str]:
        """
        Returns:
            - torch.Tensor [3,H,W] on GPU (uint8) when use_gpu & CUDA available
            - else numpy.ndarray [H,W,3] on CPU (uint8)
        """
        t = frames.as_tensor()

        # Preferred: stay on GPU
        if self.use_gpu and torch.cuda.is_available():
            x = from_dlpack(t)  # typically [N,H,W,C] uint8 on GPU
            while x.ndim > 3:
                x = x[0]
            if x.ndim != 3:
                raise ValueError(f"Unexpected GPU frame tensor shape: {x.shape}")
            if x.shape[-1] == 3:  # HWC -> CHW
                x = x.permute(2, 0, 1)
            elif x.shape[0] != 3:
                raise ValueError(f"Unexpected channel layout for GPU tensor: {x.shape}")
            return x.contiguous(), "gpu"

        # CPU fallback
        t_cpu = t.as_cpu()
        if hasattr(t_cpu, "as_array"):
            np_frame = t_cpu.as_array()
        elif hasattr(t_cpu, "numpy"):
            np_frame = t_cpu.numpy()
        else:
            np_frame = np.array(t_cpu)

        while np_frame.ndim > 3:
            np_frame = np_frame[0]

        if np_frame.ndim != 3:
            raise ValueError(f"Unexpected CPU frame array shape: {np_frame.shape}")

        if np_frame.shape[0] == 3 and np_frame.shape[-1] != 3:
            np_frame = np.transpose(np_frame, (1, 2, 0))

        return np_frame, "cpu"
