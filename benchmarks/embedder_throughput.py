"""Benchmark the embedding throughput before/after GPU-native preprocessing."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List

import numpy as np  # type: ignore
import torch
import torch.nn.functional as F

from main import Embedder, FrameBatchEntry


@dataclass
class BenchmarkResult:
    name: str
    wall_time: float
    gpu_time: float
    frames: int

    @property
    def fps(self) -> float:
        return 0.0 if self.wall_time <= 0 else self.frames / self.wall_time

    @property
    def gpu_utilization(self) -> float:
        if self.wall_time <= 0:
            return 0.0
        return min(1.0, self.gpu_time / self.wall_time)


def _run_legacy(embedder: Embedder, frames: List[FrameBatchEntry]) -> BenchmarkResult:
    cfg = embedder._image_config
    if cfg is None:
        raise RuntimeError("Image processor configuration missing; cannot benchmark legacy path")

    start_wall = time.perf_counter()
    total_gpu = 0.0
    for i in range(0, len(frames), embedder.batch):
        chunk = frames[i : i + embedder.batch]
        cpu_preprocessed = []
        for entry in chunk:
            payload = entry.data
            if torch.is_tensor(payload):
                tensor = payload.detach().cpu()
            else:
                tensor = torch.tensor(payload)
            if tensor.ndim != 3:
                raise ValueError(f"Unexpected ndim for legacy payload: {tensor.ndim}")
            if tensor.shape[0] == 3:
                chw = tensor
            elif tensor.shape[-1] == 3:
                chw = tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported legacy tensor shape: {tuple(tensor.shape)}")

            chw = chw.float()
            if cfg.get("do_rescale"):
                chw = chw * cfg["rescale_factor"]
            else:
                chw = chw / 255.0

            if cfg.get("do_resize"):
                h, w = chw.shape[-2:]
                target_h, target_w = embedder._compute_resize_dims(h, w, cfg)
                if (target_h, target_w) != (h, w):
                    mode = cfg.get("resize_mode", "bilinear")
                    align = cfg.get("resize_align_corners")
                    kwargs = {}
                    if mode in {"bilinear", "bicubic"}:
                        kwargs["align_corners"] = bool(align is True)
                    chw = (
                        F.interpolate(
                            chw.unsqueeze(0),
                            size=(target_h, target_w),
                            mode=mode,
                            **kwargs,
                        )
                        .squeeze(0)
                        .contiguous()
                    )

            if cfg.get("do_center_crop") and cfg.get("crop_hw"):
                chw = embedder._center_crop(chw, cfg["crop_hw"]).contiguous()

            chw = chw.to(dtype=embedder.model_dtype)
            if cfg.get("do_normalize"):
                mean = cfg["mean"].to(device=chw.device, dtype=chw.dtype).view(-1, 1, 1)
                std = cfg["std"].to(device=chw.device, dtype=chw.dtype).view(-1, 1, 1)
                chw = (chw - mean) / std

            cpu_preprocessed.append(chw)

        batch = torch.stack(cpu_preprocessed, dim=0).to(embedder.device)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        feats = embedder.model.get_image_features(pixel_values=batch)
        end_evt.record()
        torch.cuda.synchronize()
        total_gpu += start_evt.elapsed_time(end_evt) / 1000.0
        _ = F.normalize(feats, p=2, dim=-1)
    wall = time.perf_counter() - start_wall
    return BenchmarkResult("cpu-preprocess", wall, total_gpu, len(frames))


def _run_gpu(embedder: Embedder, frames: List[FrameBatchEntry]) -> BenchmarkResult:
    start_wall = time.perf_counter()
    total_gpu = 0.0
    for i in range(0, len(frames), embedder.batch):
        chunk = frames[i : i + embedder.batch]
        pixel_values = embedder._preprocess_batch(chunk)
        feats, gpu_time = embedder._forward_features(pixel_values, measure_gpu_time=True)
        if gpu_time is None:
            raise RuntimeError("Expected GPU timing information")
        total_gpu += gpu_time
        _ = F.normalize(feats, p=2, dim=-1)
    wall = time.perf_counter() - start_wall
    return BenchmarkResult("gpu-preprocess", wall, total_gpu, len(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip-base-patch16-256")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError("CUDA device is required for the benchmark")

    embedder = Embedder(
        args.model,
        device="cuda",
        fp16=args.fp16,
        batch_size=args.batch_size,
        auto_batch=False,
    )

    torch.cuda.synchronize()

    frames = [
        FrameBatchEntry(
            data=torch.randint(
                0,
                256,
                (3, args.height, args.width),
                dtype=torch.uint8,
                device="cuda",
            ),
            on_cuda=True,
        )
        for _ in range(args.frames)
    ]

    # Warmup pass
    embedder.embed_images(frames[: embedder.batch])
    torch.cuda.synchronize()

    legacy = _run_legacy(embedder, frames)
    gpu_native = _run_gpu(embedder, frames)

    print("Legacy CPU preprocessing:")
    print(f"  Wall time: {legacy.wall_time:.3f}s")
    print(f"  GPU time:  {legacy.gpu_time:.3f}s")
    print(f"  Throughput: {legacy.fps:.2f} frames/s")
    print(f"  GPU utilization (approx.): {legacy.gpu_utilization:.2%}")

    print("\nGPU-native preprocessing:")
    print(f"  Wall time: {gpu_native.wall_time:.3f}s")
    print(f"  GPU time:  {gpu_native.gpu_time:.3f}s")
    print(f"  Throughput: {gpu_native.fps:.2f} frames/s")
    print(f"  GPU utilization (approx.): {gpu_native.gpu_utilization:.2%}")

    speedup = legacy.wall_time / gpu_native.wall_time if gpu_native.wall_time > 0 else float("inf")
    print(f"\nThroughput speedup: {speedup:.2f}x")
if __name__ == "__main__":
    main()

