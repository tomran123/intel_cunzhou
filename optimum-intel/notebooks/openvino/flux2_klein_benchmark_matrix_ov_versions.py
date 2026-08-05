"""FLUX.2-klein OpenVINO benchmark matrix ï¿½ CFG disabled, RAM pipeline-start baseline method.

Tests 4 configurations by default:
    FP16  @ 512x512   @ 4 steps
    FP16  @ 1024x1024 @ 4 steps
    INT4  @ 512x512   @ 4 steps
    INT4  @ 1024x1024 @ 4 steps

This script extends the working lifecycle demonstrated in ``flux2_klein_inference.py``
(LOAD -> INFERENCE -> RELEASE per component, one heavy model compiled at a time)
into a full, repeatable benchmark matrix with two important differences from the
older ``flux2_klein_benchmark.py`` in this same folder:

1. CFG is always OFF. ``guidance_scale`` is fixed at 1.0 and ``negative_prompt_embeds``
   is always ``None`` ï¿½ only the positive prompt is ever encoded, and every
   denoising step runs the transformer exactly once (not twice for +/- prompts).
   (``OVFlux2KleinPipeline.__call__`` also forces ``guidance_scale`` back to 1.0
   internally since FLUX.2-klein is a distilled, guidance-free model.)

2. RAM is measured with a pipeline-start baseline. For each configuration, we
   take one RSS snapshot immediately after pipeline initialization; this single
   value is the baseline for all components in that configuration. RAM After
   Load / Peak RAM are then reported as:
       RAM After Load = RSS right after compile()   - pipeline_start_rss
       Peak RAM       = max RSS during LOAD+INFER   - pipeline_start_rss
   This follows the "peak - pipeline start point" method.

Component lifecycle (one heavy model compiled at a time):
    text_encoder -> transformer -> vae_decoder
    each:  LOAD (compile()) -> INFERENCE -> RELEASE (request=None, model=None, gc)

The transformer's INFERENCE end time is captured via ``callback_on_step_end`` at
the last denoising step (NOT by timing the whole ``pipe.__call__``). Right after
that, the transformer is released and the VAE decoder is compiled *inside* the
callback so only one heavy model is ever resident. VAE INFERENCE time is
``pipe_end - vae_ready`` (post-loop decode only); saving the image is excluded.

Nothing about the target machine (CPU/GPU model, driver, RAM size) is hard-coded;
``collect_environment_metadata()`` reads it all at run time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# psutil is used for RSS memory sampling. It is optional: if it is not
# installed, every RAM metric is honestly reported as N/A instead of crashing
# or fabricating a value.
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_PSUTIL = False



def get_openvino_module() -> Any:
    """Import OpenVINO lazily so launcher mode can change package versions
    before any OpenVINO DLL is loaded into the current process.
    """
    return importlib.import_module("openvino")

# --------------------------------------------------------------------------- #
# Fixed, non-configurable invariants (see module docstring, item 1)
# --------------------------------------------------------------------------- #
# guidance_scale / output_type are intentionally NOT exposed as CLI flags: this
# benchmark must never accidentally run with CFG enabled or a non-PIL output
# that would break the image-saving code path.
GUIDANCE_SCALE: float = 1.0
OUTPUT_TYPE: str = "pil"

COMPONENT_ORDER: List[str] = ["text_encoder", "transformer", "vae_decoder"]
COMPONENT_DISPLAY: Dict[str, str] = {
    "text_encoder": "Text Encoder",
    "transformer": "Transformer",
    "vae_decoder": "Small VAE Decoder",
}
RESULT_HEADERS: List[str] = ["Resolution", "Model", "Load Time", "Infer Time", "RAM After Load", "Peak RAM"]
OV_MATRIX_HEADERS: List[str] = ["OV Version", *RESULT_HEADERS]
MEMORY_DETAIL_HEADERS: List[str] = [
    "Resolution",
    "Component",
    "Precision",
    "Baseline RAM",
    "After-Load RAM",
    "Peak RAM (absolute)",
    "RAM After Load Delta",
    "Peak RAM Delta",
    "Trough After Release",
]

BYTES_PER_GB = 1024**3
_LOW_BIT_TYPES = {"u4", "i4", "nf4"}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class RamSample:
    """One background RSS sample tagged with the component/phase active then."""

    perf_ts: float
    wall_ts: str
    component: str
    phase: str
    rss: int


@dataclass
class ComponentResult:
    """Benchmark outcome for a single (resolution, component) pair.

    ``baseline_rss`` is the pipeline-start RSS for this configuration (shared
    by all components in that configuration). ``ram_after_load_delta`` and
    ``peak_ram_delta`` are always relative to this single start-point.
    """

    resolution: str
    component: str
    tier: str
    model_label: str
    precision: str
    load_time_s: Optional[float]
    infer_time_s: Optional[float]
    baseline_rss: Optional[int]
    after_load_rss: Optional[int]
    peak_rss: Optional[int]
    trough_after_release: Optional[int]
    error: Optional[str] = None

    @property
    def ram_after_load_delta(self) -> Optional[int]:
        if self.after_load_rss is None or self.baseline_rss is None:
            return None
        return self.after_load_rss - self.baseline_rss

    @property
    def peak_ram_delta(self) -> Optional[int]:
        if self.peak_rss is None or self.baseline_rss is None:
            return None
        return self.peak_rss - self.baseline_rss


@dataclass
class BenchmarkConfig:
    """One (precision, resolution) cell of the benchmark matrix."""

    precision: str  # "fp16" | "int4"
    model_dir: Path
    width: int
    height: int


# --------------------------------------------------------------------------- #
# RAM sampler
# --------------------------------------------------------------------------- #
class RamSampler:
    """Background thread sampling this process' RSS, tagging every sample with
    the (component, phase) that was active at sampling time.

    This is what makes the "trough-to-peak" method possible: instead of one
    whole-process peak, we can ask "what was the min/max RSS while component X
    was in phase Y", which is exactly what ``wait_for_ram_trough`` and the
    per-component Peak RAM computation need. If ``psutil`` is unavailable the
    sampler becomes a no-op and every query returns ``None``.
    """

    def __init__(self, interval_s: float = 0.03) -> None:
        self.interval_s = interval_s
        self._enabled = _HAS_PSUTIL
        self._process = psutil.Process(os.getpid()) if self._enabled else None
        self._samples: List[RamSample] = []
        self._lock = threading.Lock()
        self._component = "init"
        self._phase = "stabilize"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_phase(self, component: str, phase: str) -> None:
        """Tag subsequent samples with ``component``/``phase`` (e.g. "transformer"/"inference")."""
        with self._lock:
            self._component = component
            self._phase = phase

    def _record(self) -> Optional[int]:
        if not self._enabled:
            return None
        try:
            rss = self._process.memory_info().rss
        except Exception:
            return None
        with self._lock:
            self._samples.append(
                RamSample(
                    perf_ts=time.perf_counter(),
                    wall_ts=datetime.now().isoformat(timespec="milliseconds"),
                    component=self._component,
                    phase=self._phase,
                    rss=rss,
                )
            )
        return rss

    def current_rss(self) -> Optional[int]:
        """Take one immediate sample now (tagged with the currently active phase)."""
        return self._record()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._record()
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop_event.clear()
        self._record()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def samples_between(
        self, start_ts: float, end_ts: float, component: Optional[str] = None
    ) -> List[RamSample]:
        with self._lock:
            data = list(self._samples)
        return [
            s
            for s in data
            if start_ts <= s.perf_ts <= end_ts and (component is None or s.component == component)
        ]

    def peak_between(self, start_ts: float, end_ts: float, component: Optional[str] = None) -> Optional[int]:
        samples = self.samples_between(start_ts, end_ts, component)
        if not samples:
            return None
        return max(s.rss for s in samples)

    def min_between(self, start_ts: float, end_ts: float, component: Optional[str] = None) -> Optional[int]:
        samples = self.samples_between(start_ts, end_ts, component)
        if not samples:
            return None
        return min(s.rss for s in samples)

    def all_samples(self) -> List[RamSample]:
        with self._lock:
            return list(self._samples)


def wait_for_ram_trough(
    sampler: RamSampler,
    component: str,
    duration_s: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """Find the local RAM trough that becomes ``component``'s baseline.

    Call this AFTER the previous component has been released. It forces a GC
    pass, then samples RSS for ``duration_s`` seconds (tagged as
    ``(component, "stabilize")``) and returns the MINIMUM RSS observed in that
    window ï¿½ i.e. "current peak minus previous trough", never the process'
    initial RSS. Returns ``None`` if RAM cannot be measured (psutil missing).
    """
    gc.collect()
    if not sampler.enabled:
        return None
    sampler.set_phase(component, "stabilize")
    start_ts = time.perf_counter()
    time.sleep(max(duration_s, 0.0))
    end_ts = time.perf_counter()
    trough = sampler.min_between(start_ts, end_ts, component=component)
    if trough is None:
        trough = sampler.current_rss()
    if logger is not None:
        logger.debug("[STABILIZE] %-13s trough=%s over %.2fs", component, trough, duration_s)
    return trough


# --------------------------------------------------------------------------- #
# LOAD / RELEASE
# --------------------------------------------------------------------------- #
def compile_component(
    pipe: Any,
    component_name: str,
    sampler: RamSampler,
    logger: logging.Logger,
    cache_root: Optional[Path] = None,
) -> Tuple[float, Optional[int]]:
    """LOAD phase: ``component.compile()``, timing ONLY the compile call.

    Returns ``(load_time_s, after_load_rss)`` where ``after_load_rss`` is the
    absolute process RSS sampled immediately after compilation finishes.
    """
    component = getattr(pipe, component_name, None)
    if component is None:
        raise RuntimeError(f"Pipeline has no component named '{component_name}'")

    # Keep OpenVINO compile blobs in per-component subfolders so each
    # submodel cache is isolated under one root directory.
    if cache_root is not None:
        cache_dir = cache_root / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(pipe, "ov_config") or pipe.ov_config is None:
            pipe.ov_config = {}
        pipe.ov_config["CACHE_DIR"] = str(cache_dir)

    sampler.set_phase(component_name, "load")
    t0 = time.perf_counter()
    component.compile()
    load_time_s = time.perf_counter() - t0
    after_load_rss = sampler.current_rss()
    logger.info("[LOAD]    %-13s %8.3fs", component_name, load_time_s)
    return load_time_s, after_load_rss


def release_component(
    pipe: Any,
    component_name: str,
    sampler: RamSampler,
    logger: logging.Logger,
) -> None:
    """RELEASE (deep) phase: drop the compiled request (GPU/device memory) and
    the ``ov.Model`` itself (host RAM), then force a GC pass.

    Only one heavy model is ever compiled at a time: every component is fully
    released before the next one is compiled.
    """
    component = getattr(pipe, component_name, None)
    if component is None:
        return
    sampler.set_phase(component_name, "release")
    component.request = None
    component.model = None
    gc.collect()
    sampler.current_rss()
    logger.info("[RELEASE] %-13s (GPU + host RAM freed)", component_name)


def detect_component_precision(component: Any, component_name: str, fallback: str) -> str:
    """Best-effort detection of a component's real weight precision.

    Scans the component's ``ov.Model`` constants for low-bit (4-bit) element
    types (u4/i4/nf4). Any such constant means the component was quantized to
    INT4; otherwise it is reported as FP16.

    The VAE decoder is always reported as FP16 regardless of tier: inspection
    of both exported tiers confirmed it never carries dominant low-bit weight
    compression (only a handful of incidental low-bit constants unrelated to
    its actual compute precision), so it is never mislabeled by scanning it
    the same way as the other two components would.
    """
    if component_name == "vae_decoder":
        return "FP16"
    model = getattr(component, "model", None)
    if model is None:
        return fallback
    try:
        for op in model.get_ordered_ops():
            if op.get_type_name() != "Constant":
                continue
            if op.get_element_type().get_type_name() in _LOW_BIT_TYPES:
                return "INT4"
    except Exception:
        return fallback
    return "FP16"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.3f}s"


def fmt_gb(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "N/A"
    sign = "-" if num_bytes < 0 else ""
    return f"{sign}{abs(num_bytes) / BYTES_PER_GB:.2f}GB"


# --------------------------------------------------------------------------- #
# Environment metadata (nothing hard-coded ï¿½ all read at run time)
# --------------------------------------------------------------------------- #
def _ov_property(core: "ov.Core", device: str, key: str) -> Optional[str]:
    try:
        return str(core.get_property(device, key))
    except Exception:
        return None


def _windows_gpu_driver_version() -> Optional[str]:
    """Best-effort GPU driver version on Windows via WMI. Returns None on any
    failure (including on non-Windows platforms) so the caller can honestly
    fall back to N/A instead of guessing.
    """
    if platform.system() != "Windows":
        return None
    ps_cmd = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -match 'Intel|Arc|NVIDIA|AMD' } | "
        "Select-Object -First 1 -ExpandProperty DriverVersion"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=15,
        )
        version = result.stdout.strip()
        return version or None
    except Exception:
        return None


def _optimum_intel_version() -> str:
    try:
        from optimum.intel.version import __version__

        return __version__
    except Exception:
        try:
            from importlib.metadata import version as pkg_version

            return pkg_version("optimum-intel")
        except Exception:
            return "N/A"


def collect_environment_metadata(device: str) -> Dict[str, Any]:
    """Collect real environment / hardware information at run time.

    Nothing here is hard-coded: CPU/GPU names, driver version, total RAM, and
    software versions are all queried from OpenVINO / the OS / installed
    packages, so the report always reflects the machine it actually ran on.
    """
    ov = get_openvino_module()
    core = ov.Core()
    base_device = device.split(":")[0].upper() if device else "CPU"

    cpu_name = _ov_property(core, "CPU", "FULL_DEVICE_NAME") or platform.processor() or "N/A"
    gpu_name = _ov_property(core, "GPU", "FULL_DEVICE_NAME") or "N/A"

    gpu_driver = None
    for key in ("GPU_DRIVER_VERSION", "DRIVER_VERSION"):
        value = _ov_property(core, "GPU", key)
        if value:
            gpu_driver = value
            break
    if gpu_driver is None:
        gpu_driver = _windows_gpu_driver_version()

    try:
        ov_version = ov.get_version()
    except Exception:
        ov_version = getattr(ov, "__version__", "N/A")

    total_memory_bytes = psutil.virtual_memory().total if _HAS_PSUTIL else None

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "openvino_version": ov_version,
        "optimum_intel_version": _optimum_intel_version(),
        "cpu_name": cpu_name,
        "gpu_name": gpu_name,
        "gpu_driver_version": gpu_driver or "N/A",
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gb": round(total_memory_bytes / (1024**3), 1) if total_memory_bytes else "N/A",
        "device": device,
        "selected_device_name": _ov_property(core, base_device, "FULL_DEVICE_NAME") or base_device,
        "available_devices": core.available_devices,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }


def collect_model_metadata(model_dir: Path) -> Dict[str, Any]:
    """Read whatever model provenance is actually present on disk (repo id,
    diffusers version, export commit/revision if recoverable, optimum version
    used at export time). Missing fields stay ``None`` rather than being
    guessed or hard-coded.
    """
    metadata: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "repo_id": None,
        "diffusers_version": None,
        "revision": None,
        "optimum_version_at_export": None,
    }

    index_path = model_dir / "model_index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            metadata["repo_id"] = data.get("_name_or_path")
            metadata["diffusers_version"] = data.get("_diffusers_version")
        except Exception:
            pass

    # Recover the export commit hash, if any, from a component's cached
    # snapshot path (e.g. ".../snapshots/<hash>/transformer").
    for component_name in ("transformer", "text_encoder", "vae_decoder"):
        cfg_path = model_dir / component_name / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name_or_path = str(cfg.get("_name_or_path", ""))
        match = re.search(r"snapshots[\\/]([0-9a-fA-F]{7,40})", name_or_path)
        if match:
            metadata["revision"] = match.group(1)
            break

    ov_config_path = model_dir / "openvino_config.json"
    if ov_config_path.exists():
        try:
            ov_cfg = json.loads(ov_config_path.read_text(encoding="utf-8"))
            metadata["optimum_version_at_export"] = ov_cfg.get("optimum_version")
        except Exception:
            pass

    return metadata


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def parse_resolutions(spec: str) -> List[Tuple[int, int]]:
    """Parse ``"512,1024"`` or ``"512x512,1024x768"`` into ``[(w, h), ...]``."""
    resolutions: List[Tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" in token:
            w_str, h_str = token.split("x")
            resolutions.append((int(w_str), int(h_str)))
        else:
            size = int(token)
            resolutions.append((size, size))
    return resolutions


# --------------------------------------------------------------------------- #
# Core: one (precision, resolution) cell of the matrix
# --------------------------------------------------------------------------- #
def benchmark_single_configuration(
    config: BenchmarkConfig,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[List[ComponentResult], Optional[Path], Optional[str]]:
    """Run one (precision, resolution) configuration with warmup + measured runs.

    Lifecycle logic is unchanged for each pass: text_encoder -> transformer ->
    vae_decoder, one heavy component compiled at a time. Warmup passes are
    executed first and discarded; measured passes are then averaged component-
    wise into one output row set for this configuration.
    """
    resolution_label = f"{config.width}x{config.height}"
    config_tag = f"{config.precision}_{resolution_label}_{args.steps}steps_no_cfg"
    logger.info("=" * 78)
    logger.info("Configuration: %s | model_dir=%s", config_tag, config.model_dir)
    logger.info("=" * 78)
    cache_root = Path(args.cache_dir)

    def _single_pass(pass_tag: str, save_image: bool) -> Tuple[List[ComponentResult], Optional[Path], Optional[str]]:
        sampler = RamSampler(interval_s=args.ram_interval_ms / 1000.0)
        sampler.set_phase("pipeline_init", "load")
        sampler.start()

        pass_results: List[ComponentResult] = []
        pass_image_path: Optional[Path] = None
        pass_error: Optional[str] = None
        pipe: Any = None

        try:
            from optimum.intel import OVDiffusionPipeline

            # ---- pipeline init: read IR into host RAM, compile NOTHING yet ----
            t0 = time.perf_counter()
            pipe = OVDiffusionPipeline.from_pretrained(
                str(config.model_dir), compile=False, cache_dir=args.cache_dir
            )
            pipe.to(args.device)
            pipe.set_progress_bar_config(disable=True)
            init_time = time.perf_counter() - t0
            logger.info("[%s][INIT]    pipeline_init  %8.3fs", pass_tag, init_time)

            precision_labels = {
                "text_encoder": detect_component_precision(
                    pipe.text_encoder, "text_encoder", config.precision.upper()
                ),
                "transformer": detect_component_precision(
                    pipe.transformer, "transformer", config.precision.upper()
                ),
                "vae_decoder": detect_component_precision(pipe.vae_decoder, "vae_decoder", "FP16"),
            }

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            # RAM baseline for this whole pass: pipeline start point.
            pipeline_start_rss = sampler.current_rss()

            # =========================== text_encoder ===========================
            te_baseline = pipeline_start_rss
            te_window_start = time.perf_counter()
            te_load_time, te_after_load = compile_component(
                pipe,
                "text_encoder",
                sampler,
                logger,
                cache_root=cache_root,
            )

            sampler.set_phase("text_encoder", "inference")
            t0 = time.perf_counter()
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=args.prompt,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )
            te_infer_time = time.perf_counter() - t0
            te_window_end = time.perf_counter()
            te_peak = sampler.peak_between(te_window_start, te_window_end, component="text_encoder")
            logger.info(
                "[%s][INFER]   text_encoder   %8.3fs  embeds=%s",
                pass_tag,
                te_infer_time,
                tuple(prompt_embeds.shape),
            )

            release_component(pipe, "text_encoder", sampler, logger)
            te_release_rss = sampler.current_rss()

            pass_results.append(
                ComponentResult(
                    resolution=resolution_label,
                    component="text_encoder",
                    tier=config.precision,
                    model_label=f"{COMPONENT_DISPLAY['text_encoder']}-{precision_labels['text_encoder']}",
                    precision=precision_labels["text_encoder"],
                    load_time_s=te_load_time,
                    infer_time_s=te_infer_time,
                    baseline_rss=te_baseline,
                    after_load_rss=te_after_load,
                    peak_rss=te_peak,
                    trough_after_release=te_release_rss,
                )
            )

            # =========================== transformer =============================
            tr_baseline = pipeline_start_rss

            tr_window_start = time.perf_counter()
            tr_load_time, tr_after_load = compile_component(
                pipe,
                "transformer",
                sampler,
                logger,
                cache_root=cache_root,
            )

            callback_state: Dict[str, Any] = {}
            last_step = args.steps - 1

            def on_step_end(
                pipeline: Any, step: int, timestep: Any, callback_kwargs: Dict[str, Any]
            ) -> Dict[str, Any]:
                if step == last_step:
                    callback_state["denoise_end"] = time.perf_counter()
                    logger.info(
                        "[%s][INFER]   transformer    %8.3fs",
                        pass_tag,
                        callback_state["denoise_end"] - callback_state["denoise_start"],
                    )
                    # transformer RELEASE (deep) ï¿½ must happen before vae_decoder LOAD
                    # so only one heavy model is ever compiled at a time.
                    release_component(pipeline, "transformer", sampler, logger)
                    callback_state["transformer_release_rss"] = sampler.current_rss()
                    callback_state["vae_baseline"] = pipeline_start_rss
                    callback_state["vae_window_start"] = time.perf_counter()
                    vae_load_time, vae_after_load = compile_component(
                        pipeline,
                        "vae_decoder",
                        sampler,
                        logger,
                        cache_root=cache_root,
                    )
                    callback_state["vae_load_time"] = vae_load_time
                    callback_state["vae_after_load"] = vae_after_load
                    sampler.set_phase("vae_decoder", "inference")
                    callback_state["vae_ready"] = time.perf_counter()
                return callback_kwargs

            sampler.set_phase("transformer", "inference")
            callback_state["denoise_start"] = time.perf_counter()
            out = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=None,
                guidance_scale=GUIDANCE_SCALE,
                height=config.height,
                width=config.width,
                num_inference_steps=args.steps,
                output_type=OUTPUT_TYPE,
                callback_on_step_end=on_step_end,
            )
            pipe_end = time.perf_counter()

            if "denoise_end" not in callback_state:
                raise RuntimeError(
                    "callback_on_step_end never fired for the last denoising step; "
                    "transformer/vae timing could not be captured"
                )

            tr_infer_time = callback_state["denoise_end"] - callback_state["denoise_start"]
            tr_peak = sampler.peak_between(tr_window_start, callback_state["denoise_end"], component="transformer")

            pass_results.append(
                ComponentResult(
                    resolution=resolution_label,
                    component="transformer",
                    tier=config.precision,
                    model_label=f"{COMPONENT_DISPLAY['transformer']}-{precision_labels['transformer']}",
                    precision=precision_labels["transformer"],
                    load_time_s=tr_load_time,
                    infer_time_s=tr_infer_time,
                    baseline_rss=tr_baseline,
                    after_load_rss=tr_after_load,
                    peak_rss=tr_peak,
                    trough_after_release=callback_state["vae_baseline"],
                )
            )

            # =========================== vae_decoder =============================
            vae_infer_time = pipe_end - callback_state["vae_ready"]
            vae_peak = sampler.peak_between(callback_state["vae_window_start"], pipe_end, component="vae_decoder")
            logger.info("[%s][INFER]   vae_decoder    %8.3fs", pass_tag, vae_infer_time)

            # RELEASE happens right after decode, BEFORE saving the image, so image
            # I/O time can never leak into any measured metric.
            release_component(pipe, "vae_decoder", sampler, logger)
            if getattr(pipe, "vae_encoder", None) is not None:
                release_component(pipe, "vae_encoder", sampler, logger)  # bonus cleanup, not benchmarked

            vae_trough_after = sampler.current_rss()

            pass_results.append(
                ComponentResult(
                    resolution=resolution_label,
                    component="vae_decoder",
                    tier=config.precision,
                    model_label=f"{COMPONENT_DISPLAY['vae_decoder']}-{precision_labels['vae_decoder']}",
                    precision=precision_labels["vae_decoder"],
                    load_time_s=callback_state["vae_load_time"],
                    infer_time_s=vae_infer_time,
                    baseline_rss=callback_state["vae_baseline"],
                    after_load_rss=callback_state["vae_after_load"],
                    peak_rss=vae_peak,
                    trough_after_release=vae_trough_after,
                )
            )

            if save_image:
                # ---- save image (excluded from every timed metric above) ----
                images_dir = Path(args.output_dir) / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                pass_image_path = images_dir / f"flux2_klein_{config_tag}.png"
                out.images[0].save(pass_image_path)
                logger.info("Saved image: %s (size=%s)", pass_image_path, out.images[0].size)

        except Exception as exc:  # noqa: BLE001 - one configuration failing must not crash the matrix
            pass_error = f"{type(exc).__name__}: {exc}"
            logger.error("Configuration %s FAILED in %s: %s", config_tag, pass_tag, pass_error)
            logger.debug("Traceback:", exc_info=True)

        finally:
            sampler.stop()
            try:
                del pipe
            except Exception:
                pass
            gc.collect()

        completed = {r.component for r in pass_results}
        for component in COMPONENT_ORDER:
            if component in completed:
                continue
            fallback_precision = "FP16" if component == "vae_decoder" else config.precision.upper()
            pass_results.append(
                ComponentResult(
                    resolution=resolution_label,
                    component=component,
                    tier=config.precision,
                    model_label=f"{COMPONENT_DISPLAY[component]}-{fallback_precision}",
                    precision=fallback_precision,
                    load_time_s=None,
                    infer_time_s=None,
                    baseline_rss=None,
                    after_load_rss=None,
                    peak_rss=None,
                    trough_after_release=None,
                    error=pass_error or "not reached",
                )
            )
        order_index = {name: i for i, name in enumerate(COMPONENT_ORDER)}
        pass_results.sort(key=lambda r: order_index[r.component])

        return pass_results, pass_image_path, pass_error

    def _mean_time(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return float(sum(valid) / len(valid))

    def _mean_int(values: List[Optional[int]]) -> Optional[int]:
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return int(round(sum(valid) / len(valid)))

    measured_passes: List[List[ComponentResult]] = []
    image_path: Optional[Path] = None

    for idx in range(args.warmup_runs):
        pass_tag = f"warmup {idx + 1}/{args.warmup_runs}"
        logger.info("[%s] starting", pass_tag)
        _, _, warmup_error = _single_pass(pass_tag=pass_tag, save_image=False)
        if warmup_error is not None:
            return _placeholder_rows(resolution_label, config.precision, warmup_error), None, warmup_error

    for idx in range(args.measure_runs):
        pass_tag = f"measure {idx + 1}/{args.measure_runs}"
        logger.info("[%s] starting", pass_tag)
        pass_results, pass_image_path, measure_error = _single_pass(
            pass_tag=pass_tag,
            save_image=(idx == args.measure_runs - 1),
        )
        if measure_error is not None:
            return _placeholder_rows(resolution_label, config.precision, measure_error), None, measure_error
        measured_passes.append(pass_results)
        if pass_image_path is not None:
            image_path = pass_image_path

    aggregated: List[ComponentResult] = []
    for component in COMPONENT_ORDER:
        component_runs = [
            run_result
            for run in measured_passes
            for run_result in run
            if run_result.component == component
        ]
        if not component_runs:
            fallback_precision = "FP16" if component == "vae_decoder" else config.precision.upper()
            aggregated.append(
                ComponentResult(
                    resolution=resolution_label,
                    component=component,
                    tier=config.precision,
                    model_label=f"{COMPONENT_DISPLAY[component]}-{fallback_precision}",
                    precision=fallback_precision,
                    load_time_s=None,
                    infer_time_s=None,
                    baseline_rss=None,
                    after_load_rss=None,
                    peak_rss=None,
                    trough_after_release=None,
                    error="not reached",
                )
            )
            continue

        aggregated.append(
            ComponentResult(
                resolution=resolution_label,
                component=component,
                tier=config.precision,
                model_label=component_runs[0].model_label,
                precision=component_runs[0].precision,
                load_time_s=_mean_time([r.load_time_s for r in component_runs]),
                infer_time_s=_mean_time([r.infer_time_s for r in component_runs]),
                baseline_rss=_mean_int([r.baseline_rss for r in component_runs]),
                after_load_rss=_mean_int([r.after_load_rss for r in component_runs]),
                peak_rss=_mean_int([r.peak_rss for r in component_runs]),
                trough_after_release=_mean_int([r.trough_after_release for r in component_runs]),
                error=None,
            )
        )

    return aggregated, image_path, None


# --------------------------------------------------------------------------- #
# Output rendering / export
# --------------------------------------------------------------------------- #
def result_row_cells(result: ComponentResult) -> List[str]:
    return [
        result.resolution,
        result.model_label,
        fmt_time(result.load_time_s),
        fmt_time(result.infer_time_s),
        fmt_gb(result.ram_after_load_delta),
        fmt_gb(result.peak_ram_delta),
    ]


def sort_results_for_display(results: List[ComponentResult]) -> List[ComponentResult]:
    """Sort rows as: 512-FP16, 512-INT4, 1024-FP16, 1024-INT4, per component."""

    def resolution_key(resolution: str) -> Tuple[int, int]:
        w_str, h_str = resolution.lower().split("x")
        return int(w_str), int(h_str)

    tier_order = {"fp16": 0, "int4": 1}
    component_order = {name: idx for idx, name in enumerate(COMPONENT_ORDER)}
    return sorted(
        results,
        key=lambda r: (
            resolution_key(r.resolution),
            tier_order.get(r.tier.lower(), 99),
            component_order.get(r.component, 99),
        ),
    )


def to_display_rows(results: List[ComponentResult], steps: int) -> List[List[str]]:
    """Render rows with client-style grouped Resolution column.

    For each (resolution, precision) block, only the first row shows the
    resolution plus steps on two lines; the next two component rows keep the
    resolution cell empty.
    """
    sorted_results = sort_results_for_display(results)
    rows: List[List[str]] = []
    previous_group: Optional[Tuple[str, str]] = None
    for result in sorted_results:
        group = (result.resolution, result.tier)
        resolution_cell = ""
        if group != previous_group:
            resolution_cell = f"{result.resolution}<br>{steps}steps"
            previous_group = group

        base_cells = result_row_cells(result)
        base_cells[0] = resolution_cell
        rows.append(base_cells)
    return rows


def print_summary_table(results: List[ComponentResult], logger: logging.Logger, steps: int) -> None:
    rows = to_display_rows(results, steps=steps)
    widths = [len(h) for h in RESULT_HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [render(RESULT_HEADERS), "-+-".join("-" * w for w in widths)]
    lines.extend(render(row) for row in rows)
    logger.info("\n%s", "\n".join(lines))


def write_benchmark_csv(path: Path, results: List[ComponentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = [r for r in results if r.error]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_HEADERS)
        for result in sort_results_for_display(results):
            writer.writerow(result_row_cells(result))
        if errors:
            writer.writerow([])
            writer.writerow(["# Errors"])
            for r in errors:
                writer.writerow([f"# {r.resolution}/{r.component}: {r.error}"])


def write_benchmark_md(
    path: Path,
    results: List[ComponentResult],
    environment: Dict[str, Any],
    run_params: Dict[str, Any],
    model_metadata: Dict[str, Dict[str, Any]],
    notes: List[str],
) -> None:
    lines: List[str] = ["# FLUX.2-klein OpenVINO Benchmark Results (No-CFG, RAM Pipeline-Start Baseline)", ""]

    lines.append("## Environment")
    lines.append("")
    for key, value in environment.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Run parameters")
    lines.append("")
    for key, value in run_params.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Model metadata")
    lines.append("")
    for precision, meta in model_metadata.items():
        lines.append(f"- **{precision}**: {meta}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| " + " | ".join(RESULT_HEADERS) + " |")
    lines.append("| " + " | ".join(["---"] * len(RESULT_HEADERS)) + " |")
    for row in to_display_rows(results, steps=int(run_params["num_inference_steps"])):
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start "
        "RSS (single shared baseline per configuration). "
        "See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values."
    )
    lines.append("- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).")
    if not _HAS_PSUTIL:
        lines.append("- `psutil` is not installed, so all RAM values are `N/A`. Install it with `pip install psutil`.")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")

    errors = [r for r in results if r.error]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r.resolution} / {r.component}**: {r.error}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_memory_details_csv(path: Path, results: List[ComponentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(MEMORY_DETAIL_HEADERS)
        for r in results:
            writer.writerow(
                [
                    r.resolution,
                    r.component,
                    r.precision,
                    fmt_gb(r.baseline_rss),
                    fmt_gb(r.after_load_rss),
                    fmt_gb(r.peak_rss),
                    fmt_gb(r.ram_after_load_delta),
                    fmt_gb(r.peak_ram_delta),
                    fmt_gb(r.trough_after_release),
                ]
            )


def write_metadata_json(
    path: Path,
    environment: Dict[str, Any],
    run_params: Dict[str, Any],
    model_metadata: Dict[str, Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": environment,
        "run_parameters": run_params,
        "models": model_metadata,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def export_results(
    output_dir: Path,
    results: List[ComponentResult],
    environment: Dict[str, Any],
    run_params: Dict[str, Any],
    model_metadata: Dict[str, Dict[str, Any]],
    notes: List[str],
    logger: logging.Logger,
) -> None:
    """Write every output file (called after each configuration so progress
    is never lost if a later configuration crashes the whole process)."""
    results_dir = output_dir / "results"
    write_benchmark_csv(results_dir / "flux2_klein_benchmark.csv", results)
    write_benchmark_md(results_dir / "flux2_klein_benchmark.md", results, environment, run_params, model_metadata, notes)
    write_memory_details_csv(results_dir / "flux2_klein_memory_details.csv", results)
    write_metadata_json(results_dir / "flux2_klein_metadata.json", environment, run_params, model_metadata)
    logger.info("Exported results to %s", results_dir)


# --------------------------------------------------------------------------- #
# Matrix orchestration
# --------------------------------------------------------------------------- #
def _placeholder_rows(resolution_label: str, precision: str, error: str) -> List[ComponentResult]:
    rows = []
    for component in COMPONENT_ORDER:
        fallback_precision = "FP16" if component == "vae_decoder" else precision.upper()
        rows.append(
            ComponentResult(
                resolution=resolution_label,
                component=component,
                tier=precision,
                model_label=f"{COMPONENT_DISPLAY[component]}-{fallback_precision}",
                precision=fallback_precision,
                load_time_s=None,
                infer_time_s=None,
                baseline_rss=None,
                after_load_rss=None,
                peak_rss=None,
                trough_after_release=None,
                error=error,
            )
        )
    return rows


def run_benchmark_matrix(args: argparse.Namespace, logger: logging.Logger) -> List[ComponentResult]:
    """Run every (precision, resolution) cell requested on the CLI (INT4 only by default)."""
    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    resolutions = parse_resolutions(args.resolutions)
    model_dirs = {"int4": Path(args.int4_model_dir)}

    environment = collect_environment_metadata(args.device)
    logger.info("Environment: %s", json.dumps(environment, indent=2, default=str))

    run_params: Dict[str, Any] = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "max_sequence_length": args.max_sequence_length,
        "guidance_scale": GUIDANCE_SCALE,
        "cfg_enabled": False,
        "negative_prompt_embeds": None,
        "output_type": OUTPUT_TYPE,
        "seed": args.seed,
        "device": args.device,
        "resolutions": [f"{w}x{h}" for w, h in resolutions],
        "precisions": precisions,
    }

    model_metadata: Dict[str, Dict[str, Any]] = {}
    all_results: List[ComponentResult] = []
    notes: List[str] = []

    for precision in precisions:
        if precision not in model_dirs:
            msg = f"Unknown precision '{precision}' (expected 'fp16' or 'int4') - skipping"
            logger.warning(msg)
            notes.append(msg)
            continue

        model_dir = model_dirs[precision]
        if not (model_dir / "model_index.json").exists():
            msg = f"model dir for '{precision}' not found or incomplete: {model_dir}"
            logger.error(msg)
            notes.append(msg)
            continue
        model_metadata[precision] = collect_model_metadata(model_dir)

    for width, height in resolutions:
        for precision in precisions:
            if precision not in model_dirs:
                continue
            model_dir = model_dirs[precision]
            if not (model_dir / "model_index.json").exists():
                msg = f"model dir for '{precision}' not found or incomplete: {model_dir}"
                all_results.extend(_placeholder_rows(f"{width}x{height}", precision, msg))
                export_results(output_dir, all_results, environment, run_params, model_metadata, notes, logger)
                continue

            config = BenchmarkConfig(precision=precision, model_dir=model_dir, width=width, height=height)
            try:
                results, image_path, error = benchmark_single_configuration(config, args, logger)
            except Exception as exc:  # last-resort safety net around our own orchestration code
                logger.exception("Unexpected failure for %s %sx%s", precision, width, height)
                error = f"{type(exc).__name__}: {exc}"
                results = _placeholder_rows(f"{width}x{height}", precision, error)
                image_path = None

            all_results.extend(results)
            if error:
                notes.append(f"{precision} {width}x{height}: FAILED - {error}")
            elif image_path is not None:
                notes.append(f"{precision} {width}x{height}: image saved to {image_path}")

            # Export after every configuration so partial progress is never
            # lost if a later configuration crashes the process.
            export_results(output_dir, all_results, environment, run_params, model_metadata, notes, logger)

    print_summary_table(all_results, logger, steps=args.steps)
    return all_results


# --------------------------------------------------------------------------- #
# Logging / CLI / main
# --------------------------------------------------------------------------- #
def setup_logging(output_dir: Path, level: str) -> logging.Logger:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "flux2_klein_benchmark.log"

    logger = logging.getLogger("flux2_klein_benchmark")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()  # avoid duplicate handlers on repeated calls (e.g. notebooks)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FLUX.2-klein OpenVINO benchmark matrix: LOAD -> INFERENCE -> RELEASE lifecycle per "
            "component (text_encoder -> transformer -> vae_decoder), CFG always disabled, RAM "
            "measured as deltas from pipeline start point. Runs INT4 @ 512x512 and 1024x1024 by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="GPU", help="OpenVINO device: GPU, CPU or AUTO.")
    parser.add_argument("--precisions", default="int4", help="Comma-separated precision tiers to run.")
    parser.add_argument(
        "--resolutions",
        default="512x512,1024x1024",
        help="Comma-separated sizes; 'N' means NxN, or use 'WxH'.",
    )
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps per image.")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warmup passes per configuration (not reported).")
    parser.add_argument("--measure-runs", type=int, default=5, help="Measured passes per configuration (averaged).")
    parser.add_argument("--max-sequence-length", type=int, default=512, help="Text encoder max sequence length.")
    parser.add_argument("--prompt", default="Astronaut on the moon", help="Text-to-image prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible latents.")
    parser.add_argument(
        "--fp16-model-dir",
        default="./benchmark_outputs/models/flux2-klein-fp16",
        help="OpenVINO FP16 model directory.",
    )
    parser.add_argument(
        "--int4-model-dir",
        default="./benchmark_outputs/models/flux2-klein-int4",
        help="OpenVINO INT4 model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_outputs",
        help="Root output directory (images/, results/, logs/ are created under it).",
    )
    parser.add_argument("--cache-dir", default="./ov_model_cache", help="from_pretrained cache_dir.")
    parser.add_argument(
        "--stabilize-seconds",
        type=float,
        default=1.0,
        help="Seconds to sample RSS before each component to find its RAM trough.",
    )
    parser.add_argument(
        "--ram-interval-ms",
        type=float,
        default=30.0,
        help="Background RAM sampling interval in milliseconds (suggested 20-50ms).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    parser.add_argument(
        "--ov-versions",
        default="26.1,26.2,26.3",
        help="Comma-separated OV versions to benchmark in launcher mode.",
    )
    parser.add_argument(
        "--run-single-ov-version",
        default="",
        help="Internal switch: run one already-installed OV version in this process.",
    )
    parser.add_argument(
        "--skip-ov-install",
        action="store_true",
        help="Skip pip install step in launcher mode.",
    )
    return parser


def parse_ov_versions(spec: str) -> List[str]:
    return [token.strip() for token in spec.split(",") if token.strip()]


def normalize_ov_version_for_pip(version_tag: str) -> str:
    """Map user-facing OV tags to pip constraints.

    Examples:
        26.1 -> 2026.1.*
        2026.2 -> 2026.2.*
        2026.3.1 -> 2026.3.1
    """
    match = re.fullmatch(r"(\d{2}|\d{4})\.(\d+)(?:\.(\d+))?", version_tag)
    if not match:
        raise ValueError(f"Invalid OV version tag: {version_tag}")

    major_raw, minor, patch = match.groups()
    major = f"20{major_raw}" if len(major_raw) == 2 else major_raw
    if patch is None:
        return f"{major}.{minor}.*"
    return f"{major}.{minor}.{patch}"


def safe_ov_tag(version_tag: str) -> str:
    return version_tag.replace(".", "_")


def is_ov_26_3_tag(version_tag: str) -> bool:
    """Return True when OV tag points to the 2026.3 line."""
    match = re.fullmatch(r"(\d{2}|\d{4})\.(\d+)(?:\.(\d+))?", version_tag)
    if not match:
        return False
    major_raw, minor, _patch = match.groups()
    major = f"20{major_raw}" if len(major_raw) == 2 else major_raw
    return major == "2026" and minor == "3"


def install_openvino_version(version_tag: str, python_exe: str, logger: logging.Logger) -> None:
    spec = normalize_ov_version_for_pip(version_tag)
    if is_ov_26_3_tag(version_tag):
        nightly_cmd = [
            python_exe,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--pre",
            "openvino==2026.3.0rc1",
            "openvino-tokenizers==2026.3.0.0rc1",
            "openvino-genai==2026.3.0.0rc1",
            "--extra-index-url",
            "https://storage.openvinotoolkit.org/simple/wheels/nightly",
        ]
        logger.info(
            "Installing OpenVINO for OV %s via nightly-first strategy: %s",
            version_tag,
            " ".join(nightly_cmd),
        )
        try:
            subprocess.run(nightly_cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Nightly install for OV %s failed (%s). Falling back to stable spec openvino==%s",
                version_tag,
                exc,
                spec,
            )

    cmd = [python_exe, "-m", "pip", "install", "--upgrade", f"openvino=={spec}"]
    logger.info("Installing OpenVINO for OV %s using stable spec: %s", version_tag, " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_child_command(args: argparse.Namespace, output_dir: Path, version_tag: str) -> List[str]:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--run-single-ov-version",
        version_tag,
        "--skip-ov-install",
        "--device",
        args.device,
        "--precisions",
        "int4",
        "--resolutions",
        args.resolutions,
        "--steps",
        str(args.steps),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measure-runs",
        str(args.measure_runs),
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--fp16-model-dir",
        args.fp16_model_dir,
        "--int4-model-dir",
        args.int4_model_dir,
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        args.cache_dir,
        "--stabilize-seconds",
        str(args.stabilize_seconds),
        "--ram-interval-ms",
        str(args.ram_interval_ms),
        "--log-level",
        args.log_level,
    ]
    return cmd


def aggregate_ov_results(root_output_dir: Path, version_tags: List[str], logger: logging.Logger) -> None:
    rows: List[List[str]] = []
    notes: List[str] = []

    for version_tag in version_tags:
        version_dir = root_output_dir / f"ov_{safe_ov_tag(version_tag)}" / "results"
        csv_path = version_dir / "flux2_klein_benchmark.csv"
        if not csv_path.exists():
            notes.append(f"OV {version_tag}: missing {csv_path}")
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header_seen = False
            for record in reader:
                if not record:
                    continue
                if record[0].startswith("#"):
                    continue
                if not header_seen:
                    header_seen = True
                    continue
                if len(record) != len(RESULT_HEADERS):
                    continue
                rows.append([version_tag, *record])

    root_results = root_output_dir / "results"
    root_results.mkdir(parents=True, exist_ok=True)
    csv_out = root_results / "flux2_klein_benchmark_ov_matrix.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OV_MATRIX_HEADERS)
        writer.writerows(rows)
        if notes:
            writer.writerow([])
            writer.writerow(["# Notes"])
            for note in notes:
                writer.writerow([f"# {note}"])

    md_out = root_results / "flux2_klein_benchmark_ov_matrix.md"
    md_lines = ["# FLUX.2-klein OV Version Matrix (INT4 only)", ""]
    md_lines.append("| " + " | ".join(OV_MATRIX_HEADERS) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(OV_MATRIX_HEADERS)) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    if notes:
        md_lines.append("")
        md_lines.append("## Notes")
        md_lines.append("")
        for note in notes:
            md_lines.append(f"- {note}")
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Aggregated OV matrix results to %s", root_results)


def run_ov_version_matrix(args: argparse.Namespace, logger: logging.Logger) -> int:
    version_tags = parse_ov_versions(args.ov_versions)
    if not version_tags:
        logger.error("No OV versions provided in --ov-versions")
        return 1

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for version_tag in version_tags:
        logger.info("=" * 78)
        logger.info("OV version matrix cell: %s", version_tag)
        logger.info("=" * 78)

        if not args.skip_ov_install:
            try:
                install_openvino_version(version_tag, sys.executable, logger)
            except Exception as exc:  # noqa: BLE001 - continue matrix run even if one install fails
                message = f"OV {version_tag}: install failed - {type(exc).__name__}: {exc}"
                logger.error(message)
                failures.append(message)
                continue

        version_output_dir = root_output_dir / f"ov_{safe_ov_tag(version_tag)}"
        cmd = build_child_command(args, output_dir=version_output_dir, version_tag=version_tag)
        logger.info("Running benchmark for OV %s", version_tag)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            message = f"OV {version_tag}: benchmark failed with exit code {result.returncode}"
            logger.error(message)
            failures.append(message)

    aggregate_ov_results(root_output_dir, version_tags, logger)

    if failures:
        logger.error("Completed with failures: %s", failures)
        return 1
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    args.device = args.device.upper()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, args.log_level)

    if not _HAS_PSUTIL:
        logger.warning("psutil is not installed - all RAM metrics will be N/A. Install with `pip install psutil`.")

    logger.info("FLUX.2-klein OpenVINO benchmark matrix starting")
    logger.info("Args: %s", vars(args))

    try:
        if args.run_single_ov_version:
            logger.info("Single OV process mode for OV tag %s", args.run_single_ov_version)
            run_benchmark_matrix(args, logger)
        else:
            return run_ov_version_matrix(args, logger)
    except Exception:
        logger.exception("Benchmark matrix failed with an unhandled error")
        return 1

    logger.info("Benchmark matrix finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
