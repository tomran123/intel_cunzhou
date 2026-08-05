"""FLUX.2-klein OpenVINO performance benchmark.

This script measures load / inference time and RAM usage of the FLUX.2-klein
text-to-image pipeline on an Intel platform (defaults to the Intel GPU on the
PTL machine, e.g. `--device GPU`).

IMPORTANT — pipeline reuse
--------------------------
It intentionally reuses the *same* OpenVINO pipeline that the working
`flux2_klein_inference.py` script already uses (``optimum.intel.OVDiffusionPipeline``
which resolves to ``OVFlux2KleinPipeline`` for FLUX.2-klein). It does NOT
re-implement or redesign the pipeline; it only:
  * loads the pipeline,
  * compiles each component,
  * instruments the component ``forward`` methods to time them,
  * runs warmup + measured generations,
  * samples process RSS to track peak RAM.

Configurations benchmarked (2 resolutions x 2 precision tiers = 4 cases):
  * Resolutions : 512x512 and 1024x1024, 4 steps (both configurable)
  * FP16 tier   : Text Encoder-FP16, Transformer-FP16, Small VAE decoder-FP16
  * int4 tier   : Text Encoder-int4, Transformer-int4, Small VAE decoder-FP16

Each precision tier maps to one exported OpenVINO model directory. The tier is
prepared once with ``optimum-cli export openvino --weight-format {fp16,int4}``
and cached under ``<output-dir>/models`` (this export is a one-time preparation
step and is NOT counted in "Load Time"). You can also point a tier at an
already-exported directory with ``--fp16-model-dir`` / ``--int4-model-dir`` to
skip exporting entirely.

Metric definitions
------------------
* Load Time      : OpenVINO ``compile_model`` time for that component on the
                   target device. The one-time IR read done by
                   ``from_pretrained`` is shared across components and is
                   reported in the notes section, not split per component.
* Infer Time     : Wall-clock time spent inside that component's ``forward``
                   during the MEASURED runs only (warmup excluded), divided by
                   the number of runs. The Transformer value therefore covers
                   all ``--steps`` denoising calls for a single image.
* RAM After Load : Process RSS captured immediately after that component is
                   compiled. It is cumulative (memory of all components loaded
                   so far), because all components live in one Python process.
* Peak RAM       : Maximum process RSS observed while THAT component was active
                   -- i.e. while its OpenVINO sub-models were compiling and
                   while its ``forward`` ran during the measured runs. Each
                   component therefore gets its OWN peak instead of one shared
                   case-wide value. Because process RSS is cumulative, a
                   component that loads/runs later includes the memory of
                   everything already resident plus its own transient
                   activation buffers (this is intended and documented in the
                   notes).

Honesty rules (per task spec):
  * Nothing is hard-coded / faked.
  * Anything that cannot be measured is reported as ``N/A`` with the reason
    given in the notes section and in code comments.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# psutil is used for RSS memory sampling. It is optional: if it is not
# installed, all RAM metrics are reported as N/A instead of crashing.
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_PSUTIL = False

import openvino as ov

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# We report memory in binary gigabytes (GiB) with a trailing "G", e.g. "10.6G".
BYTES_PER_GB = 1024**3

# Components are grouped so the table has exactly one row per logical block,
# matching the reference format. Each group lists the pipeline attribute names
# that may hold an OpenVINO sub-model; only the ones that actually exist on the
# loaded pipeline are used. This makes the script robust across FLUX variants
# (e.g. single vs. multiple text encoders, transformer vs. unet).
COMPONENT_GROUPS: List[tuple] = [
    ("Text Encoder", ["text_encoder", "text_encoder_2", "text_encoder_3"]),
    ("Transformer", ["transformer", "unet"]),
    ("Small VAE decoder", ["vae_decoder"]),
]

# Precision tiers. "weight_format" is passed to `optimum-cli export openvino`.
# "precision" holds the label shown in the "Model" column for each group.
TIERS: Dict[str, dict] = {
    "fp16": {
        "weight_format": "fp16",
        "precision": {
            "Text Encoder": "FP16",
            "Transformer": "FP16",
            "Small VAE decoder": "FP16",
        },
    },
    "int4": {
        "weight_format": "int4",
        "precision": {
            "Text Encoder": "int4",
            "Transformer": "int4",
            # VAE decoder is deliberately kept at FP16 in the int4 tier.
            "Small VAE decoder": "FP16",
        },
    },
}

TABLE_HEADERS = ["Resolution", "Model", "Load Time", "Infer Time", "RAM After Load", "Peak RAM"]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_time(seconds: Optional[float]) -> str:
    """Format a duration as e.g. ``12.18s`` (or ``N/A`` when unavailable)."""
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}s"


def fmt_mem(num_bytes: Optional[int]) -> str:
    """Format a byte count as e.g. ``10.6G`` (or ``N/A`` when unavailable)."""
    if num_bytes is None:
        return "N/A"
    return f"{num_bytes / BYTES_PER_GB:.1f}G"


# --------------------------------------------------------------------------- #
# Memory sampling
# --------------------------------------------------------------------------- #
def current_rss() -> Optional[int]:
    """Return the current process RSS in bytes, or None if psutil is missing."""
    if not _HAS_PSUTIL:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


class PerComponentPeakSampler:
    """Background thread that samples this process' RSS and attributes each
    sample to the component group that is currently *active*.

    A group is marked active (a) while its OpenVINO sub-models are being
    compiled and (b) while any of its ``forward`` methods runs during the
    measured runs. ``peak_for(group)`` therefore returns the maximum process
    RSS observed *while that specific component was doing work*, giving a
    distinct Peak RAM per component/stage instead of one case-wide value.

    Because process RSS is cumulative, a component that loads/runs later
    naturally includes the memory of everything already resident (e.g. the
    transformer stage "costs" more than the text-encoder stage before it, and
    its large transient activation buffers push its peak above the others).

    If psutil is unavailable the sampler becomes a no-op and every
    ``peak_for`` returns None.
    """

    def __init__(self, interval_seconds: float):
        self.interval = interval_seconds
        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        self._peaks: Dict[str, int] = {}
        self._active: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _fold(self, group_label: Optional[str], rss: int) -> None:
        """Record ``rss`` as the new peak for ``group_label`` if it is larger.

        Caller must hold ``self._lock``.
        """
        if group_label is None:
            return
        if rss > self._peaks.get(group_label, 0):
            self._peaks[group_label] = rss

    def record(self, group_label: Optional[str]) -> None:
        """Fold the current RSS into ``group_label`` immediately.

        Used at the end of a compile step and around every ``forward`` so that
        very short activity windows still contribute at least one sample.
        """
        if self._proc is None or group_label is None:
            return
        try:
            rss = self._proc.memory_info().rss
        except Exception:
            return
        with self._lock:
            self._fold(group_label, rss)

    def set_active(self, group_label: Optional[str]) -> Optional[str]:
        """Mark ``group_label`` as the active group and return the previous one.

        The current RSS is folded into the newly active group right away so the
        entry point of the activity window is always captured, even if it is
        shorter than one sampling interval.
        """
        if self._proc is None:
            previous, self._active = self._active, group_label
            return previous
        try:
            rss = self._proc.memory_info().rss
        except Exception:
            rss = None
        with self._lock:
            previous = self._active
            self._active = group_label
            if rss is not None:
                self._fold(group_label, rss)
        return previous

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                with self._lock:
                    self._fold(self._active, rss)
            except Exception:
                # Never let a sampling hiccup interfere with the benchmark.
                pass
            # wait() returns early if stop() is called -> responsive shutdown.
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._proc is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def peak_for(self, group_label: str) -> Optional[int]:
        """Return the peak RSS attributed to ``group_label`` (None if unknown)."""
        if self._proc is None:
            return None
        with self._lock:
            return self._peaks.get(group_label)


# --------------------------------------------------------------------------- #
# Environment information
# --------------------------------------------------------------------------- #
def _optimum_intel_version() -> str:
    try:
        from optimum.intel.version import __version__

        return __version__
    except Exception:
        try:
            from importlib.metadata import version

            return version("optimum-intel")
        except Exception:
            return "N/A"


def _ov_property(core: "ov.Core", device: str, key: str) -> Optional[str]:
    """Safely read an OpenVINO device property, returning None on failure."""
    try:
        return str(core.get_property(device, key))
    except Exception:
        return None


def _windows_gpu_driver_version() -> Optional[str]:
    """Best-effort GPU driver version on Windows via WMI (Win32_VideoController).

    OpenVINO does not expose a driver-version property on all GPU builds, so on
    Windows we read it from the OS. Prefers an Intel/Arc adapter. Returns None
    on any failure so the caller can honestly fall back to N/A.
    """
    if platform.system() != "Windows":
        return None
    ps_cmd = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -match 'Intel|Arc' } | "
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


def collect_env(device: str) -> Dict[str, str]:
    """Collect environment / platform information for the report header.

    Values that cannot be obtained are reported as ``N/A`` rather than guessed.
    """
    core = ov.Core()
    # The device family (strip any ":0"/"AUTO:GPU" style suffixes) for queries.
    base_device = device.split(":")[0].upper() if device else "CPU"

    # OpenVINO exposes the CPU brand string and GPU name via FULL_DEVICE_NAME.
    cpu_name = _ov_property(core, "CPU", "FULL_DEVICE_NAME") or platform.processor() or "N/A"
    gpu_name = _ov_property(core, "GPU", "FULL_DEVICE_NAME") or "N/A"

    # GPU driver version key differs between OpenVINO builds; try known keys,
    # then fall back to the OS (Windows WMI) before giving up with N/A.
    gpu_driver = "N/A"
    for key in ("GPU_DRIVER_VERSION", "DRIVER_VERSION"):
        value = _ov_property(core, "GPU", key)
        if value:
            gpu_driver = value
            break
    if gpu_driver == "N/A":
        gpu_driver = _windows_gpu_driver_version() or "N/A"

    selected_name = _ov_property(core, base_device, "FULL_DEVICE_NAME") or base_device

    try:
        ov_version = ov.get_version()
    except Exception:
        ov_version = getattr(ov, "__version__", "N/A")

    total_mem = fmt_mem(psutil.virtual_memory().total) if _HAS_PSUTIL else "N/A"

    return {
        "Benchmark device": device,
        "Selected device name": selected_name,
        "CPU model": cpu_name,
        "GPU device name": gpu_name,
        "GPU driver version": gpu_driver,
        "Total system memory": total_mem,
        "Available OV devices": ", ".join(core.available_devices),
        "OpenVINO version": ov_version,
        "optimum-intel version": _optimum_intel_version(),
        "Python version": platform.python_version(),
        "OS version": platform.platform(),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
# Per-component inference-time instrumentation
# --------------------------------------------------------------------------- #
def instrument_component(component, sampler, group_label):
    """Wrap ``component.forward`` to accumulate wall time and attribute RSS.

    ``OVPipelinePart.__call__`` delegates to ``self.forward``; overriding the
    instance attribute therefore times every diffusers-side call to the
    component without touching the pipeline internals. While the wrapped
    ``forward`` runs, ``sampler`` attributes process RSS samples to
    ``group_label`` so each component gets its own Peak RAM. Returns ``(stats,
    restore)`` where ``stats`` holds accumulated time / call count and
    ``restore`` puts the original method back.
    """
    stats = {"time": 0.0, "calls": 0}
    original = component.forward

    def timed_forward(*args, **kwargs):
        previous = sampler.set_active(group_label)  # attribute RSS to this group
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stats["time"] += time.perf_counter() - start
            stats["calls"] += 1
            sampler.record(group_label)   # fold the post-forward RSS in too
            sampler.set_active(previous)  # restore the previously active group

    component.forward = timed_forward

    def restore():
        component.forward = original

    return stats, restore


# --------------------------------------------------------------------------- #
# Model preparation (one-time export per precision tier)
# --------------------------------------------------------------------------- #
def parse_resolutions(spec: str) -> List[tuple]:
    """Parse ``"512,1024"`` or ``"512x512,1024x768"`` into a list of (w, h)."""
    resolutions = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" in token:
            w, h = token.split("x")
            resolutions.append((int(w), int(h)))
        else:
            size = int(token)
            resolutions.append((size, size))
    return resolutions


def export_tier_model(model_id: str, out_dir: Path, weight_format: str) -> None:
    """Export ``model_id`` to OpenVINO IR at ``out_dir`` with the given weight
    format using the standard optimum-cli command.

    This is a one-time preparation step (cached on disk); its runtime is NOT
    included in any benchmark metric.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        model_id,
        "--weight-format",
        weight_format,
        str(out_dir),
    ]
    print(f"  [export] {' '.join(cmd)}")
    # check=True -> raises CalledProcessError, handled by the per-case try/except
    # so a failed export marks only that tier as failed instead of crashing.
    subprocess.run(cmd, check=True)


def ensure_tier_model(tier_key: str, args: argparse.Namespace) -> str:
    """Return a local OpenVINO model directory for the given tier.

    Resolution order:
      1. explicit ``--{tier}-model-dir`` override, if it exists;
      2. cached export under ``<output-dir>/models/flux2-klein-<tier>``;
      3. fresh ``optimum-cli`` export into the cache directory.
    """
    override = getattr(args, f"{tier_key}_model_dir")
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"--{tier_key}-model-dir does not exist: {override}")
        print(f"  Using provided {tier_key} model dir: {override}")
        return override

    cache_dir = Path(args.output_dir) / "models" / f"flux2-klein-{tier_key}"
    if (cache_dir / "model_index.json").exists():
        print(f"  Using cached {tier_key} model dir: {cache_dir}")
        return str(cache_dir)

    print(f"  Exporting {tier_key} tier (one-time, not timed) from {args.model_id} ...")
    export_tier_model(args.model_id, cache_dir, TIERS[tier_key]["weight_format"])
    return str(cache_dir)


# --------------------------------------------------------------------------- #
# Pipeline loading / generation
# --------------------------------------------------------------------------- #
def load_pipeline(model_dir: str, device: str):
    """Load the pipeline WITHOUT compiling so each component's compile time can
    be measured individually afterwards.

    Returns ``(pipe, ir_read_time)`` where ``ir_read_time`` is the shared cost
    of reading all component IRs from disk in ``from_pretrained``.
    """
    from optimum.intel import OVDiffusionPipeline

    start = time.perf_counter()
    pipe = OVDiffusionPipeline.from_pretrained(model_dir, device=device, compile=False)
    ir_read_time = time.perf_counter() - start
    return pipe, ir_read_time


def generate(pipe, prompt: str, width: int, height: int, steps: int):
    """Run one text-to-image generation using the existing pipeline call
    signature (same as flux2_klein_inference.py's text2image path)."""
    return pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        output_type="pil",
    )


# --------------------------------------------------------------------------- #
# Core benchmark of a single (tier, resolution) case
# --------------------------------------------------------------------------- #
def run_case(
    tier_key: str,
    width: int,
    height: int,
    model_dir: str,
    args: argparse.Namespace,
    notes: List[str],
) -> List[dict]:
    """Benchmark one precision tier at one resolution.

    Returns a list of result rows (one per component group). On failure it
    returns rows with N/A metrics and records the error in ``notes`` so the
    overall run keeps going.
    """
    resolution_label = f"{width}x{height} {args.steps}steps"
    precision = TIERS[tier_key]["precision"]
    print(f"\n=== Case: {tier_key} tier | {resolution_label} ===")

    sampler = PerComponentPeakSampler(args.ram_interval / 1000.0)
    sampler.start()

    pipe = None
    restores: List = []
    try:
        # ---- Load (read IR, then compile each component and time it) --------
        pipe, ir_read_time = load_pipeline(model_dir, args.device)

        group_load: Dict[str, Optional[float]] = {}
        group_rss: Dict[str, Optional[int]] = {}
        group_components: Dict[str, list] = {}

        for group_label, attr_names in COMPONENT_GROUPS:
            components = [getattr(pipe, name, None) for name in attr_names]
            components = [c for c in components if c is not None]
            if not components:
                continue  # this group does not exist for this model

            # Attribute every RSS sample taken during this group's compile to the
            # group, so its Peak RAM also reflects any transient compile spike.
            sampler.set_active(group_label)

            # Sum compile time across all sub-models that make up the group.
            compile_time = 0.0
            for comp in components:
                try:
                    comp.clear_requests()  # ensure a real (timed) compile happens
                except Exception:
                    pass
                start = time.perf_counter()
                comp.compile()
                compile_time += time.perf_counter() - start

            sampler.record(group_label)  # fold the post-compile RSS into the group
            sampler.set_active(None)      # stop attributing until this group runs

            group_load[group_label] = compile_time
            group_rss[group_label] = current_rss()  # cumulative RSS after compile
            group_components[group_label] = components

        # ---- Warmup (NOT timed) --------------------------------------------
        # Warmup runs use the un-instrumented forward so their cost can never
        # leak into the measured per-component inference time.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        for i in range(args.warmup):
            print(f"  warmup {i + 1}/{args.warmup} ...")
            generate(pipe, args.prompt, width, height, args.steps)

        # ---- Instrument component forwards AFTER warmup ---------------------
        infer_stats: Dict[str, list] = {}
        for group_label, components in group_components.items():
            holders = []
            for comp in components:
                stats, restore = instrument_component(comp, sampler, group_label)
                holders.append(stats)
                restores.append(restore)
            infer_stats[group_label] = holders

        # ---- Measured runs --------------------------------------------------
        last_image = None
        measured_start = time.perf_counter()
        for i in range(args.runs):
            print(f"  measured run {i + 1}/{args.runs} ...")
            result = generate(pipe, args.prompt, width, height, args.steps)
            last_image = result.images[0]
        e2e_per_image = (time.perf_counter() - measured_start) / max(args.runs, 1)

        # Remove instrumentation before doing anything else.
        for restore in restores:
            restore()
        restores = []

        # ---- Save one image so results can be visually verified -------------
        if last_image is not None:
            img_dir = Path(args.output_dir) / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"flux2_{tier_key}_{width}x{height}_{args.steps}steps.png"
            last_image.save(img_path)
            print(f"  saved image: {img_path}")

        sampler.stop()

        # ---- Build one row per component group ------------------------------
        rows = []
        for group_label, _ in COMPONENT_GROUPS:
            if group_label not in group_components:
                continue
            infer_time = sum(h["time"] for h in infer_stats[group_label]) / max(args.runs, 1)
            rows.append(
                {
                    "resolution": resolution_label,
                    "model": f"{group_label}-{precision[group_label]}",
                    "load_time": group_load[group_label],
                    "infer_time": infer_time,
                    "ram_after": group_rss[group_label],
                    # Per-component peak: max RSS while THIS component compiled or
                    # ran its forward during the measured runs (distinct per row).
                    "peak_ram": sampler.peak_for(group_label),
                }
            )

        notes.append(
            f"{tier_key} tier @ {resolution_label}: pipeline IR read = {fmt_time(ir_read_time)} "
            f"(shared, not split per component); end-to-end inference = {fmt_time(e2e_per_image)} per image; "
            f"Peak RAM is measured per component (max process RSS while that component was "
            f"compiling or running its forward)."
        )
        return rows

    except Exception as exc:  # noqa: BLE001 - one failing case must not stop others
        # Restore any instrumentation that was installed before the failure.
        for restore in restores:
            try:
                restore()
            except Exception:
                pass
        sampler.stop()
        err = f"{type(exc).__name__}: {exc}"
        print(f"  !! case FAILED: {err}", file=sys.stderr)
        traceback.print_exc()
        notes.append(f"{tier_key} tier @ {resolution_label}: FAILED - {err}")
        # Emit N/A rows so the failure is visible in the table without faking data.
        return [
            {
                "resolution": resolution_label,
                "model": f"{group_label}-{precision[group_label]}",
                "load_time": None,
                "infer_time": None,
                "ram_after": None,
                "peak_ram": None,
                "error": err,
            }
            for group_label, _ in COMPONENT_GROUPS
        ]

    finally:
        # ---- Cleanup between cases -----------------------------------------
        # Drop every reference to the pipeline and its compiled models, then run
        # the garbage collector. OpenVINO releases device (GPU) memory once the
        # CompiledModel / InferRequest objects are collected; there is no public
        # "flush GPU cache" API, so we rely on GC rather than faking a flush.
        try:
            del pipe
        except Exception:
            pass
        gc.collect()


# --------------------------------------------------------------------------- #
# Output rendering
# --------------------------------------------------------------------------- #
def rows_to_cells(rows: List[dict]) -> List[List[str]]:
    """Convert result rows into formatted string cells for display / files."""
    cells = []
    for row in rows:
        cells.append(
            [
                row["resolution"],
                row["model"],
                fmt_time(row["load_time"]),
                fmt_time(row["infer_time"]),
                fmt_mem(row["ram_after"]),
                fmt_mem(row["peak_ram"]),
            ]
        )
    return cells


def print_table(cells: List[List[str]]) -> None:
    """Pretty-print the results table to the terminal."""
    widths = [len(h) for h in TABLE_HEADERS]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(row: List[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    print("\n" + render(TABLE_HEADERS))
    print("-+-".join("-" * w for w in widths))
    for row in cells:
        print(render(row))


def write_csv(path: Path, cells: List[List[str]], env: Dict[str, str], notes: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # Environment header as leading comment-style rows.
        writer.writerow(["# Environment"])
        for key, value in env.items():
            writer.writerow([f"# {key}", value])
        writer.writerow([])
        writer.writerow(TABLE_HEADERS)
        writer.writerows(cells)
        if notes:
            writer.writerow([])
            writer.writerow(["# Notes"])
            for note in notes:
                writer.writerow([f"# {note}"])


def write_md(path: Path, cells: List[List[str]], env: Dict[str, str], notes: List[str]) -> None:
    lines: List[str] = ["# FLUX.2-klein OpenVINO Benchmark Results", ""]
    lines.append("## Environment")
    lines.append("")
    for key, value in env.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| " + " | ".join(TABLE_HEADERS) + " |")
    lines.append("| " + " | ".join(["---"] * len(TABLE_HEADERS)) + " |")
    for row in cells:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Load Time**: OpenVINO `compile_model` time per component; "
                 "the shared IR-read time of `from_pretrained` is listed per case below.")
    lines.append("- **Infer Time**: per-component `forward` time during measured runs only "
                 "(warmup excluded), averaged over runs; the Transformer covers all denoising steps.")
    lines.append("- **RAM After Load**: cumulative process RSS right after that component compiled.")
    lines.append("- **Peak RAM**: per-component peak process RSS - the max RSS observed while that "
                 "component was compiling or running its `forward` during the measured runs. "
                 "Values differ per component; because RSS is cumulative, later/heavier components "
                 "(e.g. the transformer) include everything already resident plus their own "
                 "transient activation buffers.")
    if not _HAS_PSUTIL:
        lines.append("- `psutil` is not installed, so all RAM values are `N/A`. "
                     "Install it with `pip install psutil` to capture memory metrics.")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark FLUX.2-klein OpenVINO inference (load/infer time + RAM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="GPU", help="OpenVINO device: GPU, CPU or AUTO.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs (not timed).")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per case.")
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps per image.")
    parser.add_argument("--resolutions", default="512,1024",
                        help="Comma-separated sizes; 'N' means NxN, or use 'WxH'.")
    parser.add_argument("--prompt", default="A scenic mountain landscape at sunset, highly detailed",
                        help="Text-to-image prompt.")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory for results + images.")
    parser.add_argument("--tiers", default="fp16,int4",
                        help="Comma-separated precision tiers to run (fp16, int4).")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-4B",
                        help="Hub id / path exported when a tier model dir is missing.")
    parser.add_argument("--fp16-model-dir", default=None,
                        help="Pre-exported FP16 OpenVINO model dir (skips export).")
    parser.add_argument("--int4-model-dir", default=None,
                        help="Pre-exported int4 OpenVINO model dir (skips export).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible images.")
    parser.add_argument("--ram-interval", type=float, default=50.0,
                        help="Peak-RAM sampling interval in milliseconds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = args.device.upper()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolutions = parse_resolutions(args.resolutions)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for tier in tiers:
        if tier not in TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {list(TIERS)}")

    env = collect_env(args.device)
    print("Environment:")
    for key, value in env.items():
        print(f"  {key}: {value}")
    if not _HAS_PSUTIL:
        print("  WARNING: psutil not installed -> RAM metrics will be N/A.", file=sys.stderr)

    all_rows: List[dict] = []
    notes: List[str] = []

    # Loop tier-first so each exported model is loaded once per resolution.
    for tier_key in tiers:
        try:
            model_dir = ensure_tier_model(tier_key, args)
        except Exception as exc:  # noqa: BLE001 - preparation failure marks the whole tier
            err = f"{type(exc).__name__}: {exc}"
            print(f"  !! could not prepare {tier_key} tier: {err}", file=sys.stderr)
            notes.append(
                f"{tier_key} tier: model preparation FAILED - {err}. "
                f"Export it manually with: optimum-cli export openvino --model {args.model_id} "
                f"--weight-format {TIERS[tier_key]['weight_format']} <dir> and pass --{tier_key}-model-dir <dir>."
            )
            precision = TIERS[tier_key]["precision"]
            for width, height in resolutions:
                for group_label, _ in COMPONENT_GROUPS:
                    all_rows.append(
                        {
                            "resolution": f"{width}x{height} {args.steps}steps",
                            "model": f"{group_label}-{precision[group_label]}",
                            "load_time": None,
                            "infer_time": None,
                            "ram_after": None,
                            "peak_ram": None,
                            "error": err,
                        }
                    )
            continue

        # Record precision provenance so the "Model" column labels are traceable.
        notes.append(
            f"{tier_key} tier model dir: {model_dir} "
            f"(weight-format '{TIERS[tier_key]['weight_format']}'; labels reflect this export configuration)."
        )

        for width, height in resolutions:
            rows = run_case(tier_key, width, height, model_dir, args, notes)
            all_rows.extend(rows)

    # ---- Emit results -----------------------------------------------------
    cells = rows_to_cells(all_rows)
    print_table(cells)

    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_results.md"
    write_csv(csv_path, cells, env, notes)
    write_md(md_path, cells, env, notes)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
