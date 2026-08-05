# Internship Portfolio — Cunzhou Ran

| | |
|---|---|
| **Intern** | Cunzhou Ran |
| **Department** | Intel China, SCP / CPEG (Software team) |
| **Internship period** | 05/18/2026 – 08/07/2026 |

This repository is an internship portfolio: AI model optimization and runtime deployment with **OpenVINO™** across Intel CPU / iGPU / NPU platforms. It contains the FLUX.2-klein Panther Lake bring-up work directly (scripts, benchmark reports, BKM document — model weights and OpenVINO compile caches are intentionally excluded, see below), and links out to the other project repositories.

## Overview

This internship focused on AI model optimization and runtime deployment using OpenVINO™ across Intel platforms. Key highlights include engineering the **Real-ESRGAN super-resolution pipeline** and delivering comprehensive evaluation reports, bringing up **FLUX.2-klein** on **Panther Lake** with BKMs, achieving a **2.8x speedup and 27% memory reduction** for **PaddleOCR-VL 1.5** via NNCF INT8 quantization, and building an end-to-end **SAM3 + LaMa** object-removal pipeline with a Gradio demo. Alongside these four formally reviewed deliverables, the work also included rebuilding the **PP-OCRv6 medium OCR pipeline** with a GPU/NPU/CPU heterogeneous deployment and its own OpenVINO BKM.

## Project index

| # | Project | Skill area | Location |
|---|---|---|---|
| 1 | Real-ESRGAN Super-Resolution | AI Runtime Engineering & Deployment | [tomran123/SuperResolution](https://github.com/tomran123/SuperResolution) |
| 2 | SAM3 + LaMa Object Removal | AI Runtime Engineering & Deployment | [tomran123/object-removal-demo](https://github.com/tomran123/object-removal-demo) |
| 3 | PaddleOCR-VL 1.5 NNCF INT8 Quantization | Model Quantization & Optimization | [tomran123/openvino_notebooks:portfolio](https://github.com/tomran123/openvino_notebooks/tree/portfolio/notebooks/paddleocr_vl) |
| 4 | PP-OCRv6 medium heterogeneous OCR pipeline + BKM *(additional)* | Model Quantization & Optimization | [tomran123/openvino_notebooks:portfolio](https://github.com/tomran123/openvino_notebooks/tree/portfolio/notebooks/paddle-ocr-webcam) |
| 5 | FLUX.2-klein bring-up on Panther Lake | Intel Hardware Bring-up & Profiling | [optimum-intel/notebooks/openvino/](optimum-intel/notebooks/openvino/) *(in this repo)* |

## About this repository's layout

- Projects 1–4 live in their **own repositories** (linked above) and aren't duplicated here.
  - Projects 3–4 are original additions on top of a full local checkout of the upstream [openvinotoolkit/openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks) sample repo; rather than push that entire multi-GB upstream tree, only the two folders with original work (`notebooks/paddleocr_vl/`, `notebooks/paddle-ocr-webcam/`) are pushed, as a single squashed snapshot on the `portfolio` branch, with the upstream CI/CD (GitHub Actions workflows, Jenkinsfile, `.ci/`, `.docker/`, Dockerfile, Makefile) removed.
- Project 5 (FLUX.2-klein) is included directly **in this repo**, under [optimum-intel/notebooks/openvino/](optimum-intel/notebooks/openvino/) — mirroring its path inside the upstream [huggingface/optimum-intel](https://github.com/huggingface/optimum-intel) checkout it was built in. Only the original bring-up scripts, benchmark reports/images, and the BKM document are included; **all exported model weights and OpenVINO compile caches were excluded** (`models/`, `ov_model_cache/`, `FLUX.2-klein-4B/` — together several hundred GB of re-creatable binary artifacts, not source work).

---

## 1. AI Runtime Engineering & Deployment

### 1.1 Real-ESRGAN Super-Resolution — [tomran123/SuperResolution](https://github.com/tomran123/SuperResolution)

A Gradio-based image super-resolution app wrapping the Real-ESRGAN model family, with an OpenVINO inference backend and a from-scratch benchmarking/analysis harness.

**Engineering work**

- Built an automated benchmark sweep across **5 models** (`RealESRGAN_x4plus`, `RealESRNet_x4plus`, `RealESRGAN_x4plus_anime_6B`, `RealESRGAN_x2plus`, `realesr-general-x4v3`) × **3 input resolutions** × **2 outscales** × **OpenVINO/PyTorch backends** × **3 tile strategies** (no-tile / half-tile / quarter-tile) — 90 paired configurations in total, run unattended with OOM/error capture so a single case failure never aborts the sweep.
- Added a background high-frequency (10 ms) RSS-polling thread to capture **peak memory**, not just before/after snapshots — important because Real-ESRGAN's intermediate feature maps spike well above steady-state usage.
- Diagnosed and worked around an OpenVINO conversion OOM/hang on large-resolution dummy inputs (1024² has ~33x the elements of the small calibration image) by converting once on a small calibration image and reusing the cached IR — and recommended pre-compiled IR distribution instead of on-device conversion for production.
- Caught a **correctness regression that the raw numbers alone would have hidden**: `realesr-general-x4v3` was the fastest model in every benchmark, but its OpenVINO output was visibly inconsistent on manual inspection, so it was kept as an experimental/fast-preview option instead of becoming the shipped default.

**Results** (12th Gen Intel Core i7-1270P, Iris Xe iGPU, 16 GB shared memory; averaged over 90 paired configs)

| Metric | PyTorch CPU | OpenVINO iGPU | Change |
|---|---:|---:|---:|
| Avg inference latency | 52,339 ms | 9,339 ms | **-82.2%** |
| Avg throughput | 0.201 FPS | 0.777 FPS | **+286%** |
| Avg peak RSS | 3,207.5 MB | 3,517.9 MB | +9.7% |

Latency reduction scaled with image size (-64.5% at small / -80.2% at 512px / -82.4% at 1024px). Quarter-tile inference on 1024px images cut peak RSS by ~40% for ~24% extra latency (e.g. `x4plus` @ 1024px: ~9.9 GB→~4.1 GB peak RSS). Final recommendation: default to **`RealESRGAN_x4plus` + OpenVINO + no tiling**, with quarter-tiling as an opt-in for memory-constrained large-image cases.

Full write-up: `benchmark_report_sections_7_10.md` in the repo above.

### 1.2 SAM3 + LaMa Object Removal Demo — [tomran123/object-removal-demo](https://github.com/tomran123/object-removal-demo)

Text-prompted object removal: upload an image, type an object name (`cat`, `person`, `car`, ...), and it's segmented by **SAM3** and erased/inpainted by **LaMa**. Runs on PyTorch or OpenVINO through a single backend switch, with a CLI and a Gradio web UI.

```
image + text prompt  ->  SAM3 (text -> mask)  ->  mask cleanup (dilation)  ->  LaMa (inpaint)  ->  result
```

**Engineering work**

- Modular pipeline split across `config.py`, `sam3_segmenter.py`, `lama_inpainter.py`, `common.py` and `pipeline.py`, each independently testable (`test_lama.py` validates the LaMa stage alone before wiring in SAM3).
- One `backend` flag (`pytorch` / `openvino`) drives both stages consistently — SAM3 text→mask and LaMa inpainting share the same device-resolution and fallback logic (CPU/GPU/AUTO with automatic fallback to CPU when GPU/AUTO isn't available).
- Merges all SAM3 instances matching a prompt into a single mask, then dilates it to remove edge-halo artifacts before inpainting.
- A `triton` CUDA-only-dependency stub is registered at import time so the SAM3 codebase (which expects Triton) still imports cleanly on CPU-only environments.
- Per-stage benchmark instrumentation (SAM3 time / mask-cleanup time / LaMa time / total) surfaced directly in the Gradio UI.

---

## 2. Model Quantization & Optimization

### 2.1 PaddleOCR-VL 1.5 — NNCF INT8 Quantization — [tomran123/openvino_notebooks:portfolio](https://github.com/tomran123/openvino_notebooks/tree/portfolio/notebooks/paddleocr_vl)

End-to-end PaddleOCR-VL-1.5 → OpenVINO conversion and quantization work: exporting the vision-language OCR model to OpenVINO IR and evaluating three compression configurations against a PyTorch baseline for both **speed/memory** and **text-output fidelity**.

**Engineering work**

- Benchmark harness (`benchmark_paddleocr_vl.py`) runs each configuration in an isolated subprocess (so memory/state never leaks between configs), with a background peak-RSS sampler and per-stage timing split (vision embedding vs. LLM generation vs. end-to-end).
- Four configurations compared: PyTorch baseline, **OV uncompressed** (FP16), **OV config A** (LLM INT8 weight compression + vision **INT8_cw** compressed weights), and **OV config B** (config A + LLM **INT8_dq** dynamic quantization, group size 128).
- Independently built a **7-scenario, 12-image** test dataset — general/document, curved page, lighting/illumination, scanned document, on-screen capture, seal/stamp, and skewed page — and a batch harness (`batch_test_images_paddleocr_vl.py`) that runs all 4 configurations over every image and diffs the text output against the PyTorch reference (character-level accuracy + similarity).
- Root-caused 3 outlier images instead of averaging them in blindly: `curving2.jpg` / `screen2.jpg` turned out to be **base-model repetition collapse present even in PyTorch** (not an OpenVINO/quantization defect), while `illumination.jpg` was an **OpenVINO-only generation-stability issue** on formula-dense regions (present even without any compression) — separating these from normal samples avoids a misleading blended average.

**Results**

On the 9 "normal" images (outliers excluded), consistency vs. the PyTorch reference:

| Configuration | Avg accuracy | Notes |
|---|---:|---|
| OV uncompressed (FP16) | 98.46% | Accuracy baseline |
| OV config A (INT8_cw only) | 91.54% | Larger fluctuation on long-text samples |
| **OV config B (INT8_cw + INT8_dq)** | **98.91%** | Best of all three, despite being the most compressed |

Config B — the most aggressively compressed configuration — was recommended for deployment: it delivered a **~2.8x end-to-end speedup and ~27% memory-footprint reduction** versus the uncompressed baseline while matching (and on this sample set, slightly exceeding) full-precision OpenVINO text fidelity.

### 2.2 PP-OCRv6 medium — heterogeneous GPU/NPU/CPU OCR pipeline *(additional work)* — [tomran123/openvino_notebooks:portfolio](https://github.com/tomran123/openvino_notebooks/tree/portfolio/notebooks/paddle-ocr-webcam)

Rebuilt the sample's OCR pipeline on top of the newer, larger **PP-OCRv6 medium** detection + recognition models (up from the original PP-OCRv3 lightweight ~9.4M models), and authored a full deployment BKM.

**Engineering work**

- Converted the official `PP-OCRv6_medium_det_onnx` / `PP-OCRv6_medium_rec_onnx` Hugging Face models to OpenVINO FP32 IR, and rebuilt every pre/post-processing step to match PaddleX's official components exactly (`DetResizeForTest`, `NormalizeImage`, `DBPostProcess`, `SortQuadBoxes`, `CropByPolys`, `OCRReisizeNormImg`, CTC decode) with the official 18,710-class character table (versus the old ~6,623-class table) — a mismatch here silently produces garbled text even though the model still "runs".
- Final heterogeneous device split: **GPU** for full-frame dynamic-shape text detection, **NPU** for recognition on a fixed `[6, 3, 48, 320]` reshaped batch, **CPU** as fallback for oversized/dynamic-width text plus all pre/post-processing.
- Found and fixed a real correctness bug: dynamic-batch (batch=6) recognition on `AUTO`/GPU silently **mixed up results across batch slots**. Fixed-shape NPU and dynamic-shape CPU both independently verified at 100% token agreement, so the shipped pipeline pins recognition to those two paths and avoids dynamic-batch AUTO/GPU.
- Two-level parity validation: model-level (99.999% detection threshold-map agreement and 100% recognition token agreement vs. ONNX Runtime, MAE ≈ 1.52e-10) and end-to-end (13/13 text regions matched ONNX Runtime exactly on first/middle/last video frames).

**Results:** ~168–172 ms/frame (5.8–5.9 FPS) end-to-end, versus ~1,968 ms/frame (0.5 FPS) for the initial broken PP-OCRv6 port — an **~11x throughput recovery** alongside fixing a garbled-text accuracy regression.

---

## 3. Intel Hardware Bring-up & Profiling

### 3.1 FLUX.2-klein bring-up on Panther Lake — [optimum-intel/notebooks/openvino/](optimum-intel/notebooks/openvino/)

Early enablement and performance characterization of **FLUX.2-klein-4B** (Black Forest Labs' diffusion transformer, via `optimum-intel`'s `OVFlux2KleinPipeline`) on **Panther Lake** client silicon — Intel® Core™ Ultra X7 368H CPU with an Intel® Arc™ B390 iGPU — across three consecutive OpenVINO 2026.x point releases.

**Engineering work**

- Benchmark harness ([flux2_klein_benchmark.py](optimum-intel/notebooks/openvino/flux2_klein_benchmark.py), plus an image-to-image variant [flux2_klein_benchmark_i2i.py](optimum-intel/notebooks/openvino/flux2_klein_benchmark_i2i.py)) instruments each pipeline component's `forward()` (text encoder / transformer / VAE decoder) individually for load time and inference time, and runs a background RSS sampler for peak-memory tracking — all reusing the same production `optimum.intel.OVDiffusionPipeline` rather than a reimplementation.
- Compared **FP16** vs. **INT4** weight-compressed export tiers at **512×512** and **1024×1024**, with CFG disabled (single transformer pass per step) for a clean latency measurement.
- Built an OpenVINO **version regression matrix** ([flux2_klein_benchmark_matrix_ov_versions.py](optimum-intel/notebooks/openvino/flux2_klein_benchmark_matrix_ov_versions.py)) tracking the same INT4 configuration across OpenVINO **26.1 → 26.2 → 26.3**, to catch performance or memory regressions across toolkit releases.
- Authored the official BKM as a Word document ([make_bkm_docx.py](optimum-intel/notebooks/openvino/make_bkm_docx.py)) that is **generated directly from the measured CSV** — no hand-typed numbers — so the document stays reproducible and auditable against the raw benchmark data.

**Results** (1024×1024, 4 steps, no-CFG)

| Component | FP16 infer / peak RAM | INT4 infer / peak RAM |
|---|---:|---:|
| Text Encoder | 0.297s / 15.05GB | 0.252s / 4.20GB |
| Transformer | 6.852s / 13.57GB | 5.684s / 3.88GB |
| VAE Decoder | 0.978s / 1.48GB | 0.901s / 1.38GB |

INT4 weight compression's main win is **memory** (roughly a 3.5x peak-RAM cut on the transformer) with a modest latency improvement on top. Across the OV version matrix, the INT4 tier's load times consistently improved release over release (e.g. 512×512 text-encoder load: 2.44s → 1.60s → 1.14s from 26.1 to 26.3), while peak-RAM movement between releases was flagged for continued regression tracking.

**Try it:** `python flux2_klein_inference.py` (single generation), `python flux2_klein_benchmark.py` (full FP16/INT4 sweep) — run from [optimum-intel/notebooks/openvino/](optimum-intel/notebooks/openvino/) with `optimum-intel` and its OpenVINO extras installed. BKM: [FLUX2-klein_PTL_OpenVINO_BKM.docx](optimum-intel/notebooks/openvino/FLUX2-klein_PTL_OpenVINO_BKM.docx), raw results: [benchmark_outputs/results/flux2_klein_benchmark.md](optimum-intel/notebooks/openvino/benchmark_outputs/results/flux2_klein_benchmark.md).

---

## Skills developed

| Skill area | What I learned | How I applied it |
|---|---|---|
| **AI Runtime Engineering & Deployment** | OpenVINO™ Runtime, execution backends, dynamic-shape handling, and execution profiling across Intel CPU and iGPU architectures | Built and refactored pipeline implementations for Real-ESRGAN (super-resolution) and an end-to-end SAM3 + LaMa object-removal workflow with interactive Gradio interfaces |
| **Model Quantization & Optimization** | NNCF quantization methodologies, dynamic quantization parameters (`INT8_dq`), and weight compression techniques (`INT8_cw`) for vision-language models | Quantized PaddleOCR-VL 1.5, achieving a ~2.8x end-to-end acceleration and 27% memory footprint reduction while authoring dynamic-dataset benchmark reports |
| **Intel Hardware Bring-up & Profiling** | Hardware bring-up procedures, memory peak profiling, and sub-model latency breakdown methodologies on Intel silicon platforms | Led the early deployment of FLUX.2-klein on Panther Lake (Core Ultra X7 368H + Intel® Arc™ B390 GPU) across OpenVINO 2026.x releases and authored the official BKM |
