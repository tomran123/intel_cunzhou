# FLUX.2-klein OpenVINO Benchmark Results

## Environment

- **Benchmark device**: GPU
- **Selected device name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **CPU model**: Intel(R) Core(TM) Ultra X7 368H
- **GPU device name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **GPU driver version**: 32.0.101.8860
- **Total system memory**: 63.6G
- **Available OV devices**: CPU, GPU, NPU
- **OpenVINO version**: 2026.2.0-21903-52ddc073857-releases/2026/2
- **optimum-intel version**: 2.1.0.dev0
- **Python version**: 3.10.11
- **OS version**: Windows-10-10.0.26200-SP0
- **Timestamp**: 2026-07-19 20:09:58

## Results

| Resolution | Model | Load Time | Infer Time | RAM After Load | Peak RAM |
| --- | --- | --- | --- | --- | --- |
| 512x512 4steps | Text Encoder-FP16 | 8.70s | 0.25s | 15.8G | 32.3G |
| 512x512 4steps | Transformer-FP16 | 12.67s | 2.05s | 30.3G | 32.1G |
| 512x512 4steps | Small VAE decoder-FP16 | 0.42s | 0.13s | 30.5G | 32.1G |
| 1024x1024 4steps | Text Encoder-FP16 | 7.21s | 0.29s | 17.4G | 34.9G |
| 1024x1024 4steps | Transformer-FP16 | 7.44s | 9.23s | 30.5G | 34.8G |
| 1024x1024 4steps | Small VAE decoder-FP16 | 0.23s | 0.56s | 30.7G | 34.8G |
| 512x512 4steps | Text Encoder-int4 | 8.54s | 0.19s | 6.2G | 11.4G |
| 512x512 4steps | Transformer-int4 | 7.56s | 1.74s | 9.4G | 11.2G |
| 512x512 4steps | Small VAE decoder-FP16 | 1.40s | 0.13s | 9.8G | 11.2G |
| 1024x1024 4steps | Text Encoder-int4 | 3.26s | 0.23s | 7.2G | 14.0G |
| 1024x1024 4steps | Transformer-int4 | 3.12s | 6.50s | 10.7G | 13.9G |
| 1024x1024 4steps | Small VAE decoder-FP16 | 0.24s | 0.59s | 10.9G | 13.9G |

## Notes

- **Load Time**: OpenVINO `compile_model` time per component; the shared IR-read time of `from_pretrained` is listed per case below.
- **Infer Time**: per-component `forward` time during measured runs only (warmup excluded), averaged over runs; the Transformer covers all denoising steps.
- **RAM After Load**: cumulative process RSS right after that component compiled.
- **Peak RAM**: per-component peak process RSS - the max RSS observed while that component was compiling or running its `forward` during the measured runs. Values differ per component; because RSS is cumulative, later/heavier components (e.g. the transformer) include everything already resident plus their own transient activation buffers.
- fp16 tier model dir: benchmark_outputs\models\flux2-klein-fp16 (weight-format 'fp16'; labels reflect this export configuration).
- fp16 tier @ 512x512 4steps: pipeline IR read = 1.13s (shared, not split per component); end-to-end inference = 2.46s per image; Peak RAM is measured per component (max process RSS while that component was compiling or running its forward).
- fp16 tier @ 1024x1024 4steps: pipeline IR read = 3.47s (shared, not split per component); end-to-end inference = 10.15s per image; Peak RAM is measured per component (max process RSS while that component was compiling or running its forward).
- int4 tier model dir: benchmark_outputs\models\flux2-klein-int4 (weight-format 'int4'; labels reflect this export configuration).
- int4 tier @ 512x512 4steps: pipeline IR read = 3.96s (shared, not split per component); end-to-end inference = 2.08s per image; Peak RAM is measured per component (max process RSS while that component was compiling or running its forward).
- int4 tier @ 1024x1024 4steps: pipeline IR read = 2.17s (shared, not split per component); end-to-end inference = 7.40s per image; Peak RAM is measured per component (max process RSS while that component was compiling or running its forward).
