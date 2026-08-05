# FLUX.2-klein OpenVINO Benchmark Results (image-to-image)

## Environment

- **Benchmark device**: GPU
- **Selected device name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **CPU model**: Intel(R) Core(TM) Ultra X7 368H
- **GPU device name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **GPU driver version**: 32.0.101.8737
- **Total system memory**: 63.6G
- **Available OV devices**: CPU, GPU, NPU
- **OpenVINO version**: 2026.2.1-21919-ede283a88e3-releases/2026/2
- **optimum-intel version**: 2.1.0.dev0
- **Python version**: 3.10.11
- **OS version**: Windows-10-10.0.26200-SP0
- **Timestamp**: 2026-07-08 01:58:30

## Results

| Resolution | Model | Load Time | Infer Time | RAM After Load | Peak RAM |
| --- | --- | --- | --- | --- | --- |
| 512x512 4steps | Text Encoder-FP16 | 6.52s | 0.42s | 15.7G | 33.1G |
| 512x512 4steps | Transformer-FP16 | 6.19s | 6.67s | 30.2G | 33.1G |
| 512x512 4steps | Small VAE decoder-FP16 | 0.18s | 0.12s | 30.4G | 33.1G |
| 1024x1024 4steps | Text Encoder-FP16 | 3.95s | 0.42s | 18.5G | 38.0G |
| 1024x1024 4steps | Transformer-FP16 | 3.55s | 39.36s | 32.6G | 38.0G |
| 1024x1024 4steps | Small VAE decoder-FP16 | 0.09s | 0.46s | 32.7G | 38.0G |
| 512x512 4steps | Text Encoder-int4 | 3.17s | 0.33s | 8.6G | 37.9G |
| 512x512 4steps | Transformer-int4 | 2.96s | 5.65s | 12.2G | 37.9G |
| 512x512 4steps | Small VAE decoder-FP16 | 0.23s | 0.12s | 12.4G | 37.9G |
| 1024x1024 4steps | Text Encoder-int4 | 1.50s | 0.34s | 8.2G | 16.7G |
| 1024x1024 4steps | Transformer-int4 | 1.18s | 29.08s | 11.7G | 16.7G |
| 1024x1024 4steps | Small VAE decoder-FP16 | 0.07s | 0.51s | 11.8G | 16.7G |

## Notes

- **Load Time**: OpenVINO `compile_model` time per component; the shared IR-read time of `from_pretrained` is listed per case below.
- **Infer Time**: per-component `forward` time during measured runs only (warmup excluded), averaged over runs; the Transformer covers all denoising steps.
- **RAM After Load**: cumulative process RSS right after that component compiled.
- **Peak RAM**: process-level peak RSS during the whole case (same value shown on every row of a case).
- fp16 tier model dir: benchmark_outputs_i2i\models\flux2-klein-fp16 (weight-format 'fp16'; labels reflect this export configuration).
- fp16 tier @ 512x512 4steps: input image = images\512x512.jpg; pipeline IR read = 1.12s (shared, not split per component); end-to-end inference = 7.31s per image; Peak RAM 33.1G is a process-level value shared by all rows of this case.
- fp16 tier @ 1024x1024 4steps: input image = images\1024x1024.jpg; pipeline IR read = 2.42s (shared, not split per component); end-to-end inference = 40.59s per image; Peak RAM 38.0G is a process-level value shared by all rows of this case.
- int4 tier model dir: benchmark_outputs_i2i\models\flux2-klein-int4 (weight-format 'int4'; labels reflect this export configuration).
- int4 tier @ 512x512 4steps: input image = images\512x512.jpg; pipeline IR read = 2.79s (shared, not split per component); end-to-end inference = 6.21s per image; Peak RAM 37.9G is a process-level value shared by all rows of this case.
- int4 tier @ 1024x1024 4steps: input image = images\1024x1024.jpg; pipeline IR read = 1.44s (shared, not split per component); end-to-end inference = 30.31s per image; Peak RAM 16.7G is a process-level value shared by all rows of this case.
