# FLUX.2-klein OpenVINO Benchmark Results (No-CFG, RAM Pipeline-Start Baseline)

## Environment

- **os**: Windows-10-10.0.26200-SP0
- **python_version**: 3.10.11
- **openvino_version**: 2026.2.0-21903-52ddc073857-releases/2026/2
- **optimum_intel_version**: 2.1.0.dev0
- **cpu_name**: Intel(R) Core(TM) Ultra X7 368H
- **gpu_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **gpu_driver_version**: 32.0.101.8860
- **total_memory_bytes**: 68247695360
- **total_memory_gb**: 63.6
- **device**: GPU
- **selected_device_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **available_devices**: ['CPU', 'GPU', 'NPU']
- **collected_at**: 2026-07-20T00:52:09

## Run parameters

- **prompt**: Astronaut on the moon
- **num_inference_steps**: 4
- **warmup_runs**: 3
- **measure_runs**: 5
- **max_sequence_length**: 512
- **guidance_scale**: 1.0
- **cfg_enabled**: False
- **negative_prompt_embeds**: None
- **output_type**: pil
- **seed**: 0
- **device**: GPU
- **resolutions**: ['512x512', '1024x1024']
- **precisions**: ['fp16', 'int4']

## Model metadata

- **fp16**: {'model_dir': 'benchmark_outputs\\models\\flux2-klein-fp16', 'repo_id': 'black-forest-labs/FLUX.2-klein-4B', 'diffusers_version': '0.37.1', 'revision': 'e7b7dc27f91deacad38e78976d1f2b499d76a294', 'optimum_version_at_export': None}
- **int4**: {'model_dir': 'benchmark_outputs\\models\\flux2-klein-int4', 'repo_id': 'black-forest-labs/FLUX.2-klein-4B', 'diffusers_version': '0.37.1', 'revision': 'e7b7dc27f91deacad38e78976d1f2b499d76a294', 'optimum_version_at_export': '2.2.0'}

## Results

| Resolution | Model | Load Time | Infer Time | RAM After Load | Peak RAM |
| --- | --- | --- | --- | --- | --- |
| 512x512<br>4steps | Text Encoder-FP16 | 7.716s | 0.337s | 14.48GB | 14.77GB |
|  | Transformer-FP16 | 6.799s | 2.590s | 13.58GB | 13.89GB |
|  | Small VAE Decoder-FP16 | 0.149s | 0.239s | -0.15GB | 0.29GB |
| 512x512<br>4steps | Text Encoder-INT4 | 1.669s | 0.216s | 3.46GB | 3.78GB |
|  | Transformer-INT4 | 1.356s | 1.507s | 3.39GB | 3.42GB |
|  | Small VAE Decoder-FP16 | 0.075s | 0.129s | 0.05GB | 0.08GB |
| 1024x1024<br>4steps | Text Encoder-FP16 | 7.235s | 0.366s | 14.32GB | 14.56GB |
|  | Transformer-FP16 | 6.327s | 8.317s | 12.63GB | 13.62GB |
|  | Small VAE Decoder-FP16 | 0.151s | 0.922s | -0.52GB | 1.46GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 2.690s | 0.281s | 3.85GB | 4.23GB |
|  | Transformer-INT4 | 2.044s | 5.881s | 3.81GB | 3.84GB |
|  | Small VAE Decoder-FP16 | 0.108s | 1.096s | -0.34GB | 1.44GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- fp16 512x512: image saved to benchmark_outputs\images\flux2_klein_fp16_512x512_4steps_no_cfg.png
- int4 512x512: image saved to benchmark_outputs\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- fp16 1024x1024: image saved to benchmark_outputs\images\flux2_klein_fp16_1024x1024_4steps_no_cfg.png
- int4 1024x1024: image saved to benchmark_outputs\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
