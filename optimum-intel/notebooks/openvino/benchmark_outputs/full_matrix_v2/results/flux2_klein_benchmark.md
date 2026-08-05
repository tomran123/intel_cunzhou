# FLUX.2-klein OpenVINO Benchmark Results (No-CFG, RAM Pipeline-Start Baseline)

## Environment

- **os**: Windows-10-10.0.26200-SP0
- **python_version**: 3.10.11
- **openvino_version**: 2026.3.0-22433-0e931a6a878-releases/2026/3
- **optimum_intel_version**: 2.1.0.dev0
- **cpu_name**: Intel(R) Core(TM) Ultra X7 368H
- **gpu_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **gpu_driver_version**: 32.0.101.8860
- **total_memory_bytes**: 68247695360
- **total_memory_gb**: 63.6
- **device**: GPU
- **selected_device_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **available_devices**: ['CPU', 'GPU', 'NPU']
- **collected_at**: 2026-07-16T00:52:57

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
| 512x512<br>4steps | Text Encoder-FP16 | 3.080s | 1.713s | 7.40GB | 15.17GB |
|  | Transformer-FP16 | 2.934s | 3.624s | 6.79GB | 14.10GB |
|  | Small VAE Decoder-FP16 | 0.073s | 0.269s | -0.32GB | 0.21GB |
| 512x512<br>4steps | Text Encoder-INT4 | 1.101s | 0.624s | 2.13GB | 4.44GB |
|  | Transformer-INT4 | 0.897s | 1.742s | 1.87GB | 3.78GB |
|  | Small VAE Decoder-FP16 | 0.055s | 0.148s | 0.06GB | 0.15GB |
| 1024x1024<br>4steps | Text Encoder-FP16 | 3.203s | 1.822s | 7.50GB | 15.50GB |
|  | Transformer-FP16 | 3.050s | 8.586s | 6.15GB | 14.25GB |
|  | Small VAE Decoder-FP16 | 0.079s | 1.867s | -0.35GB | 1.73GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 1.168s | 0.703s | 2.13GB | 4.60GB |
|  | Transformer-INT4 | 0.935s | 6.617s | 2.03GB | 3.95GB |
|  | Small VAE Decoder-FP16 | 0.065s | 1.597s | -0.57GB | 1.33GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- fp16 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix_v2\images\flux2_klein_fp16_512x512_4steps_no_cfg.png
- int4 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix_v2\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- fp16 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix_v2\images\flux2_klein_fp16_1024x1024_4steps_no_cfg.png
- int4 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix_v2\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
