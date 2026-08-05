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
- **collected_at**: 2026-07-16T00:44:13

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
| 512x512<br>4steps | Text Encoder-FP16 | 3.054s | 1.548s | 7.44GB | 15.13GB |
|  | Transformer-FP16 | 2.865s | 3.079s | 6.97GB | 14.18GB |
|  | Small VAE Decoder-FP16 | 0.070s | 0.228s | -0.28GB | 0.21GB |
|  | Small VAE Decoder-FP16 | 0.051s | 0.141s | 0.06GB | 0.16GB |
| 512x512<br>4steps | Text Encoder-INT4 | 1.048s | 0.580s | 2.13GB | 4.43GB |
|  | Transformer-INT4 | 0.849s | 1.823s | 1.89GB | 3.79GB |
| 1024x1024<br>4steps | Text Encoder-FP16 | 3.089s | 1.691s | 7.49GB | 15.51GB |
|  | Transformer-FP16 | 3.118s | 8.330s | 6.15GB | 14.25GB |
|  | Small VAE Decoder-FP16 | 0.075s | 1.225s | -0.34GB | 1.68GB |
|  | Small VAE Decoder-FP16 | 0.053s | 1.154s | -0.58GB | 1.44GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 1.103s | 0.650s | 2.13GB | 4.43GB |
|  | Transformer-INT4 | 0.909s | 5.892s | 1.88GB | 3.80GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- fp16 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix\images\flux2_klein_fp16_512x512_4steps_no_cfg.png
- int4 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- fp16 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix\images\flux2_klein_fp16_1024x1024_4steps_no_cfg.png
- int4 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
