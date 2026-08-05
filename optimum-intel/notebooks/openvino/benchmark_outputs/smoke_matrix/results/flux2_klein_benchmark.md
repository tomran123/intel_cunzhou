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
- **collected_at**: 2026-07-16T00:39:28

## Run parameters

- **prompt**: Astronaut on the moon
- **num_inference_steps**: 4
- **warmup_runs**: 1
- **measure_runs**: 1
- **max_sequence_length**: 512
- **guidance_scale**: 1.0
- **cfg_enabled**: False
- **negative_prompt_embeds**: None
- **output_type**: pil
- **seed**: 0
- **device**: GPU
- **resolutions**: ['512x512']
- **precisions**: ['fp16']

## Model metadata

- **fp16**: {'model_dir': 'benchmark_outputs\\models\\flux2-klein-fp16', 'repo_id': 'black-forest-labs/FLUX.2-klein-4B', 'diffusers_version': '0.37.1', 'revision': 'e7b7dc27f91deacad38e78976d1f2b499d76a294', 'optimum_version_at_export': None}

## Results

| Resolution | Model | Load Time | Infer Time | RAM After Load | Peak RAM |
| --- | --- | --- | --- | --- | --- |
| 512x512<br>4steps | Text Encoder-FP16 | 3.214s | 1.586s | 7.51GB | 15.32GB |
|  | Transformer-FP16 | 2.940s | 3.274s | 6.03GB | 13.39GB |
|  | Small VAE Decoder-FP16 | 0.113s | 0.276s | -1.03GB | -0.48GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- fp16 512x512: image saved to notebooks\openvino\benchmark_outputs\smoke_matrix\images\flux2_klein_fp16_512x512_4steps_no_cfg.png
