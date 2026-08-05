# FLUX.2-klein OpenVINO Benchmark Results (No-CFG, RAM Pipeline-Start Baseline)

## Environment

- **os**: Windows-10-10.0.26200-SP0
- **python_version**: 3.10.11
- **openvino_version**: 2026.1.0-21367-63e31528c62-releases/2026/1
- **optimum_intel_version**: 2.1.0.dev0
- **cpu_name**: Intel(R) Core(TM) Ultra X7 368H
- **gpu_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **gpu_driver_version**: 32.0.101.8860
- **total_memory_bytes**: 68247695360
- **total_memory_gb**: 63.6
- **device**: GPU
- **selected_device_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **available_devices**: ['CPU', 'GPU', 'NPU']
- **collected_at**: 2026-07-16T01:29:07

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
| 512x512<br>4steps | Text Encoder-FP16 | 7.279s | 0.261s | 14.53GB | 22.02GB |
|  | Transformer-FP16 | 6.977s | 1.847s | 13.73GB | 21.16GB |
|  | Small VAE Decoder-FP16 | 0.105s | 0.231s | -0.07GB | 0.34GB |
| 512x512<br>4steps | Text Encoder-INT4 | 1.881s | 0.220s | 3.72GB | 5.87GB |
|  | Transformer-INT4 | 1.851s | 1.483s | 3.38GB | 5.28GB |
|  | Small VAE Decoder-FP16 | 0.087s | 0.135s | 0.03GB | 0.12GB |
| 1024x1024<br>4steps | Text Encoder-FP16 | 8.099s | 0.284s | 14.18GB | 21.93GB |
|  | Transformer-FP16 | 7.905s | 9.824s | 12.60GB | 19.83GB |
|  | Small VAE Decoder-FP16 | 0.145s | 1.010s | -0.52GB | 1.46GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 2.458s | 0.236s | 4.01GB | 6.16GB |
|  | Transformer-INT4 | 2.022s | 5.698s | 3.82GB | 5.71GB |
|  | Small VAE Decoder-FP16 | 0.086s | 1.155s | -0.42GB | 1.36GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- fp16 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_1\images\flux2_klein_fp16_512x512_4steps_no_cfg.png
- int4 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_1\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- fp16 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_1\images\flux2_klein_fp16_1024x1024_4steps_no_cfg.png
- int4 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_1\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
