# FLUX.2-klein OpenVINO Benchmark Results (No-CFG, RAM Pipeline-Start Baseline)

## Environment

- **os**: Windows-10-10.0.26200-SP0
- **python_version**: 3.10.11
- **openvino_version**: 2026.2.1-21919-ede283a88e3-releases/2026/2
- **optimum_intel_version**: 2.1.0.dev0
- **cpu_name**: Intel(R) Core(TM) Ultra X7 368H
- **gpu_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **gpu_driver_version**: 32.0.101.8860
- **total_memory_bytes**: 68247695360
- **total_memory_gb**: 63.6
- **device**: GPU
- **selected_device_name**: Intel(R) Arc(TM) B390 GPU (iGPU)
- **available_devices**: ['CPU', 'GPU', 'NPU']
- **collected_at**: 2026-07-16T01:07:58

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
| 512x512<br>4steps | Text Encoder-FP16 | 4.749s | 0.294s | 14.53GB | 14.86GB |
|  | Transformer-FP16 | 4.160s | 2.315s | 13.49GB | 13.61GB |
|  | Small VAE Decoder-FP16 | 0.095s | 0.241s | -0.30GB | 0.11GB |
| 512x512<br>4steps | Text Encoder-INT4 | 1.552s | 0.210s | 3.66GB | 3.98GB |
|  | Transformer-INT4 | 1.293s | 1.436s | 3.58GB | 3.61GB |
|  | Small VAE Decoder-FP16 | 0.074s | 0.125s | 0.11GB | 0.14GB |
| 1024x1024<br>4steps | Text Encoder-FP16 | 4.345s | 0.281s | 14.77GB | 15.15GB |
|  | Transformer-FP16 | 3.972s | 6.674s | 12.61GB | 13.48GB |
|  | Small VAE Decoder-FP16 | 0.099s | 0.874s | -0.49GB | 1.49GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 1.483s | 0.212s | 3.89GB | 4.25GB |
|  | Transformer-INT4 | 1.269s | 5.443s | 3.84GB | 3.87GB |
|  | Small VAE Decoder-FP16 | 0.072s | 0.818s | -0.34GB | 1.55GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- fp16 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_2\images\flux2_klein_fp16_512x512_4steps_no_cfg.png
- int4 512x512: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_2\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- fp16 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_2\images\flux2_klein_fp16_1024x1024_4steps_no_cfg.png
- int4 1024x1024: image saved to notebooks\openvino\benchmark_outputs\full_matrix_2026_2\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
