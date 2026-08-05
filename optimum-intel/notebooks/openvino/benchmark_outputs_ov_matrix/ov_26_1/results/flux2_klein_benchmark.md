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
- **collected_at**: 2026-07-20T01:44:10

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
- **precisions**: ['int4']

## Model metadata

- **int4**: {'model_dir': 'benchmark_outputs\\models\\flux2-klein-int4', 'repo_id': 'black-forest-labs/FLUX.2-klein-4B', 'diffusers_version': '0.37.1', 'revision': 'e7b7dc27f91deacad38e78976d1f2b499d76a294', 'optimum_version_at_export': '2.2.0'}

## Results

| Resolution | Model | Load Time | Infer Time | RAM After Load | Peak RAM |
| --- | --- | --- | --- | --- | --- |
| 512x512<br>4steps | Text Encoder-INT4 | 3.107s | 0.232s | 3.87GB | 6.02GB |
|  | Transformer-INT4 | 2.690s | 1.495s | 3.64GB | 5.63GB |
|  | Small VAE Decoder-FP16 | 0.118s | 0.228s | -0.19GB | 0.08GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 3.120s | 0.247s | 3.99GB | 6.15GB |
|  | Transformer-INT4 | 2.589s | 5.993s | 3.92GB | 5.98GB |
|  | Small VAE Decoder-FP16 | 0.119s | 1.350s | 0.11GB | 1.63GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- int4 512x512: image saved to benchmark_outputs_ov_matrix\ov_26_1\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- int4 1024x1024: image saved to benchmark_outputs_ov_matrix\ov_26_1\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
