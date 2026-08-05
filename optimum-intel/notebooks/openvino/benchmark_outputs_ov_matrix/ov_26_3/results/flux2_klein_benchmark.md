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
- **collected_at**: 2026-07-20T01:35:07

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
| 512x512<br>4steps | Text Encoder-INT4 | 1.136s | 0.657s | 2.13GB | 4.45GB |
|  | Transformer-INT4 | 0.942s | 1.944s | 1.87GB | 3.78GB |
|  | Small VAE Decoder-FP16 | 0.058s | 0.158s | 0.06GB | 0.15GB |
| 1024x1024<br>4steps | Text Encoder-INT4 | 1.171s | 0.726s | 2.13GB | 4.55GB |
|  | Transformer-INT4 | 0.987s | 6.438s | 1.99GB | 3.91GB |
|  | Small VAE Decoder-FP16 | 0.062s | 1.675s | -0.36GB | 1.55GB |

## Notes

- **RAM After Load / Peak RAM** are deltas relative to the configuration's pipeline start RSS (single shared baseline per configuration). See `flux2_klein_memory_details.csv` for absolute baseline/after-load/peak/release values.
- **CFG is disabled**: guidance_scale=1.0, negative_prompt_embeds=None (single transformer pass per step).
- int4 512x512: image saved to benchmark_outputs_ov_matrix\ov_26_3\images\flux2_klein_int4_512x512_4steps_no_cfg.png
- int4 1024x1024: image saved to benchmark_outputs_ov_matrix\ov_26_3\images\flux2_klein_int4_1024x1024_4steps_no_cfg.png
