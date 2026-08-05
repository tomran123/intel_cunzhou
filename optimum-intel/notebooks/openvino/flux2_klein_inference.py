"""
Flux.2-klein OpenVINO — 3 models, 3 states each, one resident at a time.

Models : text_encoder -> transformer -> vae_decoder
States : LOAD (compile) -> INFERENCE -> RELEASE (deep: GPU + host RAM)

- LOAD    = component.compile()           (core.compile_model)
- RELEASE = request=None (GPU) + model=None (host RAM) + gc  (combined "deep")

Only one heavy model is compiled at any moment. Load & inference time is printed
per model. The transformer/vae stages live inside pipe.__call__, so their timing
is captured via the step-end callback (vae is compiled right after the last
denoise step, just before the post-loop decode).
"""

from pathlib import Path
import gc
import time

import numpy as np
from optimum.intel import OVDiffusionPipeline

model_dir = Path("./benchmark_outputs/models/flux2-klein-int4")

#model_dir = Path("./benchmark_outputs/models/flux2-klein-fp16")
device = "GPU"

prompt = "Astronaut on the moon"
height = 1024
width = 1024
num_inference_steps = 4
max_sequence_length = 512

# CFG on  -> 2 transformer passes/step (pos + neg), needs negative embeds.
# CFG off -> 1 pass/step (~half denoise time), no negative embeds.
USE_CFG = False
guidance_scale = 4.0 if USE_CFG else 1.0

np.random.seed(0)

timings = {}       # name -> {"load": s, "infer": s}
last_step = num_inference_steps - 1


def compile_named_component(pipeline, name):
    """Compile a pipeline part by name using the current Optimum API."""
    component = getattr(pipeline, name)
    component.compile()


def load(name):
    """LOAD state: compile the submodel (host weights already read at from_pretrained)."""
    t = time.perf_counter()
    compile_named_component(pipe, name)
    dt = time.perf_counter() - t
    timings.setdefault(name, {})["load"] = dt
    print(f"[LOAD]    {name:<13} {dt:8.3f}s")


def release(name):
    """RELEASE state (deep): free compiled request (GPU) + ov.Model weights (host RAM)."""
    part = getattr(pipe, name)
    part.request = None
    part.model = None
    gc.collect()
    print(f"[RELEASE] {name:<13} (GPU + host RAM freed)")


# ---- init: read models into host RAM, compile NOTHING yet ----
pipe = OVDiffusionPipeline.from_pretrained(model_dir, compile=False, cache_dir="./ov_model_cache")
pipe.to(device)

# ============================ 1) text_encoder ============================
load("text_encoder")

t = time.perf_counter()
prompt_embeds, _ = pipe.encode_prompt(
    prompt=prompt, num_images_per_prompt=1, max_sequence_length=max_sequence_length
)
# With CFG on, __call__ also encodes the negative ("") prompt -> precompute it
# now so text_encoder is not needed after release. With CFG off, skip it.
negative_prompt_embeds = None
if USE_CFG:
    negative_prompt_embeds, _ = pipe.encode_prompt(
        prompt="", num_images_per_prompt=1, max_sequence_length=max_sequence_length
    )
timings["text_encoder"]["infer"] = time.perf_counter() - t
print(f"[INFER]   text_encoder  {timings['text_encoder']['infer']:8.3f}s  "
      f"embeds={tuple(prompt_embeds.shape)}")

release("text_encoder")

# ============================ 2) transformer + 3) vae_decoder ============================
load("transformer")

state = {}


def on_step_end(pipeline, step, timestep, callback_kwargs):
    if step == last_step:
        # transformer INFERENCE done
        state["denoise_end"] = time.perf_counter()
        print(f"[INFER]   {'transformer':<13} {state['denoise_end'] - denoise_start:8.3f}s")
        # transformer RELEASE (deep)
        pipeline.transformer.request = None
        pipeline.transformer.model = None
        gc.collect()
        print(f"[RELEASE] {'transformer':<13} (GPU + host RAM freed)")
        # vae_decoder LOAD (compile), timed
        t0 = time.perf_counter()
        compile_named_component(pipeline, "vae_decoder")
        timings.setdefault("vae_decoder", {})["load"] = time.perf_counter() - t0
        print(f"[LOAD]    {'vae_decoder':<13} {timings['vae_decoder']['load']:8.3f}s")
        state["vae_ready"] = time.perf_counter()
    return callback_kwargs


pipe.set_progress_bar_config(disable=True)
denoise_start = time.perf_counter()
out = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    output_type="pil",
    callback_on_step_end=on_step_end,
)
pipe_end = time.perf_counter()

# transformer INFERENCE = denoise loop (already printed inside callback)
timings["transformer"]["infer"] = state["denoise_end"] - denoise_start

# vae_decoder INFERENCE = decode (post-loop, inside __call__)
timings["vae_decoder"]["infer"] = pipe_end - state["vae_ready"]
print(f"[INFER]   vae_decoder   {timings['vae_decoder']['infer']:8.3f}s")

# vae RELEASE (deep)
release("vae_decoder")
if pipe.vae_encoder is not None:
    release("vae_encoder")

# ---- save ----
img = out.images[0]
img.save("flux2_klein_smoke_1024.png")
print("\nSaved:", Path("flux2_klein_smoke_1024.png").resolve(), "size:", img.size)

# ---- summary ----
print("\n===== timing summary (s) =====")
print(f"{'model':<14}{'load':>10}{'infer':>10}")
for name in ("text_encoder", "transformer", "vae_decoder"):
    t = timings.get(name, {})
    print(f"{name:<14}{t.get('load', 0):>10.3f}{t.get('infer', 0):>10.3f}")
