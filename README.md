# ComfyUI NInfer Qwen3.8-27B

This custom node uses a local [NInfer](https://github.com/Don-Chad/ninfer-3090)
server to improve image-generation prompts with Qwen3.8-27B. It returns the
result as a ComfyUI `STRING`, accepts an optional native ComfyUI `IMAGE`, and
stops the NInfer process after each request so its VRAM is released.

The tested setup is Windows 11 with an RTX 3090. The node does not call a
remote API, download models, or accept arbitrary image URLs or file paths.

## Installation

For RTX 40-series, RTX 50-series, Linux, and other higher-VRAM setups, see the
[higher-tier GPU installation guide](GPU_INSTALLATION_GUIDE.md) before choosing
an NInfer runtime.

This is the same layout used by the working setup:

1. Copy or clone this repository to:

   ```text
   C:\~\custom_nodes\ComfyUI-amp-ninfer
   ```

2. Install the node dependencies with the Python environment used by ComfyUI:

   ```powershell
   C:\~\venv\Scripts\python.exe -m pip install -r C:\~\custom_nodes\ComfyUI-amp-ninfer\requirements.txt
   ```

   For another ComfyUI installation, replace those paths with its Python
   executable and this repository's `requirements.txt`. ComfyUI supplies
   PyTorch and CUDA; this file must not be used to replace that installation.

3. Download and extract the [NInfer-3090 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.6.0-rtx3090),
   for example under:

   ```text
   C:\ninfer\ninfer-rtx3090-windows-x64-0.6.0-rtx3090\
   ```

4. Obtain `qwen3_8_27b.ninfer` from the [Qwen3.8-27B NInfer model card](https://huggingface.co/neroued/Qwen3.8-27B-NInfer).
   Keep the local model outside the repository, for example:

   ```text
   C:\models\qwen3_8_27b.ninfer
   ```

5. Restart ComfyUI and search for **NInfer Qwen3.8-27B**. Set these node
   fields to the full paths on your machine:

   - `ninfer_executable`: `ninfer-serve.exe`
   - `model_artifact`: `qwen3_8_27b.ninfer`
   - `model_id`: `qwen3.8-27b`

## Use

Place the node before the prompt encoder in a workflow:

```text
NInfer Qwen3.8-27B -> text prompt encoder -> image generation
```

Enable `vision` only when connecting an `IMAGE` input. Keep the default
context/KV settings on a 24 GB card, and avoid keeping another large CUDA
model resident while NInfer starts.

For fixed seeds, the node reuses the generated text. With the default seed
`-1`, each execution makes a fresh request.

## RTX 30-series tuning

The linked Windows runtime is compiled for Ampere `sm_86` and is the tested
path for an RTX 3090. It is not a generic NInfer binary: do not use it with an
RTX 40-series or RTX 50-series GPU. Use a runtime built for the target
architecture instead.

Start with this profile on a 24 GB RTX 3090:

```text
context_size: 4096
kv_capacity: 4096
speculative_backend: mtp
draft_tokens: 3
vision: false
no_cuda_graph: true
unload_comfyui_before_launch: true
unload_after_request: true
```

The node also starts NInfer with `--max-concurrency 1`,
`--max-pending-requests 1`, `--prefill-chunk 512`, and `--kv-dtype int8`.
Leave `server_launch_flags` empty until this profile works. Do not repeat
typed options such as `--max-context`, `--kv-capacity`, `--spec`, `--vision`,
or `--no-cuda-graph` there.

If startup runs out of memory, reduce `context_size` and `kv_capacity` together,
then set `speculative_backend` to `off` and keep `vision` disabled. Keep
ComfyUI's large models unloaded while NInfer starts. Cards below 24 GB need
separate validation; the artifact file itself is about 16.96 GiB before runtime
and cache allocations.

## Recommended links

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) — host application.
- [NInfer-3090 Windows guide](https://github.com/Don-Chad/ninfer-3090/blob/release/v0.6.0-rtx3090/docs/rtx-3090-windows.md) — runtime and RTX 3090 setup.
- [NInfer serving guide](https://github.com/Don-Chad/ninfer-3090/blob/release/v0.6.0-rtx3090/docs/serving.md) — HTTP API and server options.
- [Qwen3.8-27B NInfer model card](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) — artifact download, checksum, and model details.

## Tests

The test suite uses local fakes and does not require an NInfer binary or GPU:

```powershell
C:\~\venv\Scripts\python.exe -m pytest -q
```
