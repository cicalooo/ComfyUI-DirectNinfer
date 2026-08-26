# Amp NInfer for ComfyUI

Amp NInfer uses a local [NInfer](https://github.com/Don-Chad/ninfer-3090)
server to rewrite image-generation prompts and returns the result as a ComfyUI
`STRING`. It can use native ComfyUI images as visual context and shuts down its
NInfer process after each request to release VRAM.

The tested path is Windows 11 with a 24 GB RTX 3090. NInfer executables are
GPU-architecture specific; see the
[GPU installation guide](GPU_INSTALLATION_GUIDE.md) for RTX 40-series,
RTX 50-series, Linux, and source-build guidance.

## Install

Clone the node into the `custom_nodes` directory of your ComfyUI installation:

```console
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/cicalooo/ComfyUI-amp-ninfer.git
cd ComfyUI-amp-ninfer
python -m pip install -r requirements.txt
```

Run the last command with the same Python environment that starts ComfyUI
(including ComfyUI's embedded Python when applicable), then restart ComfyUI.
ComfyUI supplies PyTorch and CUDA; this repository deliberately does not
install another Torch build.

## Install NInfer and a model

The node does not bundle or download an NInfer runtime or model.

1. For the tested RTX 3090 path, download and extract the
   [NInfer-3090 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.6.0-rtx3090).
   Keep the runtime outside this repository.
2. Download a compatible `.ninfer` model. The tested Qwen3.8-27B artifact and
   its SHA-256 are linked in [artifacts/README.md](artifacts/README.md). Keep
   model files outside this repository as well.
3. Add **Amp NInfer** to a workflow and set:

   - `ninfer_executable` to the full absolute path of `ninfer-serve.exe` (or
     `ninfer-serve` on Linux);
   - `models_dir` to the directory containing the `.ninfer` files;
   - `model_artifact` from the dropdown. Click **Refresh** after changing
     `models_dir` or adding a model.

There is no user-facing `model_id` setting. The node derives a launch ID from
the artifact name and uses the ID advertised by the server at `/v1/models`.

## Nodes

### Amp NInfer

This is the main prompt-rewriting node. `system_prompt` supplies the rewrite
instructions and `user_prompt` supplies the prompt to improve; they are the
only prompt fields. The output connects to a text encoder or any other ComfyUI
input that accepts a `STRING`.

The default `context_size` and `kv_capacity` are both `16384`, with steps of
`1024`. A fixed non-negative seed reuses the generated text; the default seed
of `-1` makes a fresh request on each execution.

To use visual context, enable `vision` and connect an image to `image_1`. Only
that socket is shown at first. Connecting it adds `image_2`, and so on, up to
`image_20`. Disconnecting trailing images removes unused slots. These are
native ComfyUI `IMAGE` inputs; the node does not accept arbitrary image URLs or
file paths.

### Amp NInfer Advanced

This optional companion node groups sampling, reasoning, server, speculative
decoding, launch-flag, and timeout settings. Connect its `advanced` output to
the optional `advanced` input on **Amp NInfer**. Leave it disconnected to use
the built-in defaults.

## RTX 3090 notes

The linked Windows runtime targets Ampere `sm_86` and is the tested path. Do
not use it on an RTX 40-series or RTX 50-series GPU; use a runtime built for
the target architecture.

On the tested 24 GB profile, leave `no_cuda_graph`,
`unload_comfyui_before_launch`, and `unload_after_request` enabled initially.
If NInfer runs out of memory during startup, reduce `context_size` and
`kv_capacity` together in `1024`-token steps, disable speculation through
**Amp NInfer Advanced**, and leave `vision` off until text-only startup works.
Avoid keeping another large CUDA model resident while NInfer starts.

For runtime selection, model verification, and troubleshooting, use the
[GPU installation guide](GPU_INSTALLATION_GUIDE.md).

## Tests

The tests use local fakes and do not require a GPU, model, or NInfer binary.
From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -q
```

Hardware integration tests remain opt-in.

## License

Apache-2.0. See [LICENSE](LICENSE).
