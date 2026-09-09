# ComfyUI-DirectNinfer

ComfyUI custom nodes that use a local [NInfer](https://github.com/Don-Chad/ninfer-3090) server to improve image-generation prompts. The server is started for a request and stopped afterwards so its CUDA allocations are released. Optional ComfyUI images can be supplied as visual context.

Windows and Linux are supported. NInfer binaries are GPU-architecture specific; the tested profile is an RTX 3090 (24 GB, `sm_86`).

## Install

From the `custom_nodes` directory of your ComfyUI installation:

```console
git clone https://github.com/cicalooo/ComfyUI-DirectNinfer.git
cd ComfyUI-DirectNinfer
python -m pip install -r requirements.txt
```

Use the same Python environment that starts ComfyUI, then restart ComfyUI. The package does not install PyTorch or CUDA.

## Configure NInfer

The node does not bundle a runtime or model. Install a NInfer build for the GPU and operating system, then download a compatible `.ninfer` model.

| Platform | Executable example |
| --- | --- |
| Windows | `C:\ninfer\ninfer-serve.exe` |
| Linux | `/opt/ninfer-3090/current/ninfer-serve` |

Add **DirectNinfer** to a workflow and set `ninfer_executable` to the absolute executable path, `models_dir` to the directory containing `.ninfer` files, and `model_artifact` to a selected model. Click **Refresh** after changing the directory.

The executable can also be supplied through `NINFER_EXECUTABLE`. If it is not set, Windows uses `ninfer-serve.exe`; Linux uses `ninfer-serve` (or the tested `/opt/ninfer-3090/current/ninfer-serve` when present). The model ID is read from the server's `/v1/models` response.

## Nodes

### DirectNinfer

Rewrites `user_prompt` according to `system_prompt` and returns a `STRING`. Connect `image_list` or the expanding `IMAGE` inputs for visual context. Images are converted to small RGB PNG data URLs before they are sent to NInfer.

### DirectNinfer Advanced

Provides optional sampling, server, speculative-decoding, launch-flag, and timeout settings. Connect its `advanced` output to the main node; leaving it disconnected uses the built-in defaults.

## GPU and runtime notes

Use a runtime built for the target GPU architecture. For RTX 3090, see the [NInfer-3090 releases](https://github.com/Don-Chad/ninfer-3090/releases) and the [GPU installation guide](GPU_INSTALLATION_GUIDE.md). Keep runtime and model files outside this repository.

On a 24 GB GPU, start with `no_cuda_graph`, `unload_comfyui_before_launch`, and `unload_after_request` enabled. If startup runs out of memory, reduce `context_size` and `kv_capacity` together and disable speculation.

## Tests

The test suite uses local fakes and does not require a GPU, model, or NInfer:

```bash
python -m pytest -q
```

Hardware tests are opt-in.

## License

Apache-2.0. See [LICENSE](LICENSE).
