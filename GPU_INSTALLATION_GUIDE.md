# Higher-Tier GPU Installation Guide

This node does not install NInfer or choose a GPU backend. It starts the
`ninfer-serve` executable configured in the node, so the executable must be
built for the GPU architecture on which ComfyUI runs. The Python dependencies
remain the same for every GPU; do not install a second PyTorch or CUDA build
through this repository.

The main README documents the tested Windows RTX 3090 setup. The paths below
are for higher-tier cards and are not covered by this custom node's automated
GPU tests.

## Choose a matching runtime

| GPU family | CUDA architecture | Runtime direction | Status |
|---|---:|---|---|
| RTX 3090 | `sm_86` | NInfer-3090 Windows release | Tested path in the main README |
| RTX 4090 | `sm_89` | An NInfer build explicitly targeting RTX 4090 | Community port; use its own build/release notes |
| RTX 4080/4070/4060 | `sm_89` | Same `sm_89` requirement, if the build supports the card | Not qualified here; available VRAM may be insufficient |
| RTX 5090 | `sm_120a` | Upstream Linux build or a native Windows port | NInfer reference hardware; 32 GB is recommended |
| Other RTX 50-series cards | `sm_120a` or a related Blackwell target | Only use a binary that lists the exact card | Not assumed to work from the series name alone |

Do not use the RTX 3090 binary on an RTX 4090 or RTX 5090. A successful CUDA
installation does not make a binary compiled for another architecture usable.
Likewise, an RTX 4090 build is not automatically valid for every RTX 40-series
card.

The `qwen3_8_27b.ninfer` artifact is about 16.96 GiB before the runtime,
context/KV cache, CUDA Graph buffers, and optional vision/speculation memory are
allocated. On a 24 GB card the node defaults to 16384 context and KV; reduce
both in steps of 1024 if startup runs out of memory. Cards with 16 GB should
not be assumed to fit this artifact.

## Install the common node files

Complete steps 1 and 2 in the main [README](README.md) first. Keep the node in
the Python environment used by ComfyUI. The external NInfer runtime can live
in a separate folder, for example:

```text
C:\ninfer\
  runtime\
  models\qwen3_8_27b.ninfer
```

ComfyUI and NInfer must run in the same operating-system environment. A Linux
binary inside WSL is usable by ComfyUI running in WSL/Linux, not by a native
Windows ComfyUI process.

## Download and verify the model artifact

Use the [Qwen3.8-27B NInfer model card](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
to download `qwen3_8_27b.ninfer`. The artifact used by this node has this
SHA-256:

```text
eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e
```

With the Hugging Face CLI, an explicitly initiated download looks like this:

```powershell
hf download neroued/Qwen3.8-27B-NInfer `
  qwen3_8_27b.ninfer `
  --local-dir C:\models
```

Verify it in PowerShell before starting the server:

```powershell
(Get-FileHash C:\models\qwen3_8_27b.ninfer -Algorithm SHA256).Hash.ToLowerInvariant()
```

The printed value must exactly match the checksum above. Do not substitute a
Transformers, Safetensors, or GGUF file; NInfer accepts its own `.ninfer`
artifact format.

## RTX 4090 / 40-series

The RTX 4090 uses `sm_89`. The RTX 3090 release linked from the main README is
an `sm_86` build and is not the correct runtime for a 4090. Use a current
Qwen3.8-compatible `sm_89` build, such as the community
[NInfer-4090 port](https://github.com/UDPSendToFailed/ninfer-4090), and follow
that project's Windows prerequisites and build instructions. It is a separate
project, not a release of this custom node.

For a source build, the important part is that the CUDA architecture is
explicitly set to 89 and that the resulting executable is the server app:

```powershell
git clone https://github.com/UDPSendToFailed/ninfer-4090.git C:\ninfer\ninfer-4090
Set-Location C:\ninfer\ninfer-4090
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-ninja --parallel
```

The exact output path depends on the fork and generator. Locate
`ninfer-serve.exe`, then verify that it starts and advertises the options used
by this node:

```powershell
& C:\ninfer\ninfer-4090\build-ninja\apps\ninfer-serve.exe --help
```

If the fork's README specifies a different output directory or dependency
setup, use its instructions; the architecture and artifact requirements still
apply.

For a 24 GB RTX 4090, start with these node values:

```text
context_size: 4096
kv_capacity: 4096
speculative_backend: mtp
draft_tokens: 3
vision: false
no_cuda_graph: true
```

After a successful text request, try `8192` for both `context_size` and
`kv_capacity`. Enable `vision` only when an `IMAGE` is connected, because it
loads additional model state. An RTX 4080/4070/4060 may use the same
architecture-specific runtime, but its lower VRAM requires separate testing;
do not infer support from the 4090 results.

## RTX 5090 / 50-series

The NInfer reference runtime is specialized for a single RTX 5090 (`sm_120a`),
64-bit Linux, and CUDA Toolkit 13.1 or newer. The upstream
[NInfer repository](https://github.com/Neroued/ninfer) has the source build.
For the Qwen3.8 artifact used here, use revision
[`5232055`](https://github.com/Neroued/ninfer/tree/52320554b5e71a9da96bff809ddf67ac5773ed63)
or a later revision that still documents Qwen3.8 support.

### Linux or WSL/Linux source build

Install a CUDA 13.1-compatible NVIDIA driver and toolkit, a C++20 compiler,
CMake 3.28 or newer, Ninja, `pkg-config`, FFmpeg development libraries, and
libcurl. On Ubuntu, the common system packages are:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev git
```

Build the matching runtime:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer
git checkout 52320554b5e71a9da96bff809ddf67ac5773ed63
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The server executable is:

```text
build/apps/ninfer-serve
```

Run `./build/apps/ninfer-serve --help` before connecting it to ComfyUI. If
ComfyUI is running in WSL/Linux, set the node's `ninfer_executable` and
`model_artifact` to Linux paths. Do not paste Windows paths into a Linux node
process.

### Native Windows 5090 build

For native Windows, use a Windows runtime that explicitly supports RTX 5090
and `sm_120a`, such as the [NInfer-windows port](https://github.com/natpate/ninfer-windows).
Its portable release is the simplest option: download the CUDA 13.1 Windows
zip, extract it, and confirm that `ninfer-serve.exe --help` runs before using
the executable in the node. A source build requires Visual Studio 2022 C++
tools, CUDA 13.1 or newer, CMake, and the dependencies listed by that project.

Do not use a 5090-only binary on a 4090. Do not assume that an RTX 5080, 5070,
or another 50-series card is supported just because it is Blackwell; confirm
the runtime's exact architecture and memory requirements first.

For a 32 GB RTX 5090, begin with:

```text
context_size: 16384
kv_capacity: 16384
speculative_backend: mtp
draft_tokens: 3
vision: false
no_cuda_graph: true
```

Increase context and KV capacity gradually only after a successful request.
The node requires `kv_capacity` to be at least `context_size`, and the
allocation is fixed when NInfer starts. Keep ComfyUI models unloaded while
testing and leave VRAM headroom for the desktop and any enabled vision or
speculative-decoding buffers.

## Configure the node

Set these fields to full paths. The executable and artifact may be in separate
folders:

| Node field | Windows example | Linux example |
|---|---|---|
| `ninfer_executable` | `C:\ninfer\runtime\ninfer-serve.exe` | `/opt/ninfer/build/apps/ninfer-serve` |
| `models_dir` | `C:\models` | `/opt/ninfer/models` |
| `model_artifact` | `qwen3_8_27b.ninfer` (dropdown) | `qwen3_8_27b.ninfer` (dropdown) |
| `device` | `0` (Amp NInfer Advanced) | `0` (Amp NInfer Advanced) |

Use `nvidia-smi -L` to identify the GPU index. This node launches one NInfer
process on one CUDA device; it does not combine VRAM across multiple GPUs.

Keep `unload_comfyui_before_launch` enabled while testing. The node validates
the server's `--help` output before launch, so an `unknown option` error means
the selected NInfer build does not implement the option contract expected by
this node. Use a matching build rather than adding duplicate typed flags to
`server_launch_flags`.

## Troubleshooting

**`no kernel image is available for execution on the device`** — the runtime
was built for another compute capability. Replace it with an `sm_89` build for
an RTX 4090 or an `sm_120a` build for an RTX 5090.

**Out of memory during startup** — reduce `context_size` and `kv_capacity`
together, keep `vision` disabled, set `speculative_backend` to `off`, and
close other CUDA applications. Re-enable features one at a time.

**The server starts but the node cannot connect** — confirm that the node and
server use the same `host`/`port`, that the port is free, and that the
executable can run with `--help` from the same user account as ComfyUI.

**The model is rejected** — verify the SHA-256 and select the exact
`.ninfer` artifact. Amp NInfer uses the id advertised on `/v1/models`.

## References

- [NInfer upstream requirements and build](https://github.com/Neroued/ninfer)
- [NInfer-3090 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.6.0-rtx3090)
- [NInfer-4090 `sm_89` port](https://github.com/UDPSendToFailed/ninfer-4090)
- [NInfer-windows RTX 5090 port](https://github.com/natpate/ninfer-windows)
- [Qwen3.8-27B NInfer artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
