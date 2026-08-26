# NInfer model artifact

No model weights are stored in this repository. Download the tested
`qwen3_8_27b.ninfer` artifact from
[neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer)
and keep it outside the custom-node directory.

| File | Size | SHA-256 |
|---|---:|---|
| `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |

Verify the downloaded file before use. Set `models_dir` to its parent
directory in **Amp NInfer**, click **Refresh**, and select the file from the
`model_artifact` dropdown.

See the [GPU installation guide](../GPU_INSTALLATION_GUIDE.md) for download,
verification, and GPU-specific runtime instructions. The Hugging Face model
card is the source of truth for artifact format, provenance, evaluation, and
licensing details.
