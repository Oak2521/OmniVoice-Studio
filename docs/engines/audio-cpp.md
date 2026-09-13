# VoiceStudio — audio.cpp Engine (Breeze-TTS-2)

[audio.cpp](https://github.com/0xShug0/audio.cpp) is a pure-C++ ggml audio
inference framework with prebuilt binaries for Windows, macOS, and Linux and
no Python dependency. VoiceStudio discovers the compute providers compiled
into the installed binary and selects the best available device. VoiceStudio drives its
`audiocpp_server` over loopback HTTP — v1 serves the **`breeze_tts`**
family: **Breeze-TTS-2** (BreezeBlue, 3B params, English + Chinese, voice
clone + voice design + voice direction, 24 kHz).

> **Opt-in, and never a default.** Select `audiocpp` explicitly in **Model
> Catalogue → Engines** (or `OMNIVOICE_TTS_BACKEND=audiocpp`).

## License — read before enabling

- **audio.cpp code:** Apache-2.0.
- **Breeze-TTS-2 weights** (upstream `BreezeBlue/Breeze-TTS-2` and the
  `audio-cpp/audio.cpp-gguf` GGUF repack): **research and non-commercial
  use only** under the [BreezeBlue Research and Non-Commercial License](
  https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
  Self-hosted outputs inherit the restriction; a BreezeBlue paid
  subscription covers hosted-platform outputs only, not this engine.

## Platform support

| Host | Binary | Compute |
|---|---|---|
| Windows x64 | CPU, Vulkan, CUDA 12.4/13.3 prebuilts | CPU, Vulkan, CUDA |
| Linux x64 | CPU and Vulkan prebuilts | CPU, Vulkan; CUDA/ROCm from a self-build |
| macOS arm64 | upstream Metal prebuilt | Metal + CPU |
| macOS x64 | upstream `metal` archive (Metal disabled by upstream) | CPU |
| Linux aarch64 | none upstream | unavailable in v1 |

VoiceStudio runs `audiocpp_server --list-devices` once per installed binary,
prefers CUDA, ROCm, Metal, then a discrete Vulkan GPU, and keeps CPU as a safe
fallback. Device numbers are local to each backend registry. The Q8_0 GGUF is
approximately 4.73 GiB, plus runtime memory.

Allow about 6 GB of dedicated VRAM for CUDA, HIP, or a discrete GPU exposed
through Vulkan. Metal and Vulkan integrated GPUs use unified-memory handling
instead of that dedicated-VRAM floor.

## Install

1. Download the v0.7.2 prebuilt for your platform from
   [audio.cpp releases](https://github.com/0xShug0/audio.cpp/releases/tag/v0.7.2)
   and extract it. Use the Vulkan archive on Windows or Linux for broad GPU
   support, the CPU archive when Vulkan is unavailable, or the matching CUDA
   archive on Windows for NVIDIA. A Windows CUDA install needs both the
   `bin-…-cuda…` and matching `cudart-…-cuda…` archives extracted into the
   same directory, as required by upstream. VoiceStudio does not download
   executable code for this engine. Linux archives do not preserve the
   executable bit, so run `chmod +x audiocpp_server` after extracting one.

   Verify the archive before extracting it. The pinned SHA-256 checksums are:

   | Archive | SHA-256 |
   |---|---|
   | `audio-v0.7.2-bin-windows-x64-cpu-portable.zip` | `0b1f4bd78c5226ee3fa0eb24d95d603a429439cdf5dab45872d44a87412dd8c1` |
   | `audio-v0.7.2-bin-windows-x64-vulkan.zip` | `15b8232eae740e21e507d87f827a89966de9451b085a45932d9e214e032962c1` |
   | `audio-v0.7.2-bin-windows-x64-cuda12.4.zip` | `06c426095008022a2984ff1c75de4c9fab463c4201c0ff0a5dc4e14043f52326` |
   | `audio-v0.7.2-cudart-windows-x64-cuda12.4.zip` | `7115be4d462817ad293f7932a8ac436d51023128e6728af09bba92a85593f393` |
   | `audio-v0.7.2-bin-windows-x64-cuda13.3.zip` | `f975fec52745807b8c787e826c110acc3424a45156092455e1635604b68ec832` |
   | `audio-v0.7.2-cudart-windows-x64-cuda13.3.zip` | `9b508f702636a9cdf3bf4dd8e75a86c20a0b87bdc39ca07e714c82f748efc1fa` |
   | `audio-v0.7.2-bin-ubuntu-x64-cpu.tar.gz` | `6f5e43dd7b80e8ddf688ef84b411fadcd1f934d2c83963178bc4e2d9c4f07736` |
   | `audio-v0.7.2-bin-ubuntu-x64-vulkan.tar.gz` | `fee1f978cee76453cf17f00196554bc2ee294645739538af0726a143b6a69a23` |
   | `audio-v0.7.2-bin-macos-arm64-metal.tar.gz` | `c01e4f82971bedbe341697e63a9cebd5a5d1f72d5a9bcb51a3191f95ddab7a95` |
   | `audio-v0.7.2-bin-macos-x64-metal.tar.gz` | `3862270f33439077225324169313f727064f727305b54d8ce920244d75ddcc24` |

   Run `sha256sum <archive>` on Linux, `shasum -a 256 <archive>` on macOS,
   or `Get-FileHash <archive> -Algorithm SHA256` in PowerShell and compare the
   complete result with the table.
2. Set `OMNIVOICE_AUDIOCPP_BIN` to the `audiocpp_server` binary
   (`audiocpp_server.exe` on Windows):

   ```bash
   # macOS / Linux
   echo 'export OMNIVOICE_AUDIOCPP_BIN=$HOME/apps/audio.cpp/audiocpp_server' >> ~/.zshrc
   source ~/.zshrc
   ```

   Alternatively set `OMNIVOICE_AUDIOCPP_DIR` to the directory containing it.
3. Restart VoiceStudio, open **Model Catalogue (TTS tab → the engine's Weights)**, find
   **Breeze-TTS-2 Q8_0 for audio.cpp**, review its research/non-commercial
   license note, and click **Install**. Generation never starts this ~4.73 GiB
   download automatically.
4. Pick `audiocpp` in **Model Catalogue** (TTS tab → **Use**). The server starts
   lazily on first generate (`server.json` + `server.log` live under the app
   data `audiocpp/` directory).

## Voice modes

All three go through the one speech endpoint — reference presence selects:

- **Clone:** `ref_audio` + `ref_text` (exact transcript, as upstream).
- **Direction:** `ref_audio` + `ref_text` + `instruct`
  (e.g. "Speak slowly with a restrained, serious tone").
- **Design:** `description` (or `instruct`) with no `ref_audio`
  (e.g. "A warm, thoughtful young woman…"). Upstream strengthens
  instruction-following with `guidance_scale` ≈ 4.

## Optional env knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `OMNIVOICE_AUDIOCPP_BIN` | — | Absolute path to `audiocpp_server`. |
| `OMNIVOICE_AUDIOCPP_DIR` | — | Directory containing `audiocpp_server`. |
| `OMNIVOICE_AUDIOCPP_MODEL` | Model Catalogue cache | GGUF file or directory override. |
| `OMNIVOICE_AUDIOCPP_PACKAGE` | `breeze-tts-2-q8_0.gguf` | Package filename (`…-bf16.gguf` for full precision). |
| `OMNIVOICE_AUDIOCPP_PORT` | `17860` | Loopback port. |
| `OMNIVOICE_AUDIOCPP_BACKEND` | Settings, then auto | Exact runtime: `cuda`, `hip`/`rocm`, `vulkan`, `metal`, or `cpu`. |
| `OMNIVOICE_AUDIOCPP_DEVICE` | best device | Backend-local non-negative device index; requires `OMNIVOICE_AUDIOCPP_BACKEND`. |

The audio.cpp overrides take precedence over the global Settings compute
choice. A global CUDA/ROCm choice can match an NVIDIA/AMD GPU exposed through
Vulkan. An unavailable explicit audio.cpp override is an error; an unavailable
global preference falls back to CPU and is shown as a routing fallback.

## Common errors

### `audiocpp_server not found ...`

The binary isn't installed. Follow **Install** — the message carries the
exact release URL and SHA for your platform.

### `audiocpp_server exited during startup ...`

The managed loopback port may be taken. Check `server.log` next to
`server.json` in the app data `audiocpp/` directory, or set a different
`OMNIVOICE_AUDIOCPP_PORT` and restart VoiceStudio.

### `Breeze-TTS-2 ... not installed` or `package ... not completely installed`

Install the model from **Model Catalogue (TTS tab → the engine's Weights)**. If an interrupted install
left it incomplete, use **Reinstall** there. If the error persists after a
complete reinstall, file an issue with the package listing.

---

audio.cpp runs as a managed native server (no Python venv, no
`transformers` conflict). Only the downloaded GGUF counts toward
[sidecar disk usage](disk-usage.md).
