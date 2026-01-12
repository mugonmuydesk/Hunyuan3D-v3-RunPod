# Hunyuan3D-v3 RunPod Project

## Overview

RunPod serverless endpoint for Tencent's Hunyuan3D-2.1 image-to-3D generation with PBR textures.

- **Endpoint ID**: `hjeptaeuf9o5kj`
- **Docker Image**: `mugonmuydesk/hunyuan3d-v3:latest`
- **GPU Requirements**: 48GB recommended (shape: ~10GB, texture: ~21GB)

## Local Testing

**Requires nvidia-container-toolkit for Docker GPU support.**

Install (one-time):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Run locally:
```bash
docker run --gpus all -p 8000:8000 mugonmuydesk/hunyuan3d-v3:latest

# Test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

## Build Tiers

| Dockerfile | Purpose | Duration | Catches |
|------------|---------|----------|---------|
| `Dockerfile.dryrun` | CI fast validation | ~10 min | Import errors, syntax |
| `Dockerfile.dryrun-full` | Full deps validation | ~20 min | Runtime errors, version conflicts |
| `Dockerfile` | Production with models | ~30+ min | Everything |

Build locally in WSL2 (native Linux fs for speed):
```bash
cp -r /mnt/c/dev/hunyuan3d-v3-runpod ~/hunyuan3d-v3-runpod
cd ~/hunyuan3d-v3-runpod
docker build -t hunyuan3d-v3:test -f Dockerfile.dryrun-full .
```

## Key Fixes Applied

### 1. Numpy Flags Access (handler.py:72)
```python
# Wrong: ndarray.flags.get('C_CONTIGUOUS', True)
# Fixed: ndarray.flags['C_CONTIGUOUS']
```
Numpy's flags object uses bracket access, not `.get()` method.

### 2. Numpy Version Conflicts (Dockerfile.dryrun-full)
```dockerfile
# Wrong: pip install --ignore-installed (causes dual numpy versions)
# Fixed: pip uninstall -y numpy && pip install numpy==1.26.4
```
Prevents RecursionError from conflicting numpy installations.

### 3. ONNX Runtime CUDA 12 (requirements_inference.txt)
```
# Wrong: onnxruntime-gpu==1.17.0 (CUDA 11 only)
# Fixed: onnxruntime-gpu==1.19.2 (CUDA 12 support)
```
Base image uses CUDA 12.4, need matching ONNX runtime.

### 4. Deep Import Tests (test_handler_mock.py)
Added comprehensive tests that catch runtime issues during build:
- Import chain validation (pytorch_lightning, etc.)
- Numpy patch verification
- ONNX provider availability (catches CUDA lib mismatches)

## File Structure

```
handler.py              # RunPod serverless handler with numpy patches
test_handler_mock.py    # Comprehensive dry-run tests
requirements_inference.txt  # Production dependencies
requirements_ci.txt     # Minimal CI dependencies
Dockerfile              # Production build (with models)
Dockerfile.dryrun       # Fast CI validation
Dockerfile.dryrun-full  # Full deps validation
custom_rasterizer-*.whl # Prebuilt CUDA extension (Python 3.12)
```

## Client Usage

```bash
# From C:\dev\runpod
RUNPOD_API_KEY=<key> RUNPOD_ENDPOINT_ID=hjeptaeuf9o5kj \
  python hunyuan3d_client.py image.png -o output.glb

# Mesh only (faster, ~10GB VRAM)
python hunyuan3d_client.py image.png --no-texture -o output.glb
```

## Deployment

Push changes via GitHub release - RunPod auto-rebuilds on new releases.

```bash
git add -A && git commit -m "Fix: description"
git tag v1.x.x && git push && git push --tags
# Create release on GitHub to trigger RunPod rebuild
```

## Debugging Pattern

1. Make fix locally
2. Build `Dockerfile.dryrun-full` to validate
3. If passes, push and create release
4. Monitor RunPod build logs
5. Test endpoint with client

**Key insight**: Dry-run catches import-time errors but not runtime library loading. The ONNX provider test in `test_handler_mock.py` catches CUDA library mismatches during build.
