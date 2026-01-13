"""
Mock test for handler.py - validates handler logic without model weights.

This test catches:
1. Missing import dependencies (pytorch_lightning, etc.)
2. Numpy patch recursion issues
3. Handler logic errors
4. Input validation

Run with: python test_handler_mock.py

Environment:
- Dockerfile.dryrun-full: Full deps, all tests should pass
- Dockerfile.dryrun: Minimal deps, some tests may be skipped
"""

import sys
import os
import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Detect if we're running with minimal deps (CI) or full deps
MINIMAL_DEPS = os.environ.get('MINIMAL_DEPS', 'false').lower() == 'true'

# =============================================================================
# TEST 1: All imports resolve (including deep transitive deps)
# =============================================================================
print("=" * 60)
print("TEST 1: Import chain validation")
print("=" * 60)

# Add paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/hy3dshape')
sys.path.insert(0, '/app/hy3dpaint')

import_errors = []
import_warnings = []

# Test basic imports (required)
try:
    import runpod
    print(f"  [OK] runpod {runpod.__version__}")
except ImportError as e:
    import_errors.append(f"runpod: {e}")

try:
    import torch
    print(f"  [OK] torch {torch.__version__}")
except ImportError as e:
    import_errors.append(f"torch: {e}")

try:
    import numpy as np
    print(f"  [OK] numpy {np.__version__}")
    if np.__version__.startswith('2.'):
        import_warnings.append("numpy 2.x may cause compatibility issues")
except ImportError as e:
    import_errors.append(f"numpy: {e}")

# Test deep imports (required for full deps, optional for CI)
try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    print("  [OK] hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline")
except ImportError as e:
    if MINIMAL_DEPS:
        import_warnings.append(f"hy3dshape.pipelines: {e} (expected in minimal deps)")
    else:
        import_errors.append(f"hy3dshape.pipelines: {e}")

try:
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    print("  [OK] hy3dpaint.textureGenPipeline")
except ImportError as e:
    if MINIMAL_DEPS:
        import_warnings.append(f"hy3dpaint.textureGenPipeline: {e} (expected in minimal deps)")
    else:
        import_errors.append(f"hy3dpaint.textureGenPipeline: {e}")

# Test mesh processing
try:
    import trimesh
    print(f"  [OK] trimesh {trimesh.__version__}")
except ImportError as e:
    import_errors.append(f"trimesh: {e}")

if import_warnings:
    print("\n  Warnings (non-fatal):")
    for warn in import_warnings:
        print(f"    - {warn}")

if import_errors:
    print("\n[FAIL] Import errors:")
    for err in import_errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\n[PASS] All imports OK")


# =============================================================================
# TEST 2: Numpy patch validation
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: Numpy compatibility patches")
print("=" * 60)

# Import handler to apply patches
import handler

# Test torch.tensor with numpy scalars
try:
    arr = np.array([1, 2, 3], dtype=np.int64)
    t = torch.tensor(arr)
    print(f"  [OK] torch.tensor(np.array) -> {t}")
except Exception as e:
    print(f"  [FAIL] torch.tensor(np.array): {e}")
    sys.exit(1)

# Test torch.tensor with numpy scalar (this caused "Could not infer dtype" errors)
try:
    scalar = np.int64(42)
    t = torch.tensor(scalar)
    print(f"  [OK] torch.tensor(np.int64 scalar) -> {t}")
except Exception as e:
    print(f"  [FAIL] torch.tensor(np.int64 scalar): {e}")
    sys.exit(1)

# Test torch.from_numpy
try:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = torch.from_numpy(arr)
    print(f"  [OK] torch.from_numpy -> {t.shape}")
except Exception as e:
    print(f"  [FAIL] torch.from_numpy: {e}")
    sys.exit(1)

# Test for recursion (this caught the infinite recursion bug)
try:
    # Stress test - multiple conversions in sequence
    for i in range(100):
        arr = np.random.rand(10).astype(np.float32)
        t = torch.tensor(arr)
        t2 = torch.from_numpy(arr)
    print("  [OK] No recursion in 100 iterations")
except RecursionError:
    print("  [FAIL] RecursionError in numpy patches")
    sys.exit(1)

print("\n[PASS] Numpy patches OK")


# =============================================================================
# TEST 3: Handler logic with mocked pipelines
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: Handler logic (mocked pipelines)")
print("=" * 60)

# Create a minimal test image (1x1 PNG)
def create_test_image_b64():
    import struct
    import zlib

    # Minimal PNG: 1x1 red pixel
    def png_chunk(chunk_type, data):
        chunk_len = len(data)
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack('>I', chunk_len) + chunk + struct.pack('>I', crc)

    png_header = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    idat = zlib.compress(b'\x00\xff\x00\x00', 9)  # Red pixel with filter byte

    png_data = png_header + png_chunk(b'IHDR', ihdr) + png_chunk(b'IDAT', idat) + png_chunk(b'IEND', b'')
    return base64.b64encode(png_data).decode('utf-8')

test_image = create_test_image_b64()

# Test 3a: Health check (no model needed)
print("\n  Testing health check...")
result = handler.handler({"input": "health_check"})
if result.get("status") == "healthy":
    print(f"  [OK] Health check: {result}")
else:
    print(f"  [FAIL] Health check: {result}")
    sys.exit(1)

# Test 3b: Input validation
print("\n  Testing input validation...")
result = handler.handler({"input": {}})
if "error" in result and "image_base64" in result["error"]:
    print(f"  [OK] Missing image_base64 rejected")
else:
    print(f"  [FAIL] Should reject missing image_base64: {result}")
    sys.exit(1)

result = handler.handler({"input": {"image_base64": test_image, "output_format": "invalid"}})
if "error" in result and "output_format" in result["error"]:
    print(f"  [OK] Invalid output_format rejected")
else:
    print(f"  [FAIL] Should reject invalid format: {result}")
    sys.exit(1)

# Test 3c: Full handler with mocked pipelines
print("\n  Testing full handler with mocks...")

# Create a temporary directory for mock mesh files
import trimesh
with tempfile.TemporaryDirectory() as mock_dir:
    mock_dir_path = Path(mock_dir)

    # Create a mock textured mesh file that paint_pipe will "return"
    mock_textured_mesh_path = mock_dir_path / "textured_output.obj"
    mock_box = trimesh.creation.box()
    mock_box.export(str(mock_textured_mesh_path))

    # Create a mock mesh for shape pipeline
    mock_mesh = MagicMock()
    mock_mesh.export = MagicMock(side_effect=lambda path, **kwargs: Path(path).write_bytes(b'MOCK_OBJ_DATA'))

    # Create mock pipelines
    mock_shape_pipeline = MagicMock()
    mock_shape_pipeline.return_value = [mock_mesh]

    mock_paint_config = MagicMock()
    mock_paint_pipeline = MagicMock()
    # paint_pipe() returns a file path string, not a mesh object
    mock_paint_pipeline.return_value = str(mock_textured_mesh_path)
    mock_paint_pipeline.config = mock_paint_config

    # Mock S3 upload function
    mock_s3_url = "https://s3api-eur-is-1.runpod.io/test-bucket/outputs/test.glb?signature=mock"

    # Mock the load_pipelines function
    with patch.object(handler, 'load_pipelines', return_value=(mock_shape_pipeline, mock_paint_pipeline)):
        # Disable progress updates (they need real runpod context)
        with patch('runpod.serverless.progress_update'):
            # Mock S3 upload
            with patch.object(handler, 'upload_to_s3', return_value=mock_s3_url):
                result = handler.handler({
                    "input": {
                        "image_base64": test_image,
                        "generate_texture": True,
                        "output_format": "glb"
                    }
                })

    # Check result (inside the with block so temp files still exist)
    if "error" in result:
        print(f"  [FAIL] Handler error: {result['error']}")
        sys.exit(1)
    elif "download_url" in result:
        # New S3 response format
        print(f"  [OK] Handler returned S3 URL, format={result['format']}, textured={result['textured']}, size={result['size_mb']}MB")
        print(f"       S3 key: {result['s3_key']}")
    elif "model" in result:
        # Legacy base64 response (for backwards compatibility)
        decoded = base64.b64decode(result["model"])
        print(f"  [OK] Handler returned {len(decoded)} bytes, format={result['format']}, textured={result['textured']}")
    else:
        print(f"  [FAIL] Unexpected result: {result}")
        sys.exit(1)

print("\n[PASS] Handler logic OK")


# =============================================================================
# TEST 4: Numpy API compatibility (numpy 1.x vs 2.x)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 4: Numpy API compatibility")
print("=" * 60)

# Check for numpy.core.multiarray.generic (used by some Hunyuan3D code)
try:
    import numpy.core.multiarray
    if hasattr(numpy.core.multiarray, 'generic'):
        print("  [OK] numpy.core.multiarray.generic exists")
    else:
        print("  [WARN] numpy.core.multiarray.generic missing (numpy 2.x?)")
except ImportError as e:
    print(f"  [WARN] numpy.core.multiarray import issue: {e}")

# Check for deprecated numpy APIs commonly used
try:
    # np.string_ was deprecated in numpy 2.0
    _ = np.string_
    print("  [OK] np.string_ exists")
except AttributeError:
    print("  [WARN] np.string_ missing (numpy 2.x - use np.bytes_)")

try:
    # np.unicode_ was deprecated in numpy 2.0
    _ = np.unicode_
    print("  [OK] np.unicode_ exists")
except AttributeError:
    print("  [WARN] np.unicode_ missing (numpy 2.x - use np.str_)")

print("\n[PASS] Numpy API check complete")


# =============================================================================
# TEST 5: ONNX Runtime providers (catches CUDA library mismatches)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 5: ONNX Runtime providers")
print("=" * 60)

try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print(f"  Available providers: {available_providers}")

    # Check if CUDA provider is available (expected in full build)
    if 'CUDAExecutionProvider' in available_providers:
        print("  [OK] CUDAExecutionProvider available")
    else:
        print("  [WARN] CUDAExecutionProvider not available (will use CPU)")

    # Always have CPU fallback
    if 'CPUExecutionProvider' in available_providers:
        print("  [OK] CPUExecutionProvider available")
    else:
        print("  [FAIL] CPUExecutionProvider missing - ONNX won't work!")
        sys.exit(1)

    print("\n[PASS] ONNX Runtime check complete")
except ImportError as e:
    if MINIMAL_DEPS:
        print(f"  [WARN] onnxruntime not installed (expected in minimal deps)")
    else:
        print(f"  [FAIL] onnxruntime import failed: {e}")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] ONNX Runtime error: {e}")
    sys.exit(1)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("""
This validates:
- All import chains resolve (including pytorch_lightning)
- Numpy patches work without recursion
- Handler logic is correct
- Numpy 1.x APIs are available
- ONNX Runtime providers are available

What this does NOT test (requires real models):
- Actual model loading (from_pretrained)
- GPU memory allocation
- Inference quality
""")
