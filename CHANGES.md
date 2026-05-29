# Canonical-position MLP Fix — Change Report

## Problem

The SIREN MLP inside `diff_gaussian_mlp_max_rasterization` computes opacity as a function of the Gaussian's 3D position in view space (`g_xyz_vs`). For articulated hands, this position is the **posed** (deformed) coordinate via linear blend skinning. When the hand changes pose, the MLP receives different coordinates, producing inconsistent opacity and rendering artifacts.

## Fix

Feed the MLP **canonical** (rest pose) coordinates instead of posed coordinates, while keeping posed positions for rasterization geometry (projection, sorting, blending).

The MLP element `o_tilde = inv_rot_view * g_xyz_vs * (-g_inv_max_radius)` (in `mlp_kernels.cu:42`) is the only term depending on the Gaussian position within the SIREN integral. Changing `g_xyz_vs` → `g_cano_xyz_vs` in this expression makes opacity invariant to articulation.

## Files changed (12)

All paths relative to `manus_neural/`.

### 1. `submodules/diff-gaussian-mlp-max-rasterization/cuda_rasterizer/mlp_kernels.cu`

Core SIREN kernel. Added `g_cano_xyz_vs` parameter to both `forward_mlp_siren` and `backward_mlp_siren`.

- **Forward** (line 4): `o_tilde` now computed from `g_cano_xyz_vs` instead of `g_xyz_vs`. Ray-sphere intersection still uses `g_xyz_vs` (posed) for correct geometry.
- **Backward** (line 68): same signature change. Added separate output `dL_dg_cano_xyz_vs` for the canonical gradient. MLP-phase gradients (`inv_rot_view`, `inv_max_radius` from `o_tilde`) routed to canonical; ray-sphere gradients remain on posed.

### 2. `submodules/.../cuda_rasterizer/forward_mlp.h`

Updated `FORWARD_MLP::preprocess` and `FORWARD_MLP::render` declarations:
- `preprocess`: added `cano_means3D` (world-space) input and `cano_points_xyz_view` (view-space) output.
- `render`: added `cano_points_xyz_view` input.

### 3. `submodules/.../cuda_rasterizer/forward_mlp.cu`

- **`preprocessCUDA` kernel** (line 278–452): added `cano_orig_points` and `cano_points_xyz_view` params. After the standard frustum cull and projection, computes `cano_points_xyz_view = transformPoint4x3(cano_origin, viewmatrix)`. Skipped when `cano_orig_points == nullptr` (backward compat).
- **`renderMLPCUDA` kernel** (line 483–640): added `cano_points_xyz_view` to shared memory (`collected_cano_xyz[BLOCK_SIZE]`), fetched per-Gaussian, and passed as second argument to `forward_mlp_siren`.
- **Wrappers** `FORWARD_MLP::render` and `FORWARD_MLP::preprocess`: updated signatures and kernel launch arguments.

### 4. `submodules/.../cuda_rasterizer/backward_mlp.h`

Updated `BACKWARD_MLP::renderMLP` declaration:
- Added `cano_means3D_view` input and `dL_dcano_means3D_view` output.

### 5. `submodules/.../cuda_rasterizer/backward_mlp.cu`

- **`renderMLPCUDA` backward kernel** (line 709–970): added `cano_points_xyz_view` input and `dL_dcano_points_xyz_view` output. Shared memory expanded. Batch fetch loads canonical positions. Both `forward_mlp_siren` and `backward_mlp_siren` receive `collected_cano_xyz[j]`. Canonical gradients atomically accumulated into `dL_dcano_points_xyz_view`.
- **Wrapper** `BACKWARD_MLP::renderMLP`: updated signature and launch.

### 6. `submodules/.../cuda_rasterizer/rasterizer_mlp_impl.h`

Added `float3 *cano_means3D_view` field to `GeometryMLPState` struct — allocated and populated during forward preprocess, used by backward renderMLP.

### 7. `submodules/.../cuda_rasterizer/rasterizer_mlp_impl.cu`

- **`fromChunk`**: allocates `cano_means3D_view` buffer (same size as `means3D_view`).
- **`RasterizerMLP::forward`**: passes `cano_means3D` to preprocess; passes `geomState.cano_means3D_view` to render.
- **`RasterizerMLP::backward`**: passes `geomState.cano_means3D_view` to `BACKWARD_MLP::renderMLP`; accepts `dL_dcano_means3D_view` output buffer.

### 8. `submodules/.../cuda_rasterizer/rasterizer_mlp.h`

- **`forward`**: added `const float *cano_means3D` parameter (after `means3D`).
- **`backward`**: added `float *dL_dcano_means3D_view` output parameter (last before flags).

### 9. `submodules/.../rasterize_points.h`

- **`RasterizeGaussiansMLPCUDA`**: added `const torch::Tensor &cano_means3D` (third positional arg).
- **Backward**: no change (canonical view-space positions come from geometry buffer).

### 10. `submodules/.../rasterize_points.cu`

- **Forward bridge**: when `cano_means3D.numel() > 0`, extracts data pointer; passes `nullptr` otherwise (object module or legacy callers). Pointer forwarded to `RasterizerMLP::forward`.
- **Backward bridge**: creates `dL_dcano_means3D_view = zeros({P, 3})` buffer and passes data pointer to `RasterizerMLP::backward`. Gradient not returned in the output tuple (not needed for the current autograd graph — `_xyz` gets gradient through the LBS path).

### 11. `submodules/.../diff_gaussian_mlp_max_rasterization/__init__.py`

Python-facing API:
- `rasterize_gaussians_mlp()`: added `cano_means3D=None` kwarg → forwarded to autograd Function.
- `_RasterizeGaussians_MLP.forward()`: defaults to empty tensor, inserts into C++ args tuple after `means3D`, saved for backward.
- `_RasterizeGaussians_MLP.backward()`: restores `cano_means3D` from saved tensors, returns `None` gradient for it.
- `Gaussian_MLP_Rasterizer.forward()`: added `cano_means3D=None` kwarg → passed through the chain.

### 12. `src/utils/gaussian_utils.py`

Added `cano_means3D=cano_means` to the `rasterizer()` call (line 429). The `cano_means` variable was already the 3rd parameter of `render_gaussians()` — no changes needed in `hand_dynamic.py`, `object.py`, or `composite.py`.

## Files NOT changed

- `ext.cpp` — pybind11 auto-detects updated C++ signatures.
- All module files (`src/modules/*.py`) — already pass `cano_xyz` to `render_gaussians`.
- Classic code path (`*_classic.py`, `gaussian_utils_classic.py`) — no MLP, unaffected.
- `setup.py` / `CMakeLists.txt` — no new compilation units.

## Build

```bash
cd submodules/diff-gaussian-mlp-max-rasterization
python -m setup.py install
```

## Behavior

| Scenario | `cano_means3D` | Effect |
|---|---|---|
| Static object | `== posed_means` | Identical to before (no-op) |
| Articulated hand | `cano_xyz` (rest pose) | MLP opacity invariant to pose change |
| Legacy callers | `None`/empty tensor | Falls back to using `means3D` (backward compat) |
