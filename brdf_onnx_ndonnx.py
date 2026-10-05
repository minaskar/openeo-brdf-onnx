"""Roy et al. BRDF c-factor correction authored with ndonnx, exported to ONNX.

Fourth authoring path for the same model:
  - `brdf_onnx_model.py`  raw ONNX primitives
  - `brdf_onnx_torch.py`  PyTorch + torch.onnx.export
  - `brdf_onnx_script.py` ONNX Script (@script DSL)
  - this file             ndonnx (Array API standard on top of spox)

ndonnx implements the Python Array API standard on top of spox, so the math is written
with numpy-like operations (`ndx.sin`, `ndx.clip`, `ndx.where`) and no DSL tracing rules,
no graph-capture pipeline, and no operator-signature puzzles. Export is a single call.

Run
---
    uv run --with ndonnx --no-project python brdf_onnx_ndonnx.py
    uv run --with ndonnx --no-project python brdf_onnx_ndonnx.py --out roy_brdf_ndonnx.onnx

Dependencies: `ndonnx`, which pulls `numpy>=2`, `spox` and `typing_extensions` - small,
with no CUDA wheels. NOTE: ndonnx requires numpy>=2, so it is deliberately run with
`--no-project` (this repo's venv has numpy 2.5.3, but ndonnx is not a project dependency).

Same backend constraints as every other path (these are runtime facts, not authoring
choices):
- float64 I/O, because `predict_onnx` requires the model input dtype to equal the tile
  celltype, while `Tan`/`Atan`/`Acos` only have float32 kernels in the ONNX Runtime builds
  the backend uses -> compute in float32, cast at the boundary.
- Fixed spatial dims H x W: the backend retiles the cube to the model's literal shape[-2:].
- The +/-20% clamp has per-pixel bounds, so it must be Max/Min, not Clip (ONNX Clip takes
  only scalar min/max.
- Opset 21 (as emitted by ndonnx) requires ONNX Runtime >= 1.18.

Why this may be the best of the four
------------------------------------
- vs ONNX Script: no `@script` tracing restrictions. Nested helpers, `np.pi`, `del`,
  loops and ordinary Python control flow are all fine, because ndonnx traces *array
  operations*, not source AST.
- vs raw primitives: plain numpy-style expressions.
- vs torch: no 1 GB dependency, no export pipeline, no opset-downgrade surprises.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import ndonnx as ndx

# ---------------------------------------------------------------------------
# Same constants as roy_brdf.py / brdf_onnx_model.py.
# Roy et al. (2017) Table 1 / Roy et al. (2016) Table 5 / Sen2Like ATBD Table 6.
# ---------------------------------------------------------------------------
BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]
ANGLE_ORDER = ["sunZenithAngles", "viewZenithMean", "sunAzimuthAngles", "viewAzimuthMean"]
N_INPUT_CHANNELS = len(BAND_ORDER) + len(ANGLE_ORDER) + 1  # +1 for theta_s
N_OUTPUT_CHANNELS = len(BAND_ORDER)

ROY_COEF = {
    "B02": (0.0774, 0.0079, 0.0372),
    "B03": (0.1306, 0.0178, 0.0580),
    "B04": (0.1690, 0.0227, 0.0574),
    "B8A": (0.3093, 0.0330, 0.1535),
    "B11": (0.3430, 0.0453, 0.1154),
    "B12": (0.2658, 0.0387, 0.0639),
}

CLIP_FACTOR = 0.2
H_SUR_B = 2.0
B_SUR_R = 1.0
D2R = np.pi / 180.0


def channel_order() -> list[str]:
    """The exact channel order the model expects (for documentation/asserts)."""
    return BAND_ORDER + ANGLE_ORDER + ["theta_s"]


def _coef_array() -> np.ndarray:
    """Roy coefficients as a (6, 3) float32 array, rows in BAND_ORDER."""
    return np.array([ROY_COEF[b] for b in BAND_ORDER], dtype=np.float32)


# ---------------------------------------------------------------------------
# Kernels. Plain numpy-style expressions - no DSL restrictions.
# ---------------------------------------------------------------------------
def kgeo_li_sparse(sza_deg, vza_deg, dphi_deg):
    """Li-Sparse geometric kernel. Mirrors roy_brdf.kgeo_li_sparse."""
    ts = sza_deg * D2R
    tv = vza_deg * D2R
    phi = dphi_deg * D2R

    ts_p = ndx.atan(B_SUR_R * ndx.tan(ts))
    tv_p = ndx.atan(B_SUR_R * ndx.tan(tv))

    cos_ts_p = ndx.cos(ts_p)
    cos_tv_p = ndx.cos(tv_p)
    sin_ts_p = ndx.sin(ts_p)
    sin_tv_p = ndx.sin(tv_p)
    cos_phi = ndx.cos(phi)
    sin_phi = ndx.sin(phi)

    cos_zeta_p = cos_ts_p * cos_tv_p + sin_ts_p * sin_tv_p * cos_phi

    tan_ts = ndx.tan(ts_p)
    tan_tv = ndx.tan(tv_p)
    d = ndx.sqrt(tan_ts * tan_ts + tan_tv * tan_tv - 2.0 * tan_ts * tan_tv * cos_phi)

    sec_s = 1.0 / cos_ts_p
    sec_v = 1.0 / cos_tv_p

    tps = tan_ts * tan_tv * sin_phi
    numerator = ndx.sqrt(d * d + tps * tps)

    cos_t = H_SUR_B * (numerator / (sec_s + sec_v))
    # acos domain: clamp BEFORE sqrt(1 - cos^2).
    cos_t = ndx.clip(cos_t, -1.0, 1.0)

    sin_t = ndx.sqrt(1.0 - cos_t * cos_t)
    t = ndx.acos(cos_t)

    overlap = (1.0 / np.pi) * (t - sin_t * cos_t) * (sec_s + sec_v)
    return overlap - sec_s - sec_v + 0.5 * (1.0 + cos_zeta_p) * sec_s * sec_v


def kvol_ross_thick(sza_deg, vza_deg, dphi_deg):
    """Ross-Thick volumetric kernel. Mirrors roy_brdf.kvol_ross_thick."""
    ts = sza_deg * D2R
    tv = vza_deg * D2R
    phi = dphi_deg * D2R

    cos_zeta = ndx.cos(ts) * ndx.cos(tv) + ndx.sin(ts) * ndx.sin(tv) * ndx.cos(phi)
    cos_zeta = ndx.clip(cos_zeta, -1.0, 1.0)
    zeta = ndx.acos(cos_zeta)

    numerator = (np.pi / 2.0 - zeta) * ndx.cos(zeta) + ndx.sin(zeta)
    denominator = ndx.cos(tv) + ndx.cos(ts)
    return (4.0 / (3.0 * np.pi)) * (numerator / denominator) - (1.0 / 3.0)


def build_graph(x, coef):
    """Build the output from traced arrays. Returns the (6, H, W) output array.

    `x` is the (11, H, W) float64 input array; `coef` is a (6, 3) float32 constant.
    """
    # Internal compute in float32 (see module docstring).
    x32 = x.astype(ndx.float32)

    # ndonnx (unlike numpy) requires full-rank indexing: a bare integer or a short
    # slice raises "length of 'key' ... must match array's rank". The spatial axes
    # must therefore be spelled out explicitly.
    rho = x32[0 : len(BAND_ORDER), :, :]
    sza = x32[len(BAND_ORDER) + 0, :, :]
    vza = x32[len(BAND_ORDER) + 1, :, :]
    saa = x32[len(BAND_ORDER) + 2, :, :]
    vaa = x32[len(BAND_ORDER) + 3, :, :]
    theta_s = x32[len(BAND_ORDER) + 4, :, :]

    dphi = saa - vaa

    kgeo_in = kgeo_li_sparse(sza, vza, dphi)
    kvol_in = kvol_ross_thick(sza, vza, dphi)

    # Reference geometry: view straight down, sun at the scene reference angle.
    zero = theta_s * 0.0
    kgeo_norm = kgeo_li_sparse(theta_s, zero, zero)
    kvol_norm = kvol_ross_thick(theta_s, zero, zero)

    # Coefficients as (6, 1, 1) so they broadcast against the (H, W) kernels.
    # Slicing a (6, 3) table as [:, i:i+1] gives (6, 1); reshape adds the last axis.
    f_iso = ndx.reshape(coef[:, 0:1], (len(BAND_ORDER), 1, 1))
    f_geo = ndx.reshape(coef[:, 1:2], (len(BAND_ORDER), 1, 1))
    f_vol = ndx.reshape(coef[:, 2:3], (len(BAND_ORDER), 1, 1))

    # Expand the (H, W) kernels to (1, H, W) so the broadcast is unambiguous.
    kgeo_in3 = ndx.reshape(kgeo_in, (1,) + kgeo_in.shape)
    kvol_in3 = ndx.reshape(kvol_in, (1,) + kvol_in.shape)
    kgeo_norm3 = ndx.reshape(kgeo_norm, (1,) + kgeo_norm.shape)
    kvol_norm3 = ndx.reshape(kvol_norm, (1,) + kvol_norm.shape)

    numerator = f_iso + f_geo * kgeo_norm3 + f_vol * kvol_norm3
    denominator = f_iso + f_geo * kgeo_in3 + f_vol * kvol_in3
    c = numerator / denominator

    out = c * rho
    # Not ndx.clip: the bounds are per-pixel arrays and ONNX Clip needs scalars.
    out = ndx.minimum(ndx.maximum(out, (1.0 - CLIP_FACTOR) * rho), (1.0 + CLIP_FACTOR) * rho)
    # nodata guard: rho <= 0 -> 0
    out = ndx.where(rho <= 0.0, ndx.zeros_like(out), out)

    # Cast back to float64 so output dtype matches input dtype (predict_onnx requires
    # input type == output type, and it must equal the tile celltype).
    return out.astype(ndx.float64)


def build(height: int = 32, width: int = 32) -> onnx.ModelProto:
    """Trace the graph and return a float64-in/float64-out ONNX model."""
    x = ndx.argument(shape=(N_INPUT_CHANNELS, height, width), dtype=ndx.float64)
    coef = ndx.asarray(_coef_array())
    out = build_graph(x, coef)
    return ndx.build({"x": x}, {"output": out})


def export(path: str = "roy_brdf_ndonnx.onnx", height: int = 32, width: int = 32) -> str:
    """Build, validate and save the model as emitted by ndonnx (opset 21, IR 8).

    No opset/IR downgrade is applied: opset 21 needs ONNX Runtime >= 1.18, which the
    openEO backend satisfies (verified with a predict_onnx job; output identical to an
    opset-17 build).
    """
    model = build(height, width)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def verify(height: int = 32, width: int = 32) -> bool:
    """Compare the exported graph against roy_brdf.py, and check I/O dtype/shape."""
    import onnxruntime as ort

    import roy_brdf

    m = build(height, width)
    names = {1: "float32", 11: "float64"}
    in_elem = m.graph.input[0].type.tensor_type.elem_type
    out_elem = m.graph.output[0].type.tensor_type.elem_type
    print(f"  I/O dtype: {names.get(in_elem)} in / {names.get(out_elem)} out")
    if in_elem != 11 or out_elem != 11:
        print("  FAILED: expected float64 in and out (backend tile celltype)")
        return False

    path = "/tmp/_roy_brdf_ndx_verify.onnx"
    onnx.save(m, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    x = np.zeros((N_INPUT_CHANNELS, height, width), dtype=np.float64)
    for i in range(len(BAND_ORDER)):
        x[i] = rng.uniform(100, 5000, (height, width))
    x[6], x[7], x[8], x[9] = 36.8562, 6.9308, 159.9136, 107.9809
    theta_s = float(roy_brdf.mean_sun_zenith(51.22))
    x[10] = theta_s

    got: np.ndarray = np.asarray(sess.run(None, {"x": x})[0])
    expected_shape = (N_OUTPUT_CHANNELS, height, width)
    print(f"  output shape: {got.shape} (expected {expected_shape})")
    if got.shape != expected_shape:
        print("  FAILED: wrong output shape")
        return False

    worst = 0.0
    for i, band in enumerate(BAND_ORDER):
        ref: np.ndarray = np.asarray(
            roy_brdf.nbar(
                x[i], x[6], x[7], x[8] - x[9], roy_brdf.ROY_COEF[band], sza_norm=theta_s
            )
        )
        rel = float(np.abs(got[i] - ref).max() / max(1e-9, np.abs(ref).max()))
        worst = max(worst, rel)
        print(f"  {band}: rel diff vs roy_brdf = {rel:.2e}")
    print(f"  worst relative difference: {worst:.2e} -> {'OK' if worst < 1e-5 else 'FAILED'}")

    ident = x.copy()
    ident[6], ident[7], ident[8], ident[9], ident[10] = 41.0, 0.0, 100.0, 100.0, 41.0
    out_id: np.ndarray = np.asarray(sess.run(None, {"x": ident})[0])
    rel_dev = float(np.abs(out_id - ident[0:6]).max() / max(1e-9, np.abs(ident[0:6]).max()))
    print(f"  identity geometry relative deviation: {rel_dev:.2e}"
          f" -> {'OK' if rel_dev < 1e-6 else 'FAILED'}")

    return worst < 1e-5 and rel_dev < 1e-6


def verify_against_backend() -> None:
    """Compare against the real backend UDF output, if the NetCDF files are present."""
    import os

    if not os.path.exists("brdf_input.nc") or not os.path.exists("brdf_nbar.nc"):
        print("  (skipped: brdf_input.nc / brdf_nbar.nc not present)")
        return

    import onnxruntime as ort
    import xarray as xr

    import roy_brdf

    ds_in = xr.open_dataset("brdf_input.nc")
    ref = xr.open_dataset("brdf_nbar.nc")
    hy, wx = ds_in.sizes["y"], ds_in.sizes["x"]
    path = "/tmp/_roy_brdf_ndx_backend.onnx"
    onnx.save(build(hy, wx), path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    theta_s = float(roy_brdf.mean_sun_zenith(51.22))

    for ti in range(ds_in.sizes["t"]):
        x = np.zeros((N_INPUT_CHANNELS, hy, wx), dtype=np.float64)
        for i, b in enumerate(BAND_ORDER):
            x[i] = ds_in[b].isel(t=ti).values
        x[6] = ds_in["sunZenithAngles"].isel(t=ti).values
        x[7] = ds_in["viewZenithMean"].isel(t=ti).values
        x[8] = ds_in["sunAzimuthAngles"].isel(t=ti).values
        x[9] = ds_in["viewAzimuthMean"].isel(t=ti).values
        x[10] = theta_s
        got: np.ndarray = np.asarray(sess.run(None, {"x": x})[0])
        worst = max(
            np.abs(got[i] - ref[b].isel(t=ti).values).max()
            / max(1e-9, np.abs(ref[b].isel(t=ti).values).max())
            for i, b in enumerate(BAND_ORDER)
        )
        print(f"  t={ti}: max rel diff vs backend UDF = {worst:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="roy_brdf_ndonnx.onnx")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--skip-backend", action="store_true")
    args = ap.parse_args()

    p = export(args.out, args.height, args.width)
    m = onnx.load(p)
    print(f"wrote {p}")
    print(f"  nodes: {len(m.graph.node)} | ir: {m.ir_version} | opset: {m.opset_import[0].version}")

    print("\nverify vs roy_brdf.py")
    ok = verify(args.height, args.width)

    if not args.skip_backend:
        print("\nverify vs backend UDF output")
        verify_against_backend()

    raise SystemExit(0 if ok else 1)
