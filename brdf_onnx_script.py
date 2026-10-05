"""Roy et al. BRDF c-factor correction authored with ONNX Script, exported to ONNX.

Third authoring path for the same model as `brdf_onnx_model.py` (raw ONNX primitives) and
`brdf_onnx_torch.py` (PyTorch + exporter). ONNX Script is Microsoft's official DSL for
authoring ONNX directly in a subset of Python, and it is the foundation the PyTorch ONNX
exporter is built on - so this gives similar readability to the torch version without
torch's ~1 GB of dependencies (including CUDA wheels).

Run
---
    uv run --with onnxscript python brdf_onnx_script.py
    uv run --with onnxscript python brdf_onnx_script.py --out roy_brdf_onnxscript.onnx

Why consider this over the other two
------------------------------------
- vs raw primitives: the math reads as ordinary expressions (`a * b + c`) instead of
  nested `helper.make_node(..., [add(...), add(...)])` with hand-threaded names.
- vs torch: one small dependency, no CUDA wheels, and no graph-capture / decomposition /
  opset-downgrade pipeline between source and artifact.

Same runtime constraints as both (backend facts, not authoring choices):
- float64 I/O is required (`predict_onnx` input dtype must equal the tile celltype), but
  `Tan`/`Atan`/`Acos` only have float32 kernels in common ONNX Runtime builds, so the
  trigonometry runs in float32 and the graph casts at the boundary.
- Fixed spatial dims H x W, because the backend retiles the cube to the model's literal
  `shape[-2:]`.
- The +/-20% clamp has per-pixel bounds, so it must be `Max`/`Min`, not `Clip`
  (ONNX `Clip` accepts only scalar min/max).

ONNX Script constraints worth knowing
-------------------------------------
- `@script()` traces the function *source*, so this must live in a real file. It cannot be
  used on a function defined in a REPL or via `exec`/`compile`.
- The traced subset of Python is limited: control flow and statements like `del` are
  rejected by its AST analyzer, and a nested helper is only usable if it is also traced
  (here the kernels are nested `def`s returning ONNX values, which works because they are
  inlined during tracing).
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnxscript import DOUBLE, FLOAT, script
from onnxscript import opset18 as op

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
D2R = 0.017453292519943295  # pi / 180
INV_PI = 1.0 / np.pi
PI_OVER_2 = np.pi / 2.0
FOUR_OVER_3PI = 4.0 / (3.0 * np.pi)
ONE_THIRD = 1.0 / 3.0


def channel_order() -> list[str]:
    """The exact channel order the model expects (for documentation/asserts)."""
    return BAND_ORDER + ANGLE_ORDER + ["theta_s"]


def _coef_array() -> np.ndarray:
    """Roy coefficients as a (6, 3) float32 array, rows in BAND_ORDER."""
    return np.array([ROY_COEF[b] for b in BAND_ORDER], dtype=np.float32)


# ---------------------------------------------------------------------------
# The model, written as ordinary Python expressions.
#
# Input (11, H, W) float64 -> output (6, H, W) float64.
#
# `coef` is a (6, 3) float32 initializer; `clip_lo`/`clip_hi` are scalar float32
# initializers. Supplying them as inputs keeps the constants out of the traced source,
# so the traced function stays pure math.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernels as module-level traced functions.
#
# NOTE: ONNX Script cannot trace a nested `def` that captures the enclosing function's
# locals or parameters - it errors with "Missing names ... at: Function 'chan'". So the
# kernels live at module level and take everything they need as arguments, and the
# channel extraction below is written out per channel rather than via a helper.
# ---------------------------------------------------------------------------
@script()
def kgeo_li_sparse_script(sza_deg: FLOAT, vza_deg: FLOAT, dphi_deg: FLOAT) -> FLOAT:
    """Li-Sparse geometric kernel. Mirrors roy_brdf.kgeo_li_sparse."""
    b_sur_r = op.Constant(value_floats=[B_SUR_R])
    one_f = op.Constant(value_floats=[1.0])
    neg_one_f = op.Constant(value_floats=[-1.0])
    half_f = op.Constant(value_floats=[0.5])
    two_f = op.Constant(value_floats=[2.0])

    ts = op.Mul(sza_deg, op.Constant(value_floats=[D2R]))
    tv = op.Mul(vza_deg, op.Constant(value_floats=[D2R]))
    phi = op.Mul(dphi_deg, op.Constant(value_floats=[D2R]))

    ts_p = op.Atan(op.Mul(b_sur_r, op.Tan(ts)))
    tv_p = op.Atan(op.Mul(b_sur_r, op.Tan(tv)))

    cos_ts_p = op.Cos(ts_p)
    cos_tv_p = op.Cos(tv_p)
    sin_ts_p = op.Sin(ts_p)
    sin_tv_p = op.Sin(tv_p)
    cos_phi = op.Cos(phi)
    sin_phi = op.Sin(phi)

    cos_zeta_p = cos_ts_p * cos_tv_p + sin_ts_p * sin_tv_p * cos_phi

    tan_ts = op.Tan(ts_p)
    tan_tv = op.Tan(tv_p)
    d = op.Sqrt(tan_ts * tan_ts + tan_tv * tan_tv - two_f * tan_ts * tan_tv * cos_phi)

    sec_s = op.Reciprocal(cos_ts_p)
    sec_v = op.Reciprocal(cos_tv_p)

    tps = tan_ts * tan_tv * sin_phi
    numerator = op.Sqrt(d * d + tps * tps)

    cos_t = H_SUR_B * (numerator / (sec_s + sec_v))
    # acos domain: clamp BEFORE sqrt(1 - cos^2).
    cos_t = op.Clip(cos_t, neg_one_f, one_f)

    sin_t = op.Sqrt(one_f - cos_t * cos_t)
    t = op.Acos(cos_t)

    overlap = INV_PI * (t - sin_t * cos_t) * (sec_s + sec_v)
    return overlap - sec_s - sec_v + half_f * (one_f + cos_zeta_p) * sec_s * sec_v


@script()
def kvol_ross_thick_script(sza_deg: FLOAT, vza_deg: FLOAT, dphi_deg: FLOAT) -> FLOAT:
    """Ross-Thick volumetric kernel. Mirrors roy_brdf.kvol_ross_thick."""
    one_f = op.Constant(value_floats=[1.0])
    neg_one_f = op.Constant(value_floats=[-1.0])

    ts = op.Mul(sza_deg, op.Constant(value_floats=[D2R]))
    tv = op.Mul(vza_deg, op.Constant(value_floats=[D2R]))
    phi = op.Mul(dphi_deg, op.Constant(value_floats=[D2R]))

    cos_zeta = op.Cos(ts) * op.Cos(tv) + op.Sin(ts) * op.Sin(tv) * op.Cos(phi)
    cos_zeta = op.Clip(cos_zeta, neg_one_f, one_f)
    zeta = op.Acos(cos_zeta)

    numerator = (PI_OVER_2 - zeta) * op.Cos(zeta) + op.Sin(zeta)
    denominator = op.Cos(tv) + op.Cos(ts)
    return FOUR_OVER_3PI * (numerator / denominator) - ONE_THIRD


@script()
def roy_brdf_script(x: DOUBLE, coef: FLOAT, clip_lo: FLOAT, clip_hi: FLOAT) -> DOUBLE:
    """Input (11, H, W) float64 -> output (6, H, W) float64.

    Everything internal is float32: Tan/Atan/Acos lack float64 kernels in the ONNX
    Runtime builds the backend uses. `FLOAT.dtype` is the tensor-element enum that
    Cast's `to` attribute expects (passing the type class gives a confusing
    "int() argument must be ... not 'ABCMeta'").
    """
    one_f = op.Constant(value_floats=[1.0])
    zero_f = op.Constant(value_floats=[0.0])
    axis0 = op.Constant(value_ints=[0])

    x32 = op.Cast(x, to=FLOAT.dtype)

    # Reflectance block: (6, H, W).
    # Slice takes starts/ends/axes as 1-D int tensors (inputs, not attributes).
    rho = op.Slice(
        x32,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[len(BAND_ORDER)]),
        op.Constant(value_ints=[0]),
    )

    # ------------------------------------------------------------------
    # Angle channels: each (H, W).
    #
    # API notes (these differ per operator and are easy to get wrong):
    #   Gather  -> `indices` is a tensor input; `axis` is an ATTRIBUTE (plain int).
    #   Squeeze -> `axes` is a regular argument taking a plain int list, NOT a tensor.
    # Writing it out per channel because a nested helper cannot be traced.
    # ------------------------------------------------------------------
    sza = op.Squeeze(op.Gather(x32, op.Constant(value_ints=[len(BAND_ORDER) + 0]), axis=0), [0])
    vza = op.Squeeze(op.Gather(x32, op.Constant(value_ints=[len(BAND_ORDER) + 1]), axis=0), [0])
    saa = op.Squeeze(op.Gather(x32, op.Constant(value_ints=[len(BAND_ORDER) + 2]), axis=0), [0])
    vaa = op.Squeeze(op.Gather(x32, op.Constant(value_ints=[len(BAND_ORDER) + 3]), axis=0), [0])
    theta_s = op.Squeeze(op.Gather(x32, op.Constant(value_ints=[len(BAND_ORDER) + 4]), axis=0), [0])

    dphi = op.Sub(saa, vaa)

    kgeo_in = kgeo_li_sparse_script(sza, vza, dphi)
    kvol_in = kvol_ross_thick_script(sza, vza, dphi)

    # Reference geometry: view straight down, sun at the scene reference angle.
    zero = theta_s * 0.0
    kgeo_norm = kgeo_li_sparse_script(theta_s, zero, zero)
    kvol_norm = kvol_ross_thick_script(theta_s, zero, zero)

    # ---- per-band coefficients as (6, 1, 1) to broadcast over (H, W) --------
    # Gather(col, axis=1) on a (6, 3) coefficient table yields (6, 1); a single
    # Unsqueeze at axis 2 then gives (6, 1, 1), which broadcasts against the
    # (6, H, W) reflectance. Using axes [1, 2] here would produce a 4-D (6,1,1,1)
    # that broadcasts to the wrong (6, 6, H, W) result.
    f_iso = op.Unsqueeze(op.Gather(coef, op.Constant(value_ints=[0]), axis=1), [2])
    f_geo = op.Unsqueeze(op.Gather(coef, op.Constant(value_ints=[1]), axis=1), [2])
    f_vol = op.Unsqueeze(op.Gather(coef, op.Constant(value_ints=[2]), axis=1), [2])

    # Expand the (H, W) kernels to (1, H, W) and the (6, 1, 1) coefficients so the
    # product is unambiguously (6, H, W). Without this, shape inference through the
    # custom-domain kernel functions can mis-broadcast to (6, 6, H, W).
    kgeo_in3 = op.Unsqueeze(kgeo_in, [0])
    kvol_in3 = op.Unsqueeze(kvol_in, [0])
    kgeo_norm3 = op.Unsqueeze(kgeo_norm, [0])
    kvol_norm3 = op.Unsqueeze(kvol_norm, [0])

    numerator = f_iso + f_geo * kgeo_norm3 + f_vol * kvol_norm3
    denominator = f_iso + f_geo * kgeo_in3 + f_vol * kvol_in3
    c = numerator / denominator

    # ---- apply, clamp, nodata guard ----------------------------------------
    out = c * rho
    # Not op.Clip: the bounds are (6, H, W) tensors and Clip needs scalars.
    out = op.Min(op.Max(out, clip_lo * rho), clip_hi * rho)
    # rho <= 0 -> 0
    out = op.Where(op.LessOrEqual(rho, zero_f), zero_f, out)

    return op.Cast(out, to=DOUBLE.dtype)


def build(height: int = 32, width: int = 32) -> onnx.ModelProto:
    """Build the model with fixed H x W and float64 I/O.

    The coefficient and clamp constants are supplied as initializers here, so the traced
    function receives them as ordinary inputs.
    """
    coef = onnx.numpy_helper.from_array(_coef_array(), name="coef")
    clip_lo = onnx.numpy_helper.from_array(
        np.array(1.0 - CLIP_FACTOR, dtype=np.float32), name="clip_lo"
    )
    clip_hi = onnx.numpy_helper.from_array(
        np.array(1.0 + CLIP_FACTOR, dtype=np.float32), name="clip_hi"
    )

    # ONNX Script expects ONNXType objects (not strings) for the input/output types.
    model = roy_brdf_script.to_model_proto(
        input_types=[
            DOUBLE[N_INPUT_CHANNELS, height, width],
            FLOAT[len(BAND_ORDER), 3],
            FLOAT,
            FLOAT,
        ],
        output_types=[DOUBLE[N_OUTPUT_CHANNELS, height, width]],
    )
    # The traced constants become graph inputs; retag the three constant ones as
    # initializers so the model has exactly one real input, as predict_onnx requires.
    const_names = {"coef", "clip_lo", "clip_hi"}
    for init in (coef, clip_lo, clip_hi):
        model.graph.initializer.append(init)
    kept_inputs = [i for i in model.graph.input if i.name not in const_names]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)
    return model


def export(path: str = "roy_brdf_onnxscript.onnx", height: int = 32, width: int = 32,
           opset: int = 17, ir_version: int = 10) -> str:
    """Build and save, pinning opset/IR for the backend's ONNX Runtime.

    Careful with opset_import: ONNX Script emits the two kernels as ONNX *functions* in
    a custom domain ("this"). Dropping that import (as a naive "set the opset" step
    would) makes the checker fail with "No opset import for domain 'this'". The default
    domain is therefore rewritten in place rather than the list being replaced.
    """
    model = build(height, width)

    # Rewrite only the default domain's version; preserve custom domains ("this").
    for op_import in model.opset_import:
        if op_import.domain == "" and op_import.version != opset:
            op_import.version = opset

    if model.ir_version > ir_version:
        model.ir_version = ir_version
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def verify(height: int = 32, width: int = 32) -> bool:
    """Compare the exported graph against roy_brdf.py, and check the I/O dtype."""
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

    path = "/tmp/_roy_brdf_script_verify.onnx"
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
    print(f"  output shape: {got.shape} (expected {(N_OUTPUT_CHANNELS, height, width)})")
    if got.shape != (N_OUTPUT_CHANNELS, height, width):
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

    # Identity geometry: no correction needed, so output == input (float32 tolerance).
    ident = x.copy()
    ident[6], ident[7], ident[8], ident[9], ident[10] = 41.0, 0.0, 100.0, 100.0, 41.0
    out_id: np.ndarray = np.asarray(sess.run(None, {"x": ident})[0])
    rel_dev = float(np.abs(out_id - ident[0:6]).max() / max(1e-9, np.abs(ident[0:6]).max()))
    print(f"  identity geometry relative deviation: {rel_dev:.2e}"
          f" -> {'OK' if rel_dev < 1e-6 else 'FAILED'}")

    return worst < 1e-5 and rel_dev < 1e-6


def verify_against_backend() -> None:
    """Compare against the real backend UDF output, if brdf_input.nc is available."""
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
    path = "/tmp/_roy_brdf_script_backend.onnx"
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
    ap.add_argument("--out", default="roy_brdf_onnxscript.onnx")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--skip-backend", action="store_true")
    args = ap.parse_args()

    p = export(args.out, args.height, args.width)
    m = onnx.load(p)
    print(f"wrote {p}")
    print(f"  nodes: {len(m.graph.node)} | ir: {m.ir_version} | opset: {m.opset_import[0].version}")
    print(f"  functions: {len(m.functions)} (custom domain "
          f"{sorted({o.domain for o in m.opset_import if o.domain})})")

    print("\nverify vs roy_brdf.py")
    ok = verify(args.height, args.width)

    if not args.skip_backend:
        print("\nverify vs backend UDF output")
        verify_against_backend()

    raise SystemExit(0 if ok else 1)
