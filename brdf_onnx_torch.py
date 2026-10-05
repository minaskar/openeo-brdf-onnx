"""Roy et al. BRDF c-factor correction, authored in PyTorch and exported to ONNX.

This is an **alternative authoring path** for the same model that
`brdf_onnx_model.py` builds with raw ONNX primitives. The backend still receives and
runs a `.onnx` file: torch is a development-time tool only, because openEO exposes
`predict_onnx` and has no torch runtime.

Purpose: readability. The math below is written as ordinary Python/torch operations, so
it can be read side by side with `roy_brdf.py` (the numpy reference) far more easily
than the hand-threaded node construction in `brdf_onnx_model.py`.

Usage
-----
    python brdf_onnx_torch.py                 # build + export + verify
    python brdf_onnx_torch.py --check-only    # skip export, just verify vs roy_brdf

Notes on the awkward parts (see module docstring of brdf_onnx_model.py for the rest):

float64 I/O
    The backend delivers SENTINEL2_L2A tiles as float64 and `predict_onnx` requires the
    model input dtype to equal the tile celltype. But common ONNX Runtime builds lack
    double kernels for `Tan`, `Atan` and `Acos`. So the model takes float64, casts to
    float32 for the trigonometry, and casts back. Torch's exporter is float32-centric,
    so this is the fragile part - verify_wrt_reference() checks the exported graph
    really is float64 in/out.

clamp with tensor bounds
    The +/-20% clamp has per-pixel bounds, while ONNX `Clip` accepts only scalars. In
    torch, `torch.clamp(x, min_tensor, max_tensor)` is legal and the exporter may emit a
    `Clip` node with tensor inputs, which fails at runtime on the backend. We therefore
    write the clamp as nested torch.maximum/torch.minimum, which exports to `Max`/`Min`
    and broadcasts. Same fix as the primitive builder needed.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

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

CLIP_FACTOR = 0.2  # sen2like clamps the correction to +/-20%

# Kernel shape parameters (Wanner et al. 1995).
H_SUR_B = 2.0
B_SUR_R = 1.0


def channel_order() -> list[str]:
    """The exact channel order the model expects (for documentation/asserts)."""
    return BAND_ORDER + ANGLE_ORDER + ["theta_s"]


class RoyBRDF(nn.Module):
    """Roy et al. c-factor BRDF normalization as a torch module.

    Input : (11, H, W) - 6 reflectance bands, 4 angle bands [deg], theta_s [deg]
    Output: (6, H, W)  - corrected reflectance, same band order as the input bands

    The forward pass mirrors `roy_brdf.py` function for function:
        kgeo_li_sparse  -> _kgeo
        kvol_ross_thick -> _kvol
        c_factor        -> _c_factor
        nbar            -> forward
    """

    def __init__(
        self,
        io_dtype: torch.dtype = torch.float64,
        compute_dtype: torch.dtype = torch.float32,
        clip_factor: float = CLIP_FACTOR,
    ) -> None:
        super().__init__()
        self.io_dtype = io_dtype
        self.compute_dtype = compute_dtype
        self.clip_factor = clip_factor

        # Coefficients as a constant buffer, one row per output band: (6, 3).
        # Kept in compute_dtype; `forward` casts the input to match.
        coef = torch.tensor([ROY_COEF[b] for b in BAND_ORDER], dtype=compute_dtype)
        self.register_buffer("coef", coef)  # (6, 3)

    # ---------------------------------------------------------------- kernels
    @staticmethod
    def _kgeo(sza_deg: torch.Tensor, vza_deg: torch.Tensor, dphi_deg: torch.Tensor) -> torch.Tensor:
        """Li-Sparse geometric kernel (Wanner et al. 1995). Mirrors roy_brdf.kgeo_li_sparse."""
        d2r = torch.pi / 180.0
        ts = sza_deg * d2r
        tv = vza_deg * d2r
        phi = dphi_deg * d2r

        ts_p = torch.atan(B_SUR_R * torch.tan(ts))
        tv_p = torch.atan(B_SUR_R * torch.tan(tv))

        cos_ts_p, sin_ts_p = torch.cos(ts_p), torch.sin(ts_p)
        cos_tv_p, sin_tv_p = torch.cos(tv_p), torch.sin(tv_p)
        cos_phi = torch.cos(phi)

        cos_zeta_p = cos_ts_p * cos_tv_p + sin_ts_p * sin_tv_p * cos_phi

        tan_ts, tan_tv = torch.tan(ts_p), torch.tan(tv_p)
        d = torch.sqrt(tan_ts**2 + tan_tv**2 - 2.0 * tan_ts * tan_tv * cos_phi)

        sec_s, sec_v = 1.0 / cos_ts_p, 1.0 / cos_tv_p
        numerator = torch.sqrt(d**2 + (tan_ts * tan_tv * torch.sin(phi)) ** 2)
        cos_t = H_SUR_B * (numerator / (sec_s + sec_v))
        # acos domain: clamp BEFORE sqrt(1 - cos^2).
        cos_t = torch.clamp(cos_t, -1.0, 1.0)

        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        t = torch.acos(cos_t)

        overlap = (1.0 / torch.pi) * (t - sin_t * cos_t) * (sec_s + sec_v)
        return overlap - sec_s - sec_v + 0.5 * (1.0 + cos_zeta_p) * sec_s * sec_v

    @staticmethod
    def _kvol(sza_deg: torch.Tensor, vza_deg: torch.Tensor, dphi_deg: torch.Tensor) -> torch.Tensor:
        """Ross-Thick volumetric kernel. Mirrors roy_brdf.kvol_ross_thick."""
        d2r = torch.pi / 180.0
        ts = sza_deg * d2r
        tv = vza_deg * d2r
        phi = dphi_deg * d2r

        cos_zeta = torch.cos(ts) * torch.cos(tv) + torch.sin(ts) * torch.sin(tv) * torch.cos(phi)
        cos_zeta = torch.clamp(cos_zeta, -1.0, 1.0)
        zeta = torch.acos(cos_zeta)

        numerator = (torch.pi / 2.0 - zeta) * torch.cos(zeta) + torch.sin(zeta)
        denominator = torch.cos(tv) + torch.cos(ts)
        return (4.0 / (3.0 * torch.pi)) * (numerator / denominator) - (1.0 / 3.0)

    # -------------------------------------------------------------- c-factor
    def _c_factor(
        self,
        sza: torch.Tensor,
        vza: torch.Tensor,
        dphi: torch.Tensor,
        theta_s: torch.Tensor,
        coef: torch.Tensor,
    ) -> torch.Tensor:
        """Per-band c-factor. Mirrors roy_brdf.c_factor.

        `coef` is (6, 3); sza/vza/dphi/theta_s are (H, W) (or scalars).
        Returns (6, H, W), broadcasting the per-band coefficients over the grid.
        """
        kgeo_in = self._kgeo(sza, vza, dphi)
        kvol_in = self._kvol(sza, vza, dphi)

        # Reference geometry: view straight down, sun at theta_s.
        zero = theta_s * 0.0
        kgeo_norm = self._kgeo(theta_s, zero, zero)
        kvol_norm = self._kvol(theta_s, zero, zero)

        # (6,1,1) coefficient columns, broadcast against (H,W) kernels -> (6,H,W).
        f_iso = coef[:, 0].reshape(-1, 1, 1)
        f_geo = coef[:, 1].reshape(-1, 1, 1)
        f_vol = coef[:, 2].reshape(-1, 1, 1)
        numerator = f_iso + f_geo * kgeo_norm + f_vol * kvol_norm
        denominator = f_iso + f_geo * kgeo_in + f_vol * kvol_in
        return numerator / denominator

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (11, H, W) -> (6, H, W)."""
        # Internal computation in float32 (see module docstring).
        x = x.to(self.compute_dtype)

        rho = x[0:6]                  # (6, H, W) reflectance, band order = BAND_ORDER
        sza = x[6]                    # (H, W)
        vza = x[7]
        saa = x[8]
        vaa = x[9]
        theta_s = x[10]

        dphi = saa - vaa
        # Cast coefficients to the compute dtype of the (already cast) input.
        coef = self.coef.to(x.dtype)
        c = self._c_factor(sza, vza, dphi, theta_s, coef)  # (6, H, W)

        scaled = c * rho
        lo = (1.0 - self.clip_factor) * rho
        hi = (1.0 + self.clip_factor) * rho
        # NOT torch.clamp(scaled, lo, hi): the exporter may emit a tensor-bounded `Clip`
        # node, which ONNX Runtime rejects ("min should be a scalar"). Maximum/Minimum
        # export to `Max`/`Min`, which broadcast.
        out = torch.minimum(torch.maximum(scaled, lo), hi)

        # nodata guard: rho <= 0 -> 0
        out = torch.where(rho <= 0, torch.zeros_like(out), out)

        return out.to(self.io_dtype)


def build_torch_model(height: int, width: int, io_dtype: torch.dtype = torch.float64) -> RoyBRDF:
    """Instantiate the module. height/width are not used by torch (shapes are dynamic);
    they matter only at export time, which bakes them into the graph."""
    return RoyBRDF(io_dtype=io_dtype).eval()


def export_onnx(
    path: str = "roy_brdf.onnx",
    height: int = 32,
    width: int = 32,
    onnx_dtype: str = "float64",
    opset: int = 17,
    ir_version: int | None = 10,
) -> str:
    """Export the torch module to ONNX with the layout predict_onnx expects.

    Pins opset and IR version for compatibility with the backend's ONNX Runtime
    (1.16.3 in the documented dependency archive). Torch chooses a modern default
    otherwise, which can emit ops an older runtime does not implement.
    """
    torch_dtype = torch.float64 if onnx_dtype == "float64" else torch.float32
    model = build_torch_model(height, width, io_dtype=torch_dtype)
    dummy = torch.zeros(N_INPUT_CHANNELS, height, width, dtype=torch_dtype)

    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed H/W: predict_onnx requires literal spatial dims
    )

    # Force IR version down for older ONNX Runtime builds if requested.
    if ir_version is not None:
        import onnx

        m = onnx.load(path)
        if m.ir_version > ir_version:
            print(f"  lowering IR version {m.ir_version} -> {ir_version}")
            m.ir_version = ir_version
        onnx.checker.check_model(m)
        onnx.save(m, path)

    return path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _reference(x: np.ndarray, theta_s: float) -> dict[str, np.ndarray]:
    """roy_brdf.py as the ground truth for the same 11-channel stack."""
    import roy_brdf

    rho = {b: x[i] for i, b in enumerate(BAND_ORDER)}
    return roy_brdf.apply_nbar_bands(
        rho,
        dict(sza=x[6], vza=x[7], saa=x[8], vaa=x[9]),
        scene_center_lat=51.22,  # only used via theta_s below; see note
    )


def verify_wrt_reference(height: int = 16, width: int = 16) -> bool:
    """Check the torch module against roy_brdf.py, and report the exported dtype."""
    import roy_brdf

    rng = np.random.default_rng(0)
    x = np.zeros((N_INPUT_CHANNELS, height, width), dtype=np.float64)
    for i in range(len(BAND_ORDER)):
        x[i] = rng.uniform(100, 5000, (height, width))
    x[6] = 36.8562
    x[7] = 6.9308
    x[8] = 159.9136
    x[9] = 107.9809
    theta_s = float(roy_brdf.mean_sun_zenith(51.22))
    x[10] = theta_s

    model = build_torch_model(height, width)
    with torch.no_grad():
        got = model(torch.from_numpy(x)).numpy()

    worst = 0.0
    for i, band in enumerate(BAND_ORDER):
        ref = roy_brdf.nbar(
            x[i], x[6], x[7], x[8] - x[9], roy_brdf.ROY_COEF[band], sza_norm=theta_s
        )
        rel = np.abs(got[i] - ref).max() / max(1e-9, np.abs(ref).max())
        worst = max(worst, rel)
        print(f"  {band}: rel diff vs roy_brdf = {rel:.2e}")

    print(f"  worst relative difference: {worst:.2e} -> {'OK' if worst < 1e-5 else 'FAILED'}")

    # Identity-geometry sanity: no correction needed => output == input.
    # Tolerance must account for the float32 internal compute: input and output are
    # float64, but the arithmetic round-trips through float32, so the result carries
    # float32 precision (~1e-7 relative) regardless of the I/O dtype.
    ident = x.copy()
    ident[6] = 41.0
    ident[7] = 0.0
    ident[8] = 100.0
    ident[9] = 100.0
    ident[10] = 41.0
    with torch.no_grad():
        out_id = model(torch.from_numpy(ident)).numpy()
    dev = np.abs(out_id - ident[0:6]).max()
    scale = np.abs(ident[0:6]).max()
    rel_dev = dev / max(1e-9, scale)
    print(f"  identity geometry deviation: {dev:.2e} (relative {rel_dev:.2e})"
          f" -> {'OK' if rel_dev < 1e-6 else 'FAILED'}")

    return worst < 1e-5 and rel_dev < 1e-6


def verify_exported(path: str, height: int = 32, width: int = 32) -> bool:
    """Load the exported .onnx and compare against roy_brdf.py.

    Also reports the graph's real I/O dtypes: the whole point of the float64 dance is
    that torch's exporter does not quietly downcast to float32.
    """
    import onnx
    import onnxruntime as ort

    import roy_brdf

    m = onnx.load(path)
    in_elem = m.graph.input[0].type.tensor_type.elem_type
    out_elem = m.graph.output[0].type.tensor_type.elem_type
    names = {1: "float32", 11: "float64"}
    print(f"  exported I/O dtype: {names.get(in_elem)} in / {names.get(out_elem)} out")
    print(f"  ir_version={m.ir_version} opset={m.opset_import[0].version}")
    if in_elem != 11 or out_elem != 11:
        print("  FAILED: expected float64 in and out (backend tile celltype)")
        return False

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    x = np.zeros((N_INPUT_CHANNELS, height, width), dtype=np.float64)
    for i in range(len(BAND_ORDER)):
        x[i] = rng.uniform(100, 5000, (height, width))
    x[6], x[7], x[8], x[9] = 36.8562, 6.9308, 159.9136, 107.9809
    theta_s = float(roy_brdf.mean_sun_zenith(51.22))
    x[10] = theta_s

    got: np.ndarray = np.asarray(sess.run(None, {"input": x})[0])
    worst = 0.0
    for i, band in enumerate(BAND_ORDER):
        ref: np.ndarray = np.asarray(
            roy_brdf.nbar(
                x[i], x[6], x[7], x[8] - x[9], roy_brdf.ROY_COEF[band], sza_norm=theta_s
            )
        )
        rel = float(np.abs(got[i] - ref).max() / max(1e-9, np.abs(ref).max()))
        worst = max(worst, rel)
    print(f"  worst relative difference vs roy_brdf: {worst:.2e}"
          f" -> {'OK' if worst < 1e-5 else 'FAILED'}")
    return worst < 1e-5


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true", help="verify the torch module, skip export")
    ap.add_argument("--out", default="roy_brdf_torch.onnx")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print("1. torch module vs roy_brdf.py")
    ok_module = verify_wrt_reference()

    if args.check_only:
        raise SystemExit(0 if ok_module else 1)

    print(f"\n2. exporting to {args.out} ({args.height}x{args.width})")
    export_onnx(args.out, args.height, args.width, opset=args.opset)

    print("\n3. exported graph vs roy_brdf.py")
    ok_export = verify_exported(args.out, args.height, args.width)

    raise SystemExit(0 if (ok_module and ok_export) else 1)
