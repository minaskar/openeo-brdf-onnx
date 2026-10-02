"""Build the Roy et al. BRDF c-factor correction as an ONNX model.

This is a direct transliteration of `roy_brdf.py` so the two can be diffed
function by function. It exists to be used with the openEO `predict_onnx`
process, which has a strict contract (see `brdf_onnx.ipynb`):

  - exactly ONE input tensor and ONE output tensor
  - input dtype == output dtype (we use float32)
  - FIXED spatial dimensions H, W (the backend retiles the cube to match)
  - band count must equal input shape[-3]; bands are POSITIONAL, not named

Tensor layouts
--------------
Input  (11, H, W) float32, channels 0..10:
    0  B02      reflectance
    1  B03      reflectance
    2  B04      reflectance
    3  B8A      reflectance
    4  B11      reflectance
    5  B12      reflectance
    6  SZA      sun zenith angle       [degrees]
    7  VZA      view zenith angle      [degrees]
    8  SAA      sun azimuth angle      [degrees]
    9  VAA      view azimuth angle     [degrees]
    10 theta_s  scene reference SZA    [degrees]  (scalar broadcast as a channel)

Output (6, H, W) float32: corrected reflectance, same band order as 0..5.

Channel 10 exists because `predict_onnx` has no `context` parameter, so the
per-scene reference angle cannot be passed any other way.

`theta_s` is a per-scene constant, so it is NOT a model weight we can freeze at
export time (that would require one model per AOI latitude). Feeding it as a
channel keeps a single model valid everywhere.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Band order is part of the model's contract. Changing it silently corrupts results.
BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]
ANGLE_ORDER = ["sunZenithAngles", "viewZenithMean", "sunAzimuthAngles", "viewAzimuthMean"]
N_INPUT_CHANNELS = len(BAND_ORDER) + len(ANGLE_ORDER) + 1  # +1 for theta_s
N_OUTPUT_CHANNELS = len(BAND_ORDER)

# Same constants as roy_brdf.py: Roy et al. (2017) Table 1 / Sen2Like ATBD Table 6.
# [f_iso, f_geo, f_vol]
ROY_COEF = {
    "B02": [0.0774, 0.0079, 0.0372],
    "B03": [0.1306, 0.0178, 0.0580],
    "B04": [0.1690, 0.0227, 0.0574],
    "B8A": [0.3093, 0.0330, 0.1535],
    "B11": [0.3430, 0.0453, 0.1154],
    "B12": [0.2658, 0.0387, 0.0639],
}

CLIP_FACTOR = 0.2  # sen2like clamps the correction to +/-20%
D2R = np.pi / 180.0


def _const(name, value):
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def build_model(height, width, opset=17):
    """Build the ONNX graph. Returns an onnx.ModelProto.

    The graph is a literal transliteration of roy_brdf.py:

        kgeo_li_sparse / kvol_ross_thick  -> kernel subgraphs
        c_factor                          -> ratio
        nbar                              -> clip + nodata Where
    """
    nodes = []
    inits = []

    def add(name, op, inputs, **attrs):
        nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return name

    def add_const(name, value):
        inits.append(_const(name, value))
        return name

    def add_const_i64(name, value):
        inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name))
        return name

    # ---------------------------------------------------------------- constants
    d2r = add_const("d2r", D2R)
    const_1 = add_const("const_1", 1.0)
    const_2 = add_const("const_2", 2.0)
    const_m1 = add_const("const_m1", -1.0)
    const_0 = add_const("const_0", 0.0)
    const_half = add_const("const_half", 0.5)
    const_pi_2 = add_const("const_pi_2", np.pi / 2.0)
    const_4_3pi = add_const("const_4_3pi", 4.0 / (3.0 * np.pi))
    const_1_3 = add_const("const_1_3", 1.0 / 3.0)
    const_inv_pi = add_const("const_inv_pi", 1.0 / np.pi)
    const_clip_lo = add_const("const_clip_lo", 1.0 - CLIP_FACTOR)
    const_clip_hi = add_const("const_clip_hi", 1.0 + CLIP_FACTOR)

    # ------------------------------------------------------- slice the channels
    # Gather each channel then Squeeze the leading axis, giving (H, W) arrays.
    band_ch = {}
    angle_ch = {}
    for i, name in enumerate(BAND_ORDER):
        idx = add_const_i64(f"idx_band_{i}", [i])
        g = add(f"gather_band_{name}", "Gather", ["input", idx], axis=0)
        band_ch[name] = add(f"chan_band_{name}", "Squeeze", [g, add_const_i64(f"ax0_band_{i}", [0])])
    for i, name in enumerate(ANGLE_ORDER):
        j = len(BAND_ORDER) + i
        idx = add_const_i64(f"idx_angle_{i}", [j])
        g = add(f"gather_angle_{name}", "Gather", ["input", idx], axis=0)
        angle_ch[name] = add(f"chan_angle_{name}", "Squeeze", [g, add_const_i64(f"ax0_angle_{i}", [0])])
    idx_theta = add_const_i64("idx_theta", [len(BAND_ORDER) + len(ANGLE_ORDER)])
    theta = add("chan_theta", "Squeeze",
                [add("gather_theta", "Gather", ["input", idx_theta], axis=0),
                 add_const_i64("ax0_theta", [0])])

    sza = angle_ch["sunZenithAngles"]
    vza = angle_ch["viewZenithMean"]
    saa = angle_ch["sunAzimuthAngles"]
    vaa = angle_ch["viewAzimuthMean"]

    # ------------------------------------------------------------ degrees -> rad
    def to_rad(x):
        return add(f"{x}_rad", "Mul", [x, d2r])

    # dphi = saa - vaa, then to radians
    dphi_deg = add("dphi_deg", "Sub", [saa, vaa])
    dphi = to_rad(dphi_deg)

    def kernels(sza_deg, vza_deg, dphi_rad, tag):
        """Return (kgeo, kvol) for the given geometry. Mirrors roy_brdf.py."""
        ts = to_rad(sza_deg)
        tv = to_rad(vza_deg)

        # ---- Li-Sparse (kgeo) ------------------------------------------
        # b_sur_r = 1 -> theta_p = arctan(tan(theta))
        tan_ts = add(f"tan_ts_{tag}", "Tan", [ts])
        tan_tv = add(f"tan_tv_{tag}", "Tan", [tv])
        ts_p = add(f"ts_p_{tag}", "Atan", [tan_ts])
        tv_p = add(f"tv_p_{tag}", "Atan", [tan_tv])

        cos_ts_p = add(f"cos_ts_p_{tag}", "Cos", [ts_p])
        cos_tv_p = add(f"cos_tv_p_{tag}", "Cos", [tv_p])
        sin_ts_p = add(f"sin_ts_p_{tag}", "Sin", [ts_p])
        sin_tv_p = add(f"sin_tv_p_{tag}", "Sin", [tv_p])
        cos_phi = add(f"cos_phi_{tag}", "Cos", [dphi_rad])

        # cos_zeta_p = cos(ts_p)cos(tv_p) + sin(ts_p)sin(tv_p)cos(phi)
        cos_zeta_p = add(f"cos_zeta_p_{tag}", "Add",
                         [add(f"czp_a_{tag}", "Mul", [cos_ts_p, cos_tv_p]),
                          add(f"czp_b_{tag}", "Mul",
                              [add(f"czp_b1_{tag}", "Mul", [sin_ts_p, sin_tv_p]), cos_phi])])

        # D = sqrt(tan(ts_p)^2 + tan(tv_p)^2 - 2 tan(ts_p) tan(tv_p) cos(phi))
        tan_ts_sq = add(f"tan_ts_sq_{tag}", "Mul", [tan_ts, tan_ts])
        tan_tv_sq = add(f"tan_tv_sq_{tag}", "Mul", [tan_tv, tan_tv])
        tan_prod = add(f"tan_prod_{tag}", "Mul", [tan_ts, tan_tv])
        d_sq = add(f"d_sq_{tag}", "Sub",
                   [add(f"d_sq_a_{tag}", "Add", [tan_ts_sq, tan_tv_sq]),
                    add(f"d_sq_b_{tag}", "Mul",
                        [add(f"d_sq_b1_{tag}", "Mul", [const_2, tan_prod]), cos_phi])])
        d = add(f"d_{tag}", "Sqrt", [d_sq])

        sec_ts_p = add(f"sec_ts_p_{tag}", "Reciprocal", [cos_ts_p])
        sec_tv_p = add(f"sec_tv_p_{tag}", "Reciprocal", [cos_tv_p])
        sec_sum = add(f"sec_sum_{tag}", "Add", [sec_ts_p, sec_tv_p])

        # numerator = sqrt(D^2 + (tan(ts_p) tan(tv_p) sin(phi))^2)
        sin_phi = add(f"sin_phi_{tag}", "Sin", [dphi_rad])
        tps = add(f"tps_{tag}", "Mul", [tan_prod, sin_phi])
        numerator = add(f"numerator_{tag}", "Add",
                        [add(f"d_sq2_{tag}", "Mul", [d, d]),
                         add(f"tps_sq_{tag}", "Mul", [tps, tps])])
        numerator = add(f"numerator_sqrt_{tag}", "Sqrt", [numerator])

        # cos_t = 2 * numerator / sec_sum, clamped to [-1, 1] BEFORE sqrt(1-cos^2)
        cos_t = add(f"cos_t_raw_{tag}", "Div",
                    [add(f"cos_t_num_{tag}", "Mul", [const_2, numerator]), sec_sum])
        cos_t = add(f"cos_t_{tag}", "Clip", [cos_t, const_m1, const_1])

        # sin_t = sqrt(1 - cos_t^2)
        sin_t = add(f"sin_t_{tag}", "Sqrt",
                    [add(f"sin_t_arg_{tag}", "Sub",
                         [const_1, add(f"cos_t_sq_{tag}", "Mul", [cos_t, cos_t])])])
        t_ang = add(f"t_ang_{tag}", "Acos", [cos_t])

        # overlap = (1/pi)(t - sin t cos t)(sec_s + sec_v)
        overlap = add(f"overlap_{tag}", "Mul",
                      [add(f"overlap_a_{tag}", "Mul",
                           [const_inv_pi, add(f"overlap_b_{tag}", "Sub",
                                              [t_ang, add(f"overlap_c_{tag}", "Mul", [sin_t, cos_t])])]),
                       sec_sum])

        # kgeo = overlap - sec_s - sec_v + 0.5(1 + cos_zeta_p) sec_s sec_v
        kgeo = add(f"kgeo_{tag}", "Add",
                   [add(f"kgeo_a_{tag}", "Sub", [overlap, sec_sum]),
                    add(f"kgeo_b_{tag}", "Mul",
                        [add(f"kgeo_b1_{tag}", "Mul",
                             [const_half, add(f"kgeo_b2_{tag}", "Add", [const_1, cos_zeta_p])]),
                         add(f"kgeo_b3_{tag}", "Mul", [sec_ts_p, sec_tv_p])])])

        # ---- Ross-Thick (kvol) -----------------------------------------
        # cos_zeta = cos(ts)cos(tv) + sin(ts)sin(tv)cos(phi)   [note: UNprimed angles]
        cos_ts = add(f"cos_ts_{tag}", "Cos", [ts])
        cos_tv = add(f"cos_tv_{tag}", "Cos", [tv])
        sin_ts = add(f"sin_ts_{tag}", "Sin", [ts])
        sin_tv = add(f"sin_tv_{tag}", "Sin", [tv])
        cos_zeta = add(f"cos_zeta_{tag}", "Add",
                       [add(f"cz_a_{tag}", "Mul", [cos_ts, cos_tv]),
                        add(f"cz_b_{tag}", "Mul",
                            [add(f"cz_b1_{tag}", "Mul", [sin_ts, sin_tv]), cos_phi])])
        cos_zeta = add(f"cos_zeta_clip_{tag}", "Clip", [cos_zeta, const_m1, const_1])
        zeta = add(f"zeta_{tag}", "Acos", [cos_zeta])

        # numerator = (pi/2 - zeta) cos(zeta) + sin(zeta)
        kvol_num = add(f"kvol_num_{tag}", "Add",
                       [add(f"kvol_num_a_{tag}", "Mul",
                            [add(f"kvol_num_b_{tag}", "Sub", [const_pi_2, zeta]),
                             add(f"kvol_num_c_{tag}", "Cos", [zeta])]),
                        add(f"kvol_num_d_{tag}", "Sin", [zeta])])
        kvol_den = add(f"kvol_den_{tag}", "Add", [cos_tv, cos_ts])
        kvol = add(f"kvol_{tag}", "Sub",
                   [add(f"kvol_a_{tag}", "Mul",
                        [const_4_3pi, add(f"kvol_b_{tag}", "Div", [kvol_num, kvol_den])]),
                    const_1_3])
        return kgeo, kvol

    # ---- kernels at the actual geometry and at the reference geometry --------
    kgeo_in, kvol_in = kernels(sza, vza, dphi, "in")

    # Reference geometry: vza_norm = 0, dphi_norm = 0, sza_norm = theta_s.
    zero_ch = add("zero_ch", "Mul", [theta, const_0])  # (H,W) zeros, keeps shape
    kgeo_norm, kvol_norm = kernels(theta, zero_ch, zero_ch, "norm")

    # ------------------------------------------------- per-band c and NBAR
    outputs = []
    for i, band in enumerate(BAND_ORDER):
        f_iso, f_geo, f_vol = ROY_COEF[band]
        tag = band
        # numerator = f_iso + f_geo kgeo_norm + f_vol kvol_norm
        num = add(f"num_{tag}", "Add",
                  [add_const(f"fiso_{tag}", f_iso),
                   add(f"num_a_{tag}", "Add",
                       [add(f"num_a1_{tag}", "Mul", [add_const(f"fgeo_{tag}", f_geo), kgeo_norm]),
                        add(f"num_a2_{tag}", "Mul", [add_const(f"fvol_{tag}", f_vol), kvol_norm])])])
        # denominator = f_iso + f_geo kgeo_in + f_vol kvol_in
        den = add(f"den_{tag}", "Add",
                  [add_const(f"fiso_d_{tag}", f_iso),
                   add(f"den_a_{tag}", "Add",
                       [add(f"den_a1_{tag}", "Mul", [add_const(f"fgeo_d_{tag}", f_geo), kgeo_in]),
                        add(f"den_a2_{tag}", "Mul", [add_const(f"fvol_d_{tag}", f_vol), kvol_in])])])
        c = add(f"c_{tag}", "Div", [num, den])

        rho = band_ch[band]
        # out = clip(c*rho, 0.8 rho, 1.2 rho)
        # NB: ONNX Clip only accepts SCALAR min/max, but here both bounds are
        # per-pixel tensors, so express the clamp as Max then Min (they broadcast).
        scaled = add(f"scaled_{tag}", "Mul", [c, rho])
        lo = add(f"lo_{tag}", "Mul", [const_clip_lo, rho])
        hi = add(f"hi_{tag}", "Mul", [const_clip_hi, rho])
        lower = add(f"lower_{tag}", "Max", [scaled, lo])
        clipped = add(f"clipped_{tag}", "Min", [lower, hi])
        # nodata guard: rho <= 0 -> 0
        is_nodata = add(f"is_nodata_{tag}", "LessOrEqual", [rho, const_0])
        masked = add(f"masked_{tag}", "Where", [is_nodata, const_0, clipped])
        outputs.append(masked)

    # ------------------------------------------------------------- stack output
    # (6, H, W): unsqueeze each to (1,H,W) then Concat along axis 0.
    unsq = []
    for i, band in enumerate(BAND_ORDER):
        ax = add_const_i64(f"ax0_out_{i}", [0])
        unsq.append(add(f"out_{band}", "Unsqueeze", [f"masked_{band}", ax]))
    final = add("output", "Concat", unsq, axis=0)

    # --------------------------------------------------------------- assemble
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N_INPUT_CHANNELS, height, width])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N_OUTPUT_CHANNELS, height, width])
    graph = helper.make_graph(nodes, "roy_brdf_cfactor", [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10  # keep within onnxruntime 1.16-compatible range
    onnx.checker.check_model(model)
    return model


def channel_order():
    """The exact channel order the model expects (for documentation/asserts)."""
    return BAND_ORDER + ANGLE_ORDER + ["theta_s"]


if __name__ == "__main__":
    m = build_model(8, 8)
    print("built model, channels:", channel_order())
    print("output shape:", [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])
    import onnxruntime as ort
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    print("onnxruntime loaded OK; outputs:", [o.name for o in s.get_outputs()])
