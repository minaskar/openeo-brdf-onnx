# Roy et al. BRDF c-factor — ONNX model

Analytical BRDF normalization model for openEO's `predict_onnx` process.
Context: [ESA-APEx/apex_algorithms#687](https://github.com/ESA-APEx/apex_algorithms/issues/687)

`roy_brdf.onnx` is a frozen (weights-free) model of the Roy et al. (2016/2017) c-factor
method, transliterated from the sen2like reference implementation.

## Contract

- Input:  `(11, 32, 32)` float32
- Output: `(6, 32, 32)` float32
- Fixed spatial size 32x32 (the backend retiles the cube to match)

Input channels, in order:

| # | channel | meaning |
|---|---------|---------|
| 0-5 | B02 B03 B04 B8A B11 B12 | surface reflectance |
| 6 | sunZenithAngles | sun zenith [deg] |
| 7 | viewZenithMean | view zenith [deg] |
| 8 | sunAzimuthAngles | sun azimuth [deg] |
| 9 | viewAzimuthMean | view azimuth [deg] |
| 10 | theta_s | per-scene reference SZA [deg] |

Output channels are the corrected reflectance for B02 B03 B04 B8A B11 B12, in that order.

`brdf_onnx_model.py` builds the model (`build_model(height, width)`).

## Usage

```python
nbar = cube_11.process("predict_onnx",
                       data=cube_11,
                       model="https://raw.githubusercontent.com/minaskar/openeo-brdf-onnx/main/roy_brdf.onnx")
```

## Notes

- Band order is positional; reordering silently corrupts results.
- `theta_s` is an input channel because `predict_onnx` has no `context` parameter.
- Non-Roy bands (no MODIS analogue) are not covered.
