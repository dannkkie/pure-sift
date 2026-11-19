# HPatches Benchmark Harness

Reproducible evaluation harness comparing `pure-sift` against OpenCV `cv2.SIFT_create()` on [HPatches](https://github.com/hpatches/hpatches-dataset) sequences.

## Metrics Reported
- **Repeatability @ 3 px**
- **Matching Score @ 3 px**
- **Mean Matching Accuracy (MMA) @ 1 px / 3 px / 5 px**
- **Homography AUC @ 3 px / 5 px / 10 px** (mean 4-corner projection error against HPatches ground-truth `H_1_k`)
- **Split breakdown**: `viewpoint`, `illumination`, and `overall`

## Run on Full HPatches

```bash
cargo build --release
pip install -r tools/hpatches/requirements.txt
python tools/hpatches/run_hpatches.py --download --output docs/benchmarks/hpatches_results.json
```

If you already downloaded HPatches locally, pass `--dataset-dir /path/to/hpatches-sequences-release`. Downloaded sequences are stored under `.cache/hpatches/` (git-ignored).

## Quick Offline Smoke Test (`--self-test`)

```bash
cargo build --release
python tools/hpatches/run_hpatches.py --self-test --output docs/benchmarks/synthetic_self_test.json
```

Generates deterministic viewpoint and illumination pairs from `docs/samples/aerial_west.jpg` with exact ground-truth homographies and runs the full metrics pipeline offline.
