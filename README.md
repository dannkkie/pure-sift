# pure-sift

[![CI](https://github.com/dannkkie/pure-sift/actions/workflows/ci.yml/badge.svg)](https://github.com/dannkkie/pure-sift/actions/workflows/ci.yml)
[![Live WASM Demo](https://img.shields.io/badge/demo-Live%20WASM%20in%20Browser-38bdf8.svg)](https://dannkkie.github.io/pure-sift/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust: #![forbid(unsafe_code)]](https://img.shields.io/badge/rust-%23!%5Bforbid(unsafe__code)%5D-success.svg)](src/lib.rs)

> **[Try the Live In-Browser WebAssembly Demo →](https://dannkkie.github.io/pure-sift/)** — drag and drop two images to run SIFT feature extraction, RANSAC matching, and homography stitching 100% locally in your browser (no Rust install or server upload required).

A **100% pure-Rust**, zero-`unsafe` (`#![forbid(unsafe_code)]`) computer vision engine, CLI, visual catalog indexer, and Axum HTTP workbench implementing **SIFT** (Lowe 2004, expired patent `US6711293B1`), **2D Similarity RANSAC**, and **4-Point DLT Planar Homography RANSAC**.

Zero C/C++ FFI, zero OpenCV/CMake/LLVM build dependencies — compiles cleanly to static binaries, Docker sidecars, mobile targets, and WebAssembly (`--no-default-features` omits `axum`/`tokio`).

![SIFT + RANSAC Inlier Correspondences](https://raw.githubusercontent.com/dannkkie/pure-sift/main/docs/matches_demo.jpg)
*636 geometric RANSAC inliers matched across rotated, scaled aerial tiles (`pure-sift visualize-matches`).*

![Homography-Warped Feather-Blended Mosaic](https://raw.githubusercontent.com/dannkkie/pure-sift/main/docs/stitch_demo.jpg)
*Planar 3x3 Homography alignment and feather-blended stitch (`pure-sift align`).*

---

## Core Capabilities

* **Scale-Space DoG Keypoints ([`src/pipeline.rs`](src/pipeline.rs)):** 2x bilinearly upsampled first octave (`DEFAULT_UPSAMPLE_FIRST_OCTAVE = true`) with `BASE_SIGMA = 1.6` and assumed camera blur `ASSUMED_INPUT_SIGMA = 0.5` (`--no-upsample` to disable), separable Gaussian pyramid (`scales + 3` Gaussians and `scales + 2` DoG slices per octave), Hessian edge filtering, and 3D Taylor sub-pixel / sub-scale extremum refinement (`< 0.12 px` localization on smooth Gaussian blobs).
* **Canonical Orientation & 128-D SIFT Descriptors:** `>= 80%` secondary orientation peak splitting + Rayon-parallelized 128-D descriptor extraction.
* **Geometric Verification & Alignment:** Lowe ratio matching, 2D Similarity RANSAC, horizon-guarded 4-point DLT 3x3 Homography estimation (`estimate_homography_ransac`), and bilinear inverse-mapped feather-blended stitching ([`src/visualize.rs`](src/visualize.rs)).
* **Visual Catalog Search ([`src/catalog.rs`](src/catalog.rs)):** 128-bit `XOR + POPCNT` coarse Hamming shortlisting followed by exact 128-D L2 + geometric RANSAC verification.
* **CLI + Optional Single-Binary Web UI ([`src/main.rs`](src/main.rs), [`src/web_ui.html`](src/web_ui.html)):** Drop-in Axum API and embedded 3-tab browser workbench (`Keypoints & Descriptors`, `Match & Stitch Alignment`, `Visual Catalog Search`).

---

## Quick Start

### Use as a Rust Library

Add `pure-sift` to your `Cargo.toml`. Default features enable only `rayon` (zero `axum`/`tokio` HTTP dependencies); pass `default-features = false` for single-threaded WASM or mobile builds:

```toml
[dependencies]
pure-sift = "0.2"
image = "0.25"
```

```rust
use image::ImageReader;
use pure_sift::{
    Homography, MatchOptions, Matcher, RansacOptions, Sift, SiftConfig, luma_to_array2,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q_luma = ImageReader::open("query.jpg")?.decode()?.into_luma8();
    let t_luma = ImageReader::open("target.jpg")?.decode()?.into_luma8();

    let sift = Sift::new(SiftConfig::default().octaves(4).contrast_threshold(0.015))?;
    let q_feats = sift.detect_and_compute(&luma_to_array2(&q_luma));
    let t_feats = sift.detect_and_compute(&luma_to_array2(&t_luma));

    let matches = Matcher::new(MatchOptions::default())?.match_(&q_feats, &t_feats);
    let (inliers, homography) = Homography::estimate(&matches, RansacOptions::default());

    println!("RANSAC Homography inliers: {} / {}", inliers.len(), matches.len());
    if let Some(h) = homography {
        println!("3x3 Homography matrix: {:?}", h.matrix);
    }
    Ok(())
}
```

### Run the Embedded Web UI & HTTP Microservice
```bash
cargo run --release --features server -- serve --bind 127.0.0.1:3000
# Open http://127.0.0.1:3000/ in your browser
```

**Threat Model (Local Tool vs. Exposed Service):**
- By default, `pure-sift serve` binds strictly to loopback (`127.0.0.1:3000`) for local single-user use.
- If exposing the server beyond localhost (`--bind 0.0.0.0:3000` or `BIND_ADDR`), set `API_TOKEN=<secret>` to require `Authorization: Bearer <secret>` on all `/api/*` endpoints, place it behind a TLS-terminating reverse proxy, and tune `MAX_BODY_MB` (default `16`), `MAX_CONCURRENT_JOBS` (default `4`), and `REQUEST_TIMEOUT_SECS` (default `30`).

### Use as a CLI Tool
```bash
# Build release binary
cargo build --release

# Extract DoG keypoints & 128-D SIFT descriptors (JSON or flat CSV)
./target/release/pure-sift extract image.jpg --descriptors --pretty -o features.json
./target/release/pure-sift extract image.jpg --descriptors --format csv -o features.csv

# Match features & estimate 2D Similarity + 3x3 Homography transforms
./target/release/pure-sift match query.jpg target.jpg --pretty -o matches.json
./target/release/pure-sift match query.jpg target.jpg --format csv -o matches.csv

# Render keypoint overlay or side-by-side inlier matches
./target/release/pure-sift visualize image.jpg -o keypoints.png
./target/release/pure-sift visualize-matches query.jpg target.jpg -o matches.png

# Warp & feather-blend query onto target via Homography
./target/release/pure-sift align query.jpg target.jpg -o aligned.png

# Multi-image panorama stitching with exposure compensation & multi-band blending
./target/release/pure-sift panorama img1.jpg img2.jpg img3.jpg -o panorama.png

# Build and query a 128-bit Hamming + RANSAC visual catalog
./target/release/pure-sift catalog-build ./photos -o catalog.json
./target/release/pure-sift catalog-search catalog.json probe.jpg --pretty
```

**CSV Output Headers (`--format csv`):**
- `extract`: `x,y,scale,orientation,response` (`orientation` is empty when `--descriptors` is omitted)
- `match`: `query_index,target_index,query_x,query_y,target_x,target_y,distance,ratio,dx,dy`

---

## HTTP API Endpoints

* `GET /` — Embedded interactive HTML/JS SIFT & RANSAC workbench
* `GET /health` — JSON readiness probe
* `POST /api/v1/process-features` — Extract keypoints (and optional 128-D descriptors)
* `POST /api/v1/match-features` — Match two images + Similarity & Homography transforms
* `POST /api/v1/visualize-features` — Keypoint scale/orientation overlay PNG
* `POST /api/v1/align-images` — Stitched homography warp (`mode=stitch`) or match lines (`mode=matches`) PNG
* `GET /api/v1/catalog` | `POST /api/v1/catalog/index` | `DELETE /api/v1/catalog/{id}` | `POST /api/v1/catalog/search`

---

## Benchmarks & Specification

* Mathematical formulation & implementation reference: [`patent_analysis.md`](patent_analysis.md)
* Reproducible HPatches vs. OpenCV SIFT benchmark harness: [`tools/hpatches/README.md`](tools/hpatches/README.md)

```bash
cargo build --release
pip install -r tools/hpatches/requirements.txt
python tools/hpatches/run_hpatches.py --download --output docs/benchmarks/hpatches_results.json
```

**Synthetic Self-Test Harness Output** ([`docs/benchmarks/synthetic_self_test.json`](docs/benchmarks/synthetic_self_test.json), generated via `--self-test` on viewpoint + illumination warps of `docs/samples/aerial_west.jpg`; install `opencv-python-headless` and pass `--download` for full HPatches vs. OpenCV numbers):

| Engine | Split | Pairs | Repeat@3px | MatchScore@3px | MMA@1px | MMA@3px | MMA@5px | HomAUC@3/5/10px | Mean ms |
|---|---|---|---|---|---|---|---|---|---|
| `pure-sift` | viewpoint | 3 | 84.9% | 40.5% | 97.9% | 99.1% | 99.2% | 60.0% / 76.0% / 88.0% | 198.3 ms |
| `pure-sift` | illumination | 3 | 93.4% | 54.7% | 99.2% | 99.8% | 99.8% | 99.7% / 99.8% / 99.9% | 221.7 ms |
| `pure-sift` | overall | 6 | 89.2% | 47.6% | 98.6% | 99.4% | 99.5% | 79.9% / 87.9% / 94.0% | 210.0 ms |

---

## Differences from Lowe (2004) / OpenCV SIFT

`pure-sift` `0.2` implements Lowe's (2004) scale-space formulation (`scales + 3` Gaussians and `scales + 2` DoG slices per octave, uncapped $\lceil 3\sigma \rceil$ separable kernels, octave-coordinate orientation/descriptor sampling, $2\times$ bilinearly upsampled first octave with $\sigma_0 = 1.6$ and $\sigma_{\text{in}} = 0.5$, and unbounded `match_descriptors` with optional mutual-NN cross-check). The remaining intentional design choices are documented below:

| Stage / Parameter | Lowe (2004) / OpenCV SIFT | `pure-sift` Behavior | Rationale |
|---|---|---|---|
| **3D Taylor extremum refinement** | Iterative refinement (up to 5 steps) re-centering the sample grid when $\|\delta x\|, \|\delta y\|, \|\delta s\| > 0.5$. | Single-step 3D Taylor solve in `refine_extremum_3d` accepting offsets in `[-0.5, 0.5]` with a 2D spatial quadratic fallback when the 3D Hessian is near-singular. | Intentional (deterministic constant-time pass) |
| **Orientation histogram smoothing & peak cap** | $6$ passes of circular $[1/3, 1/3, 1/3]$ box smoothing; no explicit cap on $\ge 80\%$ secondary peaks. | Single circular $[1/3, 1/3, 1/3]$ pass and `MAX_ORIENTATIONS_PER_KEYPOINT = 4` sorted by peak magnitude. | Intentional (bounds worst-case descriptor fan-out) |
| **Feature count cap (`MAX_FEATURES_PER_IMAGE`)** | Optional `nfeatures` cap (default `0` / unlimited in OpenCV). | `extract_sift_features` caps emitted features per image at `MAX_FEATURES_PER_IMAGE = 4096` sorted by `\|response\|`; `match_descriptors` does not truncate inputs. | Intentional (bounds memory/latency on large images) |
| **Convolution boundary handling** | Replicate / reflect border padding. | Clamped-to-edge indexing (`saturating_sub` / `.min(width - 1)`) in `gaussian_blur`. | Intentional (equivalent to border replication) |

---

## License

Dual-licensed under either of:

* Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
* MIT License ([`LICENSE-MIT`](LICENSE-MIT))

at your option.
