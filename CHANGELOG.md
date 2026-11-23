# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-23

### Added
- **Multi-view geometry (`src/geometry.rs`):** Normalized 7-point & 8-point Fundamental matrix solvers, Sampson-distance LO-RANSAC (`estimate_fundamental_ransac`), calibrated Essential matrix estimation (`estimate_essential_ransac`), linear DLT triangulation (`triangulate_point_linear`), 4-way cheirality-checked relative pose recovery (`recover_pose`), and Torr GRIC Homography-vs-Fundamental model selection (`select_model_gric`), exposed in `match` JSON output.
- **Multi-image panorama stitching (`src/stitch.rs`):** Pairwise match-graph construction, Maximum-Spanning-Tree homography chaining (`stitch_panorama`), cylindrical projection (`warp_image_cylindrical`), Brown & Lowe regularized exposure gain compensation (`solve_exposure_gains`), tiled feather blending, Burt-Adelson multi-band Laplacian pyramid blending, and the `pure-sift panorama` CLI subcommand.
- **Ergonomic OO & builder API:** `SiftConfig`, `Sift`, `Matcher`, `MatchOptions` (with mutual-nearest-neighbor cross-check), `RansacOptions` (with custom deterministic RNG seeds via `.seed(u64)`, `filter_matches_ransac_with_seed`, `estimate_homography_ransac_with_seed`), `Homography`, and explicit `Pyramid` / `Octave` scale-space types.
- **RootSIFT & compact descriptors:** `DescriptorKind::RootSift` normalization and 4x-smaller `u8` descriptor quantization (`quantize_descriptor_u8`, `dequantize_descriptor_u8`).
- **Scalable binary catalog:** Versioned binary `.bin` catalog format with atomic writes and JSON fallback, `16x8`-bit Multi-Index Hashing (MIH) search, `O(1)` ID lookup/removal, and `DELETE /api/v1/catalog/{id}`.
- **CLI CSV output:** `--format <json|csv>` on `extract` and `match`.
- **WASM SIMD128 & live webcam tracking:** Dual scalar + `+simd128` WASM Pages bundles with runtime `WebAssembly.validate` fallback, per-stage timing breakdown, and stateful `WebcamTracker` in the browser demo.
- **Benchmarks & QA infrastructure:** Parallelised per-slice extrema scanning behind the optional `rayon` feature, megapixel Criterion stage benchmarks (`0.3MP`, `2MP`, `10MP`), reproducible HPatches vs. OpenCV SIFT harness (`tools/hpatches/`), MSRV (`1.88`) cross-platform CI matrix, `cargo-deny`, `cargo-llvm-cov`, nightly `cargo-fuzz` targets, and tag-driven release workflow.

### Changed (Breaking)
- Made `server` a non-default Cargo feature (`default = ["rayon"]`), so `cargo add pure-sift` no longer compiles `axum`/`tokio`/`tracing`. Run the HTTP workbench with `cargo run --release --features server -- serve`.
- Scoped `src/lib.rs` crate-root exports to algorithm modules (`catalog`, `error`, `geometry`, `pipeline`, `stitch`, `visualize`), moving CLI parsers/handlers behind `#[doc(hidden)] pub mod cli`.
- Replaced string-only `VisionError` variants with `thiserror` preserving structured `#[source]` causes (`ImageDecode`, `Io`, `Json`, `Runtime`).
- Enforced `#![warn(missing_docs)]` across the public library API and renamed the embedded web UI title/banner from `Native Vision Engine` to `pure-sift`.
- Default SIFT scale-space extraction now builds `scales + 3` Gaussians per octave with uncapped Gaussian kernels, octave-coordinate orientation/descriptor sampling, and 2x upsampled first octave (`BASE_SIGMA = 1.6`, `ASSUMED_INPUT_SIGMA = 0.5`).

### Fixed
- Lowe (2004) scale-space pyramid construction (`scales + 3` Gaussians / `scales + 2` DoG slices per octave) and octave-coordinate descriptor/orientation sampling for true multi-octave scale invariance.
- Upgraded RANSAC with adaptive stopping, LO-RANSAC refinement, collinearity/area degeneracy filtering, and `f64` Jacobi null-space Normalized DLT.
- Removed silent `4096`-descriptor truncation in `match_descriptors` and trimmed default `image` crate features to PNG + JPEG (`extra-formats` for BMP/GIF/TIFF/WebP).
- Hardened HTTP server with loopback default bind (`127.0.0.1`), optional `API_TOKEN` bearer auth, concurrency semaphore, request timeouts, and `tracing` instrumentation.

## [0.1.0] - 2026-11-12

### Added
- Initial pure-Rust SIFT feature extractor (`#![forbid(unsafe_code)]`), RANSAC similarity & homography estimation, feather-blended alignment, CLI, HTTP workbench, and WASM demo.
