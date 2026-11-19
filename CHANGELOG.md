# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `SiftConfig`, `Sift`, `Matcher`, `RansacOptions`, and `Homography` builder/object-oriented APIs.
- Explicit `Pyramid` and `Octave` scale-space types.
- `RootSIFT` normalization (`DescriptorKind::RootSift`) and compact `u8` descriptor quantization.
- `MatchOptions` with symmetric mutual-nearest-neighbor cross-check.
- LO-RANSAC with adaptive stopping, degeneracy filtering, and `f64` Jacobi DLT.
- Versioned binary catalog format (`.bin`) with atomic writes, JSON fallback, `16x8`-bit Multi-Index Hashing (MIH) search, `O(1)` ID index, and `DELETE /api/v1/catalog/{id}`.
- MSRV (`1.88`), `cargo-deny` CI checks, CI-built GitHub Pages WASM demo, and tag-driven release workflow.

### Fixed
- Lowe (2004) scale-space pyramid generation (`scales + 3` Gaussians per octave) and octave-coordinate descriptor/orientation sampling.
- HTTP server hardening: loopback default bind (`127.0.0.1`), optional `API_TOKEN` bearer auth, concurrency limits, and `tracing` instrumentation.

## [0.1.0] - 2026-11-12

### Added
- Initial pure-Rust SIFT feature extractor (`#![forbid(unsafe_code)]`), RANSAC similarity & homography estimation, feather-blended alignment, CLI, HTTP workbench, and WASM demo.
