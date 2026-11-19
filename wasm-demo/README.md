# pure-sift WASM Demo

This crate builds the client-side WebAssembly demo served from `docs/`.

## Building Locally

### Scalar fallback (`docs/pkg`)
```bash
cargo build --manifest-path wasm-demo/Cargo.toml --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir docs/pkg wasm-demo/target/wasm32-unknown-unknown/release/pure_sift_wasm_demo.wasm
```

### SIMD128 bundle (`docs/pkg-simd`)
```bash
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --manifest-path wasm-demo/Cargo.toml --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir docs/pkg-simd wasm-demo/target/wasm32-unknown-unknown/release/pure_sift_wasm_demo.wasm
```

`docs/worker.js` probes WebAssembly SIMD128 support at runtime via `WebAssembly.validate(...)`, loads `./pkg-simd/pure_sift_wasm_demo.js` when supported, and falls back to `./pkg/pure_sift_wasm_demo.js` in older browsers.

## Note on `wasm-bindgen-rayon` (Multi-Threaded WASM) & GitHub Pages
Enabling `SharedArrayBuffer` threads (`wasm-bindgen-rayon`) requires cross-origin isolation headers on every HTML/JS response:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

GitHub Pages does not allow setting custom HTTP response headers natively. Because of that platform constraint, the default Pages deployment ships single-threaded **SIMD128** (with scalar fallback). To enable multi-threaded WASM when self-hosting behind NGINX/Cloudflare or on Pages via a Service Worker (`coi-serviceworker`), configure those two COOP/COEP headers and enable the `rayon` feature + `-C target-feature=+atomics,+bulk-memory,+mutable-globals`.
