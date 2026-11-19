// Minimal WebAssembly module containing a `v128.const` / `i32x4.splat` instruction to probe SIMD128 support.
const SIMD_PROBE_WASM = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
  0x03, 0x02, 0x01, 0x00,
  0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b,
]);

let wasmMod = null;
let simdActive = false;
let tracker = null;

function supportsSimd128() {
  try {
    return typeof WebAssembly !== "undefined" && WebAssembly.validate(SIMD_PROBE_WASM);
  } catch {
    return false;
  }
}

async function ensureInit() {
  if (wasmMod) return wasmMod;

  if (supportsSimd128()) {
    try {
      const simdMod = await import("./pkg-simd/pure_sift_wasm_demo.js");
      await simdMod.default();
      wasmMod = simdMod;
      simdActive = simdMod.simd128_enabled();
      return wasmMod;
    } catch {
      // Fall back to scalar WASM bundle (e.g., local dev without pkg-simd or non-SIMD build)
    }
  }

  const scalarMod = await import("./pkg/pure_sift_wasm_demo.js");
  await scalarMod.default();
  wasmMod = scalarMod;
  simdActive = scalarMod.simd128_enabled();
  return wasmMod;
}

const round1 = (v) => Math.round(v * 10) / 10;

self.onmessage = async (event) => {
  const { id, action, payload } = event.data;
  try {
    const mod = await ensureInit();
    const t0 = performance.now();

    if (action === "info") {
      self.postMessage({ id, ok: true, result: { simdEnabled: simdActive } });
    } else if (action === "extract") {
      const bytes = new Uint8Array(payload.bytes);
      const res = mod.extract_features_png(
        bytes,
        payload.octaves,
        payload.scales,
        payload.threshold
      );
      const png = res.png;
      const result = {
        keypointCount: res.keypoint_count,
        width: res.width,
        height: res.height,
        elapsedMs: round1(res.elapsed_ms || performance.now() - t0),
        simdEnabled: simdActive,
        pngBuffer: png.buffer,
      };
      res.free();
      self.postMessage({ id, ok: true, result }, [result.pngBuffer]);
    } else if (action === "match_and_stitch") {
      const queryBytes = new Uint8Array(payload.queryBytes);
      const targetBytes = new Uint8Array(payload.targetBytes);
      const res = mod.match_and_stitch(
        queryBytes,
        targetBytes,
        payload.octaves,
        payload.scales,
        payload.threshold,
        payload.ratio,
        payload.ransacThreshold
      );
      const matchesPng = res.matches_png;
      const stitchPng = res.stitch_png;
      const transfer = [matchesPng.buffer];
      if (stitchPng) {
        transfer.push(stitchPng.buffer);
      }
      const result = {
        queryFeatures: res.query_features,
        targetFeatures: res.target_features,
        rawMatches: res.raw_matches,
        inliers: res.inliers,
        simInliers: res.sim_inliers,
        homInliers: res.hom_inliers,
        stitchError: res.stitch_error ?? null,
        elapsedMs: round1(performance.now() - t0),
        simdEnabled: simdActive,
        timings: {
          decodeMs: round1(res.decode_ms),
          extractMs: round1(res.extract_ms),
          matchMs: round1(res.match_ms),
          ransacMs: round1(res.ransac_ms),
          renderMs: round1(res.render_ms),
          stitchMs: round1(res.stitch_ms),
        },
        matchesPngBuffer: matchesPng.buffer,
        stitchPngBuffer: stitchPng ? stitchPng.buffer : null,
      };
      res.free();
      self.postMessage({ id, ok: true, result }, transfer);
    } else if (action === "webcam_set_ref") {
      if (tracker) {
        tracker.free();
      }
      tracker = new mod.WebcamTracker(
        payload.octaves,
        payload.scales,
        payload.threshold,
        payload.ratio,
        payload.ransacThreshold
      );
      const rgba = new Uint8Array(payload.rgbaBuffer);
      const refFeatures = tracker.set_reference_rgba(rgba, payload.width, payload.height);
      self.postMessage({
        id,
        ok: true,
        result: {
          refFeatures,
          elapsedMs: round1(performance.now() - t0),
          simdEnabled: simdActive,
        },
      });
    } else if (action === "webcam_track") {
      if (!tracker) {
        throw new Error("Reference frame not initialized");
      }
      const rgba = new Uint8Array(payload.rgbaBuffer);
      const res = tracker.track_frame_rgba(rgba, payload.width, payload.height);
      const result = {
        frameFeatures: res.frame_features,
        rawMatches: res.raw_matches,
        inliers: res.inliers,
        hasHomography: res.has_homography,
        corners: Array.from(res.corners),
        extractMs: round1(res.extract_ms),
        matchMs: round1(res.match_ms),
        ransacMs: round1(res.ransac_ms),
        totalMs: round1(performance.now() - t0),
      };
      res.free();
      self.postMessage({ id, ok: true, result });
    } else {
      throw new Error(`Unknown worker action: ${action}`);
    }
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
