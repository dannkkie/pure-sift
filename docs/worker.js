import init, { extract_features_png, match_and_stitch } from "./pkg/pure_sift_wasm_demo.js";

let ready = false;

async function ensureInit() {
  if (!ready) {
    await init();
    ready = true;
  }
}

self.onmessage = async (event) => {
  const { id, action, payload } = event.data;
  try {
    await ensureInit();
    const t0 = performance.now();

    if (action === "extract") {
      const bytes = new Uint8Array(payload.bytes);
      const res = extract_features_png(
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
        elapsedMs: Math.round((performance.now() - t0) * 10) / 10,
        pngBuffer: png.buffer,
      };
      res.free();
      self.postMessage({ id, ok: true, result }, [result.pngBuffer]);
    } else if (action === "match_and_stitch") {
      const queryBytes = new Uint8Array(payload.queryBytes);
      const targetBytes = new Uint8Array(payload.targetBytes);
      const res = match_and_stitch(
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
        elapsedMs: Math.round((performance.now() - t0) * 10) / 10,
        matchesPngBuffer: matchesPng.buffer,
        stitchPngBuffer: stitchPng ? stitchPng.buffer : null,
      };
      res.free();
      self.postMessage({ id, ok: true, result }, transfer);
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
