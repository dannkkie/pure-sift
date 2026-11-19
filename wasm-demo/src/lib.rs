use ndarray::Array2;
use pure_sift::cli::{
    decode_validated_dynamic_image, map_image_prefix, run_visualize_features_rgba,
    validate_match_params, validate_pipeline_params,
};
use pure_sift::{
    SiftFeature, draw_matches_side_by_side, encode_rgba_png, estimate_homography_ransac,
    extract_sift_features, filter_matches_ransac, is_sane_homography_for_dims, luma_to_array2,
    match_descriptors, resolve_safe_alignment_homography, warp_and_stitch,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn simd128_enabled() -> bool {
    cfg!(target_feature = "simd128")
}

#[wasm_bindgen]
pub struct FeaturesResult {
    keypoint_count: usize,
    width: u32,
    height: u32,
    elapsed_ms: f64,
    png: Vec<u8>,
}

#[wasm_bindgen]
impl FeaturesResult {
    #[wasm_bindgen(getter)]
    pub fn keypoint_count(&self) -> usize {
        self.keypoint_count
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    #[wasm_bindgen(getter)]
    pub fn png(&self) -> Vec<u8> {
        self.png.clone()
    }
}

#[wasm_bindgen]
pub struct MatchAndStitchResult {
    query_features: usize,
    target_features: usize,
    raw_matches: usize,
    inliers: usize,
    sim_inliers: usize,
    hom_inliers: usize,
    decode_ms: f64,
    extract_ms: f64,
    match_ms: f64,
    ransac_ms: f64,
    render_ms: f64,
    stitch_ms: f64,
    matches_png: Vec<u8>,
    stitch_png: Option<Vec<u8>>,
    stitch_error: Option<String>,
}

#[wasm_bindgen]
impl MatchAndStitchResult {
    #[wasm_bindgen(getter)]
    pub fn query_features(&self) -> usize {
        self.query_features
    }

    #[wasm_bindgen(getter)]
    pub fn target_features(&self) -> usize {
        self.target_features
    }

    #[wasm_bindgen(getter)]
    pub fn raw_matches(&self) -> usize {
        self.raw_matches
    }

    #[wasm_bindgen(getter)]
    pub fn inliers(&self) -> usize {
        self.inliers
    }

    #[wasm_bindgen(getter)]
    pub fn sim_inliers(&self) -> usize {
        self.sim_inliers
    }

    #[wasm_bindgen(getter)]
    pub fn hom_inliers(&self) -> usize {
        self.hom_inliers
    }

    #[wasm_bindgen(getter)]
    pub fn decode_ms(&self) -> f64 {
        self.decode_ms
    }

    #[wasm_bindgen(getter)]
    pub fn extract_ms(&self) -> f64 {
        self.extract_ms
    }

    #[wasm_bindgen(getter)]
    pub fn match_ms(&self) -> f64 {
        self.match_ms
    }

    #[wasm_bindgen(getter)]
    pub fn ransac_ms(&self) -> f64 {
        self.ransac_ms
    }

    #[wasm_bindgen(getter)]
    pub fn render_ms(&self) -> f64 {
        self.render_ms
    }

    #[wasm_bindgen(getter)]
    pub fn stitch_ms(&self) -> f64 {
        self.stitch_ms
    }

    #[wasm_bindgen(getter)]
    pub fn matches_png(&self) -> Vec<u8> {
        self.matches_png.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn stitch_png(&self) -> Option<Vec<u8>> {
        self.stitch_png.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn stitch_error(&self) -> Option<String> {
        self.stitch_error.clone()
    }
}

#[wasm_bindgen]
pub fn extract_features_png(
    image_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
) -> Result<FeaturesResult, JsValue> {
    let t0 = js_sys::Date::now();
    let (rendered, keypoint_count) =
        run_visualize_features_rgba(image_bytes, octaves, scales, threshold, true)
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    let width = rendered.width();
    let height = rendered.height();
    let png = encode_rgba_png(&rendered).map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    let elapsed_ms = js_sys::Date::now() - t0;
    Ok(FeaturesResult {
        keypoint_count,
        width,
        height,
        elapsed_ms,
        png,
    })
}

#[wasm_bindgen]
pub fn match_and_stitch(
    query_bytes: &[u8],
    target_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
    ratio_threshold: f32,
    ransac_threshold: f32,
) -> Result<MatchAndStitchResult, JsValue> {
    validate_pipeline_params(octaves, scales, threshold)
        .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    validate_match_params(ratio_threshold, ransac_threshold)
        .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;

    let t_decode0 = js_sys::Date::now();
    let query_dyn = decode_validated_dynamic_image(query_bytes, true)
        .map_err(|e| JsValue::from_str(&format!("{:?}", map_image_prefix("Query image", e))))?;
    let target_dyn = decode_validated_dynamic_image(target_bytes, true)
        .map_err(|e| JsValue::from_str(&format!("{:?}", map_image_prefix("Target image", e))))?;

    let q_arr = luma_to_array2(&query_dyn.to_luma8());
    let t_arr = luma_to_array2(&target_dyn.to_luma8());
    let (q_h, q_w) = q_arr.dim();
    let t_extract0 = js_sys::Date::now();
    let decode_ms = t_extract0 - t_decode0;

    let query_features = extract_sift_features(&q_arr, octaves, scales, threshold);
    let target_features = extract_sift_features(&t_arr, octaves, scales, threshold);
    let t_match0 = js_sys::Date::now();
    let extract_ms = t_match0 - t_extract0;

    let raw_matches = match_descriptors(&query_features, &target_features, ratio_threshold);
    let t_ransac0 = js_sys::Date::now();
    let match_ms = t_ransac0 - t_match0;

    let (sim_inliers, sim_tf) = filter_matches_ransac(&raw_matches, ransac_threshold);
    let (hom_inliers, hom_tf) = estimate_homography_ransac(&raw_matches, ransac_threshold);
    let t_render0 = js_sys::Date::now();
    let ransac_ms = t_render0 - t_ransac0;

    let best_inliers = if hom_inliers.len() > sim_inliers.len() {
        &hom_inliers
    } else {
        &sim_inliers
    };

    let matches_canvas = draw_matches_side_by_side(&query_dyn, &target_dyn, best_inliers);
    let matches_png =
        encode_rgba_png(&matches_canvas).map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    let t_stitch0 = js_sys::Date::now();
    let render_ms = t_stitch0 - t_render0;

    let (stitch_png, stitch_error) =
        match resolve_safe_alignment_homography(hom_tf, hom_inliers.len(), sim_tf, q_w, q_h) {
            Some(alignment) => match warp_and_stitch(&query_dyn, &target_dyn, &alignment) {
                Ok(stitched) => match encode_rgba_png(&stitched) {
                    Ok(png) => (Some(png), None),
                    Err(e) => (None, Some(format!("Failed to encode stitch PNG: {e:?}"))),
                },
                Err(e) => (None, Some(format!("Stitch failed: {e:?}"))),
            },
            None => (
                None,
                Some("Not enough geometric inliers to estimate a safe alignment transform.".into()),
            ),
        };
    let stitch_ms = js_sys::Date::now() - t_stitch0;

    Ok(MatchAndStitchResult {
        query_features: query_features.len(),
        target_features: target_features.len(),
        raw_matches: raw_matches.len(),
        inliers: best_inliers.len(),
        sim_inliers: sim_inliers.len(),
        hom_inliers: hom_inliers.len(),
        decode_ms,
        extract_ms,
        match_ms,
        ransac_ms,
        render_ms,
        stitch_ms,
        matches_png,
        stitch_png,
        stitch_error,
    })
}

fn rgba_to_luma_array2(rgba: &[u8], width: u32, height: u32) -> Result<Array2<f32>, JsValue> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| JsValue::from_str("Frame dimensions overflow"))?;
    if rgba.len() != expected || w < 8 || h < 8 {
        return Err(JsValue::from_str(
            "Invalid RGBA buffer length or frame too small",
        ));
    }

    let mut data = Vec::with_capacity(w * h);
    const INV_255: f32 = 1.0 / 255.0;
    for px in rgba.chunks_exact(4) {
        let luma =
            (0.299 * (px[0] as f32) + 0.587 * (px[1] as f32) + 0.114 * (px[2] as f32)) * INV_255;
        data.push(luma);
    }
    Array2::from_shape_vec((h, w), data)
        .map_err(|e| JsValue::from_str(&format!("Array shape error: {e}")))
}

#[wasm_bindgen]
pub struct WebcamTrackResult {
    frame_features: usize,
    raw_matches: usize,
    inliers: usize,
    has_homography: bool,
    corners: Vec<f32>,
    extract_ms: f64,
    match_ms: f64,
    ransac_ms: f64,
}

#[wasm_bindgen]
impl WebcamTrackResult {
    #[wasm_bindgen(getter)]
    pub fn frame_features(&self) -> usize {
        self.frame_features
    }

    #[wasm_bindgen(getter)]
    pub fn raw_matches(&self) -> usize {
        self.raw_matches
    }

    #[wasm_bindgen(getter)]
    pub fn inliers(&self) -> usize {
        self.inliers
    }

    #[wasm_bindgen(getter)]
    pub fn has_homography(&self) -> bool {
        self.has_homography
    }

    #[wasm_bindgen(getter)]
    pub fn corners(&self) -> Vec<f32> {
        self.corners.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn extract_ms(&self) -> f64 {
        self.extract_ms
    }

    #[wasm_bindgen(getter)]
    pub fn match_ms(&self) -> f64 {
        self.match_ms
    }

    #[wasm_bindgen(getter)]
    pub fn ransac_ms(&self) -> f64 {
        self.ransac_ms
    }
}

#[wasm_bindgen]
pub struct WebcamTracker {
    octaves: usize,
    scales: usize,
    threshold: f32,
    ratio: f32,
    ransac_threshold: f32,
    ref_width: usize,
    ref_height: usize,
    ref_features: Vec<SiftFeature>,
}

#[wasm_bindgen]
impl WebcamTracker {
    #[wasm_bindgen(constructor)]
    pub fn new(
        octaves: usize,
        scales: usize,
        threshold: f32,
        ratio: f32,
        ransac_threshold: f32,
    ) -> Result<WebcamTracker, JsValue> {
        validate_pipeline_params(octaves, scales, threshold)
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
        validate_match_params(ratio, ransac_threshold)
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
        Ok(Self {
            octaves,
            scales,
            threshold,
            ratio,
            ransac_threshold,
            ref_width: 0,
            ref_height: 0,
            ref_features: Vec::new(),
        })
    }

    pub fn set_reference_rgba(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<usize, JsValue> {
        let arr = rgba_to_luma_array2(rgba, width, height)?;
        self.ref_width = width as usize;
        self.ref_height = height as usize;
        self.ref_features = extract_sift_features(&arr, self.octaves, self.scales, self.threshold);
        Ok(self.ref_features.len())
    }

    pub fn track_frame_rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<WebcamTrackResult, JsValue> {
        if self.ref_features.is_empty() {
            return Err(JsValue::from_str("No reference frame captured yet"));
        }

        let arr = rgba_to_luma_array2(rgba, width, height)?;
        let t0 = js_sys::Date::now();
        let frame_features = extract_sift_features(&arr, self.octaves, self.scales, self.threshold);
        let t1 = js_sys::Date::now();
        let extract_ms = t1 - t0;

        let raw_matches = match_descriptors(&self.ref_features, &frame_features, self.ratio);
        let t2 = js_sys::Date::now();
        let match_ms = t2 - t1;

        let (hom_inliers, hom_opt) =
            estimate_homography_ransac(&raw_matches, self.ransac_threshold);
        let ransac_ms = js_sys::Date::now() - t2;

        let mut corners = Vec::new();
        let mut has_homography = false;
        if let Some(h) = hom_opt
            && hom_inliers.len() >= 6
            && is_sane_homography_for_dims(&h, self.ref_width, self.ref_height)
        {
            let rw = self.ref_width as f32;
            let rh = self.ref_height as f32;
            let quad = [(0.0, 0.0), (rw, 0.0), (rw, rh), (0.0, rh)];
            let mut valid = true;
            for &(x, y) in &quad {
                if let Some((px, py)) = h.project(x, y) {
                    if px.is_finite() && py.is_finite() {
                        corners.push(px);
                        corners.push(py);
                    } else {
                        valid = false;
                        break;
                    }
                } else {
                    valid = false;
                    break;
                }
            }
            if valid && corners.len() == 8 {
                has_homography = true;
            } else {
                corners.clear();
            }
        }

        Ok(WebcamTrackResult {
            frame_features: frame_features.len(),
            raw_matches: raw_matches.len(),
            inliers: hom_inliers.len(),
            has_homography,
            corners,
            extract_ms,
            match_ms,
            ransac_ms,
        })
    }
}
