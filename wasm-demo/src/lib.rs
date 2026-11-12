use pure_sift::{
    decode_validated_dynamic_image, draw_matches_side_by_side, encode_rgba_png,
    estimate_homography_ransac, extract_sift_features, filter_matches_ransac, luma_to_array2,
    map_image_prefix, match_descriptors, resolve_safe_alignment_homography,
    run_visualize_features_rgba, validate_match_params, validate_pipeline_params, warp_and_stitch,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FeaturesResult {
    keypoint_count: usize,
    width: u32,
    height: u32,
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
    let (rendered, keypoint_count) =
        run_visualize_features_rgba(image_bytes, octaves, scales, threshold, true)
            .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    let width = rendered.width();
    let height = rendered.height();
    let png = encode_rgba_png(&rendered).map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    Ok(FeaturesResult {
        keypoint_count,
        width,
        height,
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

    let query_dyn = decode_validated_dynamic_image(query_bytes, true)
        .map_err(|e| JsValue::from_str(&format!("{:?}", map_image_prefix("Query image", e))))?;
    let target_dyn = decode_validated_dynamic_image(target_bytes, true)
        .map_err(|e| JsValue::from_str(&format!("{:?}", map_image_prefix("Target image", e))))?;

    let q_arr = luma_to_array2(&query_dyn.to_luma8());
    let t_arr = luma_to_array2(&target_dyn.to_luma8());
    let (q_h, q_w) = q_arr.dim();

    let query_features = extract_sift_features(&q_arr, octaves, scales, threshold);
    let target_features = extract_sift_features(&t_arr, octaves, scales, threshold);

    let raw_matches = match_descriptors(&query_features, &target_features, ratio_threshold);
    let (sim_inliers, sim_tf) = filter_matches_ransac(&raw_matches, ransac_threshold);
    let (hom_inliers, hom_tf) = estimate_homography_ransac(&raw_matches, ransac_threshold);

    let best_inliers = if hom_inliers.len() > sim_inliers.len() {
        &hom_inliers
    } else {
        &sim_inliers
    };

    let matches_canvas = draw_matches_side_by_side(&query_dyn, &target_dyn, best_inliers);
    let matches_png =
        encode_rgba_png(&matches_canvas).map_err(|e| JsValue::from_str(&format!("{e:?}")))?;

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

    Ok(MatchAndStitchResult {
        query_features: query_features.len(),
        target_features: target_features.len(),
        raw_matches: raw_matches.len(),
        inliers: best_inliers.len(),
        sim_inliers: sim_inliers.len(),
        hom_inliers: hom_inliers.len(),
        matches_png,
        stitch_png,
        stitch_error,
    })
}
