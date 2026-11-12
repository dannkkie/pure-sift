#![forbid(unsafe_code)]

use std::{
    fmt, fs,
    io::{self, Cursor, Write},
    path::PathBuf,
};

use image::{DynamicImage, ImageFormat, ImageReader, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    catalog::{CatalogSearchHit, DEFAULT_CATALOG_SHORTLIST, DEFAULT_CATALOG_TOP_K, FeatureCatalog},
    pipeline::{
        DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD,
        FeatureMatch, Homography2D, Keypoint, SiftFeature, SimilarityTransform2D, TrackingSummary,
        compute_sift_descriptors, estimate_homography_ransac, extract_extrema,
        extract_sift_features, filter_matches_ransac, generate_dog_space, luma_to_array2,
        match_descriptors, summarize_tracking,
    },
    visualize::{
        draw_keypoints_overlay, draw_matches_side_by_side, resolve_safe_alignment_homography,
        warp_and_stitch,
    },
};

/// Default number of scale-space octaves.
pub const DEFAULT_OCTAVES: usize = 3;
pub const MIN_OCTAVES: usize = 1;
pub const MAX_OCTAVES: usize = 8;

/// Default number of Difference-of-Gaussians (DoG) scales per octave.
pub const DEFAULT_SCALES: usize = 3;
pub const MIN_SCALES: usize = 3;
pub const MAX_SCALES: usize = 10;

/// Image dimension safety bounds for HTTP uploads.
pub const MIN_IMAGE_DIM: u32 = 8;
pub const MAX_IMAGE_DIM: u32 = 8192;
pub const MAX_IMAGE_PIXELS: u64 = 25_000_000;

/// Structured JSON response returned by `POST /api/v1/process-features` and CLI `extract`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureResponse {
    pub width: usize,
    pub height: usize,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    pub keypoint_count: usize,
    pub keypoints: Vec<Keypoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<SiftFeature>>,
}

/// Structured JSON response returned by `POST /api/v1/match-features` and CLI `match`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesResponse {
    pub query_dimensions: (usize, usize),
    pub target_dimensions: (usize, usize),
    pub query_feature_count: usize,
    pub target_feature_count: usize,
    pub ratio_threshold: f32,
    pub ransac_enabled: bool,
    pub ransac_threshold: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<SimilarityTransform2D>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homography: Option<Homography2D>,
    pub tracking: TrackingSummary,
    pub matches: Vec<FeatureMatch>,
}

/// Application-level error shared between the HTTP API and CLI runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionError {
    InvalidInput(String),
    UnprocessableImage(String),
    PayloadTooLarge(String),
    Io(String),
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VisionError::InvalidInput(msg)
            | VisionError::UnprocessableImage(msg)
            | VisionError::PayloadTooLarge(msg)
            | VisionError::Io(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for VisionError {}

/// Validates common scale-space pipeline parameters (`octaves`, `scales`, `threshold`).
pub fn validate_pipeline_params(
    octaves: usize,
    scales: usize,
    threshold: f32,
) -> Result<(), VisionError> {
    if !(MIN_OCTAVES..=MAX_OCTAVES).contains(&octaves) {
        return Err(VisionError::InvalidInput(format!(
            "Invalid `octaves`={octaves}: must be between {MIN_OCTAVES} and {MAX_OCTAVES}"
        )));
    }
    if !(MIN_SCALES..=MAX_SCALES).contains(&scales) {
        return Err(VisionError::InvalidInput(format!(
            "Invalid `scales`={scales}: must be between {MIN_SCALES} and {MAX_SCALES} for 3D extrema extraction"
        )));
    }
    if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
        return Err(VisionError::InvalidInput(format!(
            "Invalid `threshold`={threshold}: must be finite and in (0.0, 1.0]"
        )));
    }
    Ok(())
}

/// Validates feature matching and RANSAC parameters (`ratio`, `ransac_threshold`).
pub fn validate_match_params(ratio: f32, ransac_threshold: f32) -> Result<(), VisionError> {
    if !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0 {
        return Err(VisionError::InvalidInput(format!(
            "Invalid `ratio`={ratio}: must be finite and in (0.0, 1.0]"
        )));
    }
    if !ransac_threshold.is_finite() || !(0.5..=100.0).contains(&ransac_threshold) {
        return Err(VisionError::InvalidInput(format!(
            "Invalid `ransac_threshold`={ransac_threshold}: must be between 0.5 and 100.0"
        )));
    }
    Ok(())
}

/// Validates image header dimensions before allocating full pixel buffers, then decodes to `DynamicImage`.
///
/// When `enforce_pixel_cap` is `true` (HTTP path), enforces `MAX_IMAGE_DIM` and `MAX_IMAGE_PIXELS`
/// to guard against decompression bombs. Local CLI runs may disable the upper cap.
pub fn decode_validated_dynamic_image(
    raw_bytes: &[u8],
    enforce_pixel_cap: bool,
) -> Result<DynamicImage, VisionError> {
    let reader = ImageReader::new(Cursor::new(raw_bytes))
        .with_guessed_format()
        .map_err(|e| {
            VisionError::UnprocessableImage(format!("Failed to inspect image header: {e}"))
        })?;

    let (w, h) = reader.into_dimensions().map_err(|e| {
        VisionError::UnprocessableImage(format!("Invalid or unsupported image header: {e}"))
    })?;

    if w < MIN_IMAGE_DIM || h < MIN_IMAGE_DIM {
        return Err(VisionError::InvalidInput(format!(
            "Image dimensions {w}x{h} are too small (minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM})"
        )));
    }

    if enforce_pixel_cap {
        let total_pixels = (w as u64) * (h as u64);
        if w > MAX_IMAGE_DIM || h > MAX_IMAGE_DIM || total_pixels > MAX_IMAGE_PIXELS {
            return Err(VisionError::PayloadTooLarge(format!(
                "Image dimensions {w}x{h} ({total_pixels} px) exceed maximum allowed ({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}, {MAX_IMAGE_PIXELS} px)"
            )));
        }
    }

    image::load_from_memory(raw_bytes).map_err(|e| {
        VisionError::UnprocessableImage(format!("Failed to decode PNG/JPEG image payload: {e}"))
    })
}

/// Validates image header dimensions and decodes bytes into a grayscale `Luma8` buffer.
pub fn decode_validated_luma8(
    raw_bytes: &[u8],
    enforce_pixel_cap: bool,
) -> Result<image::GrayImage, VisionError> {
    Ok(decode_validated_dynamic_image(raw_bytes, enforce_pixel_cap)?.into_luma8())
}

/// Executes the single-image keypoint + optional 128-D SIFT descriptor extraction pipeline.
pub fn run_extract_from_bytes(
    raw_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
    include_descriptors: bool,
    enforce_pixel_cap: bool,
) -> Result<FeatureResponse, VisionError> {
    validate_pipeline_params(octaves, scales, threshold)?;

    let luma_img = decode_validated_luma8(raw_bytes, enforce_pixel_cap)?;
    let img_array = luma_to_array2(&luma_img);
    let (height, width) = img_array.dim();

    let dog_space = generate_dog_space(&img_array, octaves, scales);
    let keypoints = extract_extrema(&dog_space, threshold);
    let features = if include_descriptors {
        Some(compute_sift_descriptors(&img_array, &keypoints))
    } else {
        None
    };

    Ok(FeatureResponse {
        width,
        height,
        octaves,
        scales,
        threshold,
        keypoint_count: keypoints.len(),
        keypoints,
        features,
    })
}

/// Executes the two-image SIFT matching + Similarity/Homography RANSAC tracking pipeline.
#[allow(clippy::too_many_arguments)]
pub fn run_match_from_bytes(
    query_bytes: &[u8],
    target_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
    ratio_threshold: f32,
    ransac_enabled: bool,
    ransac_threshold: f32,
    enforce_pixel_cap: bool,
) -> Result<MatchFeaturesResponse, VisionError> {
    validate_pipeline_params(octaves, scales, threshold)?;
    validate_match_params(ratio_threshold, ransac_threshold)?;

    let query_img = decode_validated_luma8(query_bytes, enforce_pixel_cap)
        .map_err(|e| map_image_prefix("First (query) image error", e))?;
    let target_img = decode_validated_luma8(target_bytes, enforce_pixel_cap)
        .map_err(|e| map_image_prefix("Second (target) image error", e))?;

    let query_arr = luma_to_array2(&query_img);
    let target_arr = luma_to_array2(&target_img);
    let (q_h, q_w) = query_arr.dim();
    let (t_h, t_w) = target_arr.dim();

    let query_features = extract_sift_features(&query_arr, octaves, scales, threshold);
    let target_features = extract_sift_features(&target_arr, octaves, scales, threshold);

    let raw_matches = match_descriptors(&query_features, &target_features, ratio_threshold);
    let tracking = summarize_tracking(&query_features, &target_features, &raw_matches);

    let (matches, transform, homography) = if ransac_enabled {
        let (sim_inliers, sim_tf) = filter_matches_ransac(&raw_matches, ransac_threshold);
        let (hom_inliers, hom_tf) = estimate_homography_ransac(&raw_matches, ransac_threshold);
        let best_inliers = if hom_inliers.len() > sim_inliers.len() {
            hom_inliers
        } else {
            sim_inliers
        };
        (best_inliers, sim_tf, hom_tf)
    } else {
        (raw_matches, None, None)
    };

    Ok(MatchFeaturesResponse {
        query_dimensions: (q_w, q_h),
        target_dimensions: (t_w, t_h),
        query_feature_count: query_features.len(),
        target_feature_count: target_features.len(),
        ratio_threshold,
        ransac_enabled,
        ransac_threshold,
        transform,
        homography,
        tracking,
        matches,
    })
}

/// Extracts SIFT features and renders keypoint scale circles + orientation ticks onto the image.
pub fn run_visualize_features_rgba(
    raw_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
    enforce_pixel_cap: bool,
) -> Result<(RgbaImage, usize), VisionError> {
    validate_pipeline_params(octaves, scales, threshold)?;

    let dyn_img = decode_validated_dynamic_image(raw_bytes, enforce_pixel_cap)?;
    let luma = dyn_img.to_luma8();
    let arr = luma_to_array2(&luma);

    let features = extract_sift_features(&arr, octaves, scales, threshold);
    let count = features.len();
    let rendered = draw_keypoints_overlay(&dyn_img, &features);
    Ok((rendered, count))
}

/// Matches SIFT features between two images and renders a side-by-side correspondence canvas.
#[allow(clippy::too_many_arguments)]
pub fn run_visualize_matches_rgba(
    query_bytes: &[u8],
    target_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
    ratio_threshold: f32,
    ransac_enabled: bool,
    ransac_threshold: f32,
    enforce_pixel_cap: bool,
) -> Result<(RgbaImage, usize, usize), VisionError> {
    validate_pipeline_params(octaves, scales, threshold)?;
    validate_match_params(ratio_threshold, ransac_threshold)?;

    let query_dyn = decode_validated_dynamic_image(query_bytes, enforce_pixel_cap)
        .map_err(|e| map_image_prefix("First (query) image error", e))?;
    let target_dyn = decode_validated_dynamic_image(target_bytes, enforce_pixel_cap)
        .map_err(|e| map_image_prefix("Second (target) image error", e))?;

    let q_arr = luma_to_array2(&query_dyn.to_luma8());
    let t_arr = luma_to_array2(&target_dyn.to_luma8());

    let query_features = extract_sift_features(&q_arr, octaves, scales, threshold);
    let target_features = extract_sift_features(&t_arr, octaves, scales, threshold);

    let raw_matches = match_descriptors(&query_features, &target_features, ratio_threshold);
    let raw_count = raw_matches.len();

    let display_matches = if ransac_enabled {
        let (sim_inliers, _) = filter_matches_ransac(&raw_matches, ransac_threshold);
        let (hom_inliers, _) = estimate_homography_ransac(&raw_matches, ransac_threshold);
        if hom_inliers.len() > sim_inliers.len() {
            hom_inliers
        } else {
            sim_inliers
        }
    } else {
        raw_matches
    };

    let shown_count = display_matches.len();
    let canvas = draw_matches_side_by_side(&query_dyn, &target_dyn, &display_matches);
    Ok((canvas, shown_count, raw_count))
}

/// Matches SIFT features, estimates Homography (with Similarity fallback), and warps/stitches
/// `query` onto `target`'s coordinate frame using bilinear inverse mapping and feather blending.
#[allow(clippy::too_many_arguments)]
pub fn run_align_images_rgba(
    query_bytes: &[u8],
    target_bytes: &[u8],
    octaves: usize,
    scales: usize,
    threshold: f32,
    ratio_threshold: f32,
    ransac_threshold: f32,
    enforce_pixel_cap: bool,
) -> Result<(RgbaImage, usize), VisionError> {
    validate_pipeline_params(octaves, scales, threshold)?;
    validate_match_params(ratio_threshold, ransac_threshold)?;

    let query_dyn = decode_validated_dynamic_image(query_bytes, enforce_pixel_cap)
        .map_err(|e| map_image_prefix("First (query) image error", e))?;
    let target_dyn = decode_validated_dynamic_image(target_bytes, enforce_pixel_cap)
        .map_err(|e| map_image_prefix("Second (target) image error", e))?;

    let q_arr = luma_to_array2(&query_dyn.to_luma8());
    let t_arr = luma_to_array2(&target_dyn.to_luma8());

    let query_features = extract_sift_features(&q_arr, octaves, scales, threshold);
    let target_features = extract_sift_features(&t_arr, octaves, scales, threshold);

    let raw_matches = match_descriptors(&query_features, &target_features, ratio_threshold);
    if raw_matches.is_empty() {
        return Err(VisionError::UnprocessableImage(
            "No SIFT descriptor matches found between query and target images".to_string(),
        ));
    }

    let (sim_inliers, sim_tf) = filter_matches_ransac(&raw_matches, ransac_threshold);
    let (hom_inliers, hom_tf) = estimate_homography_ransac(&raw_matches, ransac_threshold);
    let inlier_count = hom_inliers.len().max(sim_inliers.len());
    let (q_h, q_w) = q_arr.dim();

    let alignment = resolve_safe_alignment_homography(hom_tf, hom_inliers.len(), sim_tf, q_w, q_h)
        .ok_or_else(|| {
            VisionError::UnprocessableImage(
                "Failed to estimate a geometric alignment transform from matches".to_string(),
            )
        })?;

    let stitched = warp_and_stitch(&query_dyn, &target_dyn, &alignment)
        .map_err(VisionError::UnprocessableImage)?;
    Ok((stitched, inlier_count))
}

pub fn map_image_prefix(prefix: &str, err: VisionError) -> VisionError {
    match err {
        VisionError::InvalidInput(m) => VisionError::InvalidInput(format!("{prefix}: {m}")),
        VisionError::UnprocessableImage(m) => {
            VisionError::UnprocessableImage(format!("{prefix}: {m}"))
        }
        VisionError::PayloadTooLarge(m) => VisionError::PayloadTooLarge(format!("{prefix}: {m}")),
        VisionError::Io(m) => VisionError::Io(format!("{prefix}: {m}")),
    }
}

/// CLI options for the `extract` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractArgs {
    pub image_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    pub include_descriptors: bool,
    pub pretty: bool,
}

/// CLI options for the `match` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArgs {
    pub query_path: PathBuf,
    pub target_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    pub ratio: f32,
    pub ransac_enabled: bool,
    pub ransac_threshold: f32,
    pub pretty: bool,
}

/// CLI options for the `visualize` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizeArgs {
    pub image_path: PathBuf,
    pub output_path: PathBuf,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
}

/// CLI options for the `visualize-matches` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizeMatchesArgs {
    pub query_path: PathBuf,
    pub target_path: PathBuf,
    pub output_path: PathBuf,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    pub ratio: f32,
    pub ransac_enabled: bool,
    pub ransac_threshold: f32,
}

/// CLI options for the `align` (alias `stitch`) subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignArgs {
    pub query_path: PathBuf,
    pub target_path: PathBuf,
    pub output_path: PathBuf,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    pub ratio: f32,
    pub ransac_threshold: f32,
}

/// CLI options for the `catalog-build` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogBuildArgs {
    pub inputs: Vec<PathBuf>,
    pub output_path: PathBuf,
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    pub pretty: bool,
}

/// CLI options for the `catalog-search` subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogSearchArgs {
    pub catalog_path: PathBuf,
    pub query_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub top_k: usize,
    pub shortlist_k: usize,
    pub ratio: f32,
    pub ransac_threshold: f32,
    pub pretty: bool,
}

/// Structured JSON response returned by `catalog-search` and `POST /api/v1/catalog/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSearchResponse {
    pub query_dimensions: (usize, usize),
    pub query_feature_count: usize,
    pub catalog_size: usize,
    pub top_k: usize,
    pub shortlist_k: usize,
    pub hit_count: usize,
    pub hits: Vec<CatalogSearchHit>,
}

/// Parsed CLI command dispatched by `main`.
#[derive(Debug, Clone, PartialEq)]
pub enum CliCommand {
    Serve { bind_addr: Option<String> },
    Extract(ExtractArgs),
    Match(MatchArgs),
    Visualize(VisualizeArgs),
    VisualizeMatches(VisualizeMatchesArgs),
    Align(AlignArgs),
    CatalogBuild(CatalogBuildArgs),
    CatalogSearch(CatalogSearchArgs),
    Help(Option<String>),
    Version,
}

pub fn general_help_text() -> String {
    format!(
        "\
pure-sift {version}
Pure-Rust US6711293B1 (SIFT/DoG) feature extraction, matching, visual alignment, catalog search, and HTTP microservice.

USAGE:
  pure-sift [COMMAND] [OPTIONS]

COMMANDS:
  serve                             Start the Axum HTTP API server (default if no command is given)
  extract <IMAGE>                   Extract DoG keypoints (and optional 128-D descriptors) as JSON
  match <QUERY> <TARGET>            Match 128-D SIFT features with RANSAC & Homography as JSON
  visualize <IMAGE>                 Draw detected keypoints (scale circles & orientation) to PNG/JPEG
  visualize-matches <Q> <T>         Draw side-by-side RANSAC inlier match lines to PNG/JPEG
  align <QUERY> <TARGET>            Warp & feather-blend QUERY onto TARGET via Homography (alias: stitch)
  catalog-build <PATH...>           Index a folder or list of images into a SIFT search catalog JSON
  catalog-search <CATALOG> <QUERY>  Search a SIFT catalog for geometric matches to QUERY
  help [COMMAND]                    Show help for a command

OPTIONS:
  -h, --help                        Print help information
  -V, --version                     Print version information

EXAMPLES:
  pure-sift serve --bind 127.0.0.1:8080
  pure-sift extract buy-cheap.jpg --descriptors --pretty -o features.json
  pure-sift match buy-cheap.jpg buy-cheap-cropped.jpg --pretty -o matches.json
  pure-sift visualize buy-cheap.jpg --threshold 0.03 -o keypoints.png
  pure-sift align buy-cheap.jpg buy-cheap-cropped.jpg --threshold 0.03 -o aligned.png
  pure-sift catalog-build ./photos -o catalog.json
  pure-sift catalog-search catalog.json buy-cheap-cropped.jpg --pretty
",
        version = env!("CARGO_PKG_VERSION")
    )
}

pub fn subcommand_help_text(cmd: &str) -> Option<&'static str> {
    match cmd {
        "serve" => Some(
            "\
Start the Axum HTTP API microservice.

USAGE:
  pure-sift serve [OPTIONS]

OPTIONS:
  -b, --bind <ADDR>             Address to bind (overrides BIND_ADDR; default: 0.0.0.0:3000)
  -h, --help                    Print help information

ENVIRONMENT VARIABLES:
  BIND_ADDR                     TCP listen address (default: 0.0.0.0:3000)
  MAX_BODY_MB                   Multipart upload limit in MiB (default: 16)
  REQUEST_TIMEOUT_SECS          Per-request timeout in seconds (default: 30)
",
        ),
        "extract" => Some(
            "\
Extract scale-space DoG keypoints and optional 128-D SIFT descriptors from an image file.

USAGE:
  pure-sift extract <IMAGE> [OPTIONS]

ARGS:
  <IMAGE>                       Path to input PNG/JPEG/WebP/TIFF image

OPTIONS:
  -o, --output <FILE>           Write JSON output to FILE instead of stdout
      --octaves <N>             Number of scale-space octaves (1..=8, default: 3)
      --scales <N>              DoG scales per octave (3..=10, default: 3)
      --threshold <FLOAT>       Contrast threshold in (0.0, 1.0] (default: 0.015)
      --descriptors             Include 128-D SIFT descriptors and canonical orientations
      --include-descriptors     Alias for --descriptors
      --pretty                  Pretty-print JSON output
  -h, --help                    Print help information
",
        ),
        "match" => Some(
            "\
Match 128-D SIFT features between two images and estimate 2D Similarity & Homography transforms.

USAGE:
  pure-sift match <QUERY> <TARGET> [OPTIONS]

ARGS:
  <QUERY>                       Path to query image
  <TARGET>                      Path to target image

OPTIONS:
  -o, --output <FILE>           Write JSON output to FILE instead of stdout
      --octaves <N>             Number of scale-space octaves (1..=8, default: 3)
      --scales <N>              DoG scales per octave (3..=10, default: 3)
      --threshold <FLOAT>       Contrast threshold in (0.0, 1.0] (default: 0.015)
      --ratio <FLOAT>           Lowe's ratio threshold in (0.0, 1.0] (default: 0.75)
      --ransac-threshold <PX>   RANSAC reprojection error threshold in px (0.5..=100.0, default: 3.0)
      --no-ransac               Disable RANSAC outlier filtering (return raw ratio matches)
      --pretty                  Pretty-print JSON output
  -h, --help                    Print help information
",
        ),
        "visualize" => Some(
            "\
Render detected SIFT keypoints (scale circles & orientation ticks) onto an image.

USAGE:
  pure-sift visualize <IMAGE> [OPTIONS]

ARGS:
  <IMAGE>                       Path to input image

OPTIONS:
  -o, --output <FILE>           Output image path (default: keypoints.png)
      --octaves <N>             Number of scale-space octaves (1..=8, default: 3)
      --scales <N>              DoG scales per octave (3..=10, default: 3)
      --threshold <FLOAT>       Contrast threshold in (0.0, 1.0] (default: 0.015)
  -h, --help                    Print help information
",
        ),
        "visualize-matches" => Some(
            "\
Render side-by-side SIFT correspondences between two images.

USAGE:
  pure-sift visualize-matches <QUERY> <TARGET> [OPTIONS]

ARGS:
  <QUERY>                       Path to query image
  <TARGET>                      Path to target image

OPTIONS:
  -o, --output <FILE>           Output image path (default: matches.png)
      --octaves <N>             Number of scale-space octaves (1..=8, default: 3)
      --scales <N>              DoG scales per octave (3..=10, default: 3)
      --threshold <FLOAT>       Contrast threshold in (0.0, 1.0] (default: 0.015)
      --ratio <FLOAT>           Lowe's ratio threshold in (0.0, 1.0] (default: 0.75)
      --ransac-threshold <PX>   RANSAC reprojection error threshold in px (0.5..=100.0, default: 3.0)
      --no-ransac               Draw raw ratio matches instead of RANSAC inliers
  -h, --help                    Print help information
",
        ),
        "align" | "stitch" => Some(
            "\
Warp QUERY into TARGET's coordinate frame via SIFT + RANSAC Homography and feather-blend.

USAGE:
  pure-sift align <QUERY> <TARGET> [OPTIONS]

ARGS:
  <QUERY>                       Path to query image to warp
  <TARGET>                      Path to reference target image

OPTIONS:
  -o, --output <FILE>           Output image path (default: aligned.png)
      --octaves <N>             Number of scale-space octaves (1..=8, default: 3)
      --scales <N>              DoG scales per octave (3..=10, default: 3)
      --threshold <FLOAT>       Contrast threshold in (0.0, 1.0] (default: 0.015)
      --ratio <FLOAT>           Lowe's ratio threshold in (0.0, 1.0] (default: 0.75)
      --ransac-threshold <PX>   RANSAC reprojection error threshold in px (0.5..=100.0, default: 3.0)
  -h, --help                    Print help information
",
        ),
        "catalog-build" => Some(
            "\
Index a folder or list of images in parallel into a searchable 128-D SIFT catalog JSON.

USAGE:
  pure-sift catalog-build <PATH...> [OPTIONS]

ARGS:
  <PATH...>                     One or more image files or directories containing PNG/JPEG/WebP/TIFF

OPTIONS:
  -o, --output <FILE>           Output catalog JSON path (default: catalog.json)
      --octaves <N>             Number of scale-space octaves (1..=8, default: 3)
      --scales <N>              DoG scales per octave (3..=10, default: 3)
      --threshold <FLOAT>       Contrast threshold in (0.0, 1.0] (default: 0.015)
      --pretty                  Pretty-print catalog JSON output
  -h, --help                    Print help information
",
        ),
        "catalog-search" => Some(
            "\
Search a SIFT FeatureCatalog for geometric matches to QUERY using 128-bit Hamming prefiltering and RANSAC.

USAGE:
  pure-sift catalog-search <CATALOG> <QUERY> [OPTIONS]

ARGS:
  <CATALOG>                     Path to catalog JSON built by `catalog-build`
  <QUERY>                       Path to query image

OPTIONS:
  -o, --output <FILE>           Write JSON output to FILE instead of stdout
      --top-k <N>               Maximum ranked hits to return (default: 5)
      --shortlist <N>           Coarse Hamming shortlist verified with full RANSAC (default: 16)
      --ratio <FLOAT>           Lowe's ratio threshold in (0.0, 1.0] (default: 0.75)
      --ransac-threshold <PX>   RANSAC reprojection error threshold in px (0.5..=100.0, default: 3.0)
      --pretty                  Pretty-print JSON output
  -h, --help                    Print help information
",
        ),
        _ => None,
    }
}

/// Parses CLI arguments (excluding `argv[0]`) into a `CliCommand`.
pub fn parse_cli_args<I, S>(args: I) -> Result<CliCommand, VisionError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let raw_args: Vec<String> = args.into_iter().map(Into::into).collect();
    if raw_args.is_empty() {
        return Ok(CliCommand::Serve { bind_addr: None });
    }

    match raw_args[0].as_str() {
        "-h" | "--help" => Ok(CliCommand::Help(None)),
        "-V" | "--version" => Ok(CliCommand::Version),
        "help" => {
            if raw_args.len() > 2 {
                return Err(VisionError::InvalidInput(format!(
                    "Unexpected argument `{}` for `help`",
                    raw_args[2]
                )));
            }
            if let Some(sub) = raw_args.get(1) {
                if subcommand_help_text(sub).is_none() {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown subcommand `{sub}`. Run `--help` for available commands."
                    )));
                }
                Ok(CliCommand::Help(Some(sub.clone())))
            } else {
                Ok(CliCommand::Help(None))
            }
        }
        "serve" => parse_serve_args(&raw_args[1..]),
        "extract" => parse_extract_args(&raw_args[1..]),
        "match" => parse_match_args(&raw_args[1..]),
        "visualize" => parse_visualize_args(&raw_args[1..]),
        "visualize-matches" => parse_visualize_matches_args(&raw_args[1..]),
        "align" | "stitch" => parse_align_args(&raw_args[1..]),
        "catalog-build" => parse_catalog_build_args(&raw_args[1..]),
        "catalog-search" => parse_catalog_search_args(&raw_args[1..]),
        other => Err(VisionError::InvalidInput(format!(
            "Unknown command `{other}`. Run with `--help` for usage."
        ))),
    }
}

fn split_flag_value(arg: &str) -> (&str, Option<&str>) {
    if let Some((flag, val)) = arg.split_once('=') {
        (flag, Some(val))
    } else {
        (arg, None)
    }
}

fn take_value(
    flag: &str,
    inline_val: Option<&str>,
    args: &[String],
    idx: &mut usize,
) -> Result<String, VisionError> {
    if let Some(v) = inline_val {
        return Ok(v.to_string());
    }
    *idx += 1;
    args.get(*idx)
        .cloned()
        .ok_or_else(|| VisionError::InvalidInput(format!("Missing value for option `{flag}`")))
}

fn parse_typed<T: std::str::FromStr>(flag: &str, val: &str) -> Result<T, VisionError> {
    val.parse::<T>().map_err(|_| {
        VisionError::InvalidInput(format!("Invalid value `{val}` for option `{flag}`"))
    })
}

fn parse_serve_args(args: &[String]) -> Result<CliCommand, VisionError> {
    let mut bind_addr = None;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        let (flag, inline_val) = split_flag_value(arg);
        match flag {
            "-h" | "--help" => return Ok(CliCommand::Help(Some("serve".to_string()))),
            "-b" | "--bind" => {
                let val = take_value(flag, inline_val, args, &mut i)?;
                if val.trim().is_empty() {
                    return Err(VisionError::InvalidInput(
                        "Bind address cannot be empty".to_string(),
                    ));
                }
                bind_addr = Some(val);
            }
            _ => {
                return Err(VisionError::InvalidInput(format!(
                    "Unknown argument `{arg}` for `serve`. Run `serve --help` for usage."
                )));
            }
        }
        i += 1;
    }

    Ok(CliCommand::Serve { bind_addr })
}

fn parse_extract_args(args: &[String]) -> Result<CliCommand, VisionError> {
    let mut image_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut octaves = DEFAULT_OCTAVES;
    let mut scales = DEFAULT_SCALES;
    let mut threshold = DEFAULT_CONTRAST_THRESHOLD;
    let mut include_descriptors = false;
    let mut pretty = false;
    let mut positional_only = false;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        if !positional_only && arg == "--" {
            positional_only = true;
            i += 1;
            continue;
        }

        if !positional_only && arg.starts_with('-') && arg != "-" {
            let (flag, inline_val) = split_flag_value(arg);
            match flag {
                "-h" | "--help" => return Ok(CliCommand::Help(Some("extract".to_string()))),
                "-o" | "--output" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    output_path = Some(PathBuf::from(val));
                }
                "--octaves" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    octaves = parse_typed(flag, &val)?;
                }
                "--scales" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    scales = parse_typed(flag, &val)?;
                }
                "--threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    threshold = parse_typed(flag, &val)?;
                }
                "--descriptors" | "--include-descriptors" => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    include_descriptors = true;
                }
                "--pretty" => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    pretty = true;
                }
                _ => {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown option `{flag}` for `extract`. Run `extract --help` for usage."
                    )));
                }
            }
        } else if image_path.is_none() {
            image_path = Some(PathBuf::from(arg));
        } else {
            return Err(VisionError::InvalidInput(format!(
                "Unexpected extra positional argument `{arg}` for `extract`"
            )));
        }

        i += 1;
    }

    let Some(img) = image_path else {
        return Err(VisionError::InvalidInput(
            "Missing required `<IMAGE>` path for `extract`. Run `extract --help` for usage."
                .to_string(),
        ));
    };

    validate_pipeline_params(octaves, scales, threshold)?;

    Ok(CliCommand::Extract(ExtractArgs {
        image_path: img,
        output_path,
        octaves,
        scales,
        threshold,
        include_descriptors,
        pretty,
    }))
}

fn parse_match_args(args: &[String]) -> Result<CliCommand, VisionError> {
    let mut query_path: Option<PathBuf> = None;
    let mut target_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut octaves = DEFAULT_OCTAVES;
    let mut scales = DEFAULT_SCALES;
    let mut threshold = DEFAULT_CONTRAST_THRESHOLD;
    let mut ratio = DEFAULT_MATCH_RATIO;
    let mut ransac_enabled = true;
    let mut ransac_threshold = DEFAULT_RANSAC_REPROJ_THRESHOLD;
    let mut pretty = false;
    let mut positional_only = false;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        if !positional_only && arg == "--" {
            positional_only = true;
            i += 1;
            continue;
        }

        if !positional_only && arg.starts_with('-') && arg != "-" {
            let (flag, inline_val) = split_flag_value(arg);
            match flag {
                "-h" | "--help" => return Ok(CliCommand::Help(Some("match".to_string()))),
                "-o" | "--output" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    output_path = Some(PathBuf::from(val));
                }
                "--octaves" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    octaves = parse_typed(flag, &val)?;
                }
                "--scales" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    scales = parse_typed(flag, &val)?;
                }
                "--threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    threshold = parse_typed(flag, &val)?;
                }
                "--ratio" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    ratio = parse_typed(flag, &val)?;
                }
                "--ransac-threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    ransac_threshold = parse_typed(flag, &val)?;
                }
                "--no-ransac" => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    ransac_enabled = false;
                }
                "--pretty" => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    pretty = true;
                }
                _ => {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown option `{flag}` for `match`. Run `match --help` for usage."
                    )));
                }
            }
        } else if query_path.is_none() {
            query_path = Some(PathBuf::from(arg));
        } else if target_path.is_none() {
            target_path = Some(PathBuf::from(arg));
        } else {
            return Err(VisionError::InvalidInput(format!(
                "Unexpected extra positional argument `{arg}` for `match`"
            )));
        }

        i += 1;
    }

    let (Some(q_path), Some(t_path)) = (query_path, target_path) else {
        return Err(VisionError::InvalidInput(
            "Missing required `<QUERY>` and `<TARGET>` image paths for `match`. Run `match --help` for usage."
                .to_string(),
        ));
    };

    validate_pipeline_params(octaves, scales, threshold)?;
    validate_match_params(ratio, ransac_threshold)?;

    Ok(CliCommand::Match(MatchArgs {
        query_path: q_path,
        target_path: t_path,
        output_path,
        octaves,
        scales,
        threshold,
        ratio,
        ransac_enabled,
        ransac_threshold,
        pretty,
    }))
}

fn parse_visualize_args(args: &[String]) -> Result<CliCommand, VisionError> {
    let mut image_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("keypoints.png");
    let mut octaves = DEFAULT_OCTAVES;
    let mut scales = DEFAULT_SCALES;
    let mut threshold = DEFAULT_CONTRAST_THRESHOLD;
    let mut positional_only = false;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        if !positional_only && arg == "--" {
            positional_only = true;
            i += 1;
            continue;
        }

        if !positional_only && arg.starts_with('-') && arg != "-" {
            let (flag, inline_val) = split_flag_value(arg);
            match flag {
                "-h" | "--help" => return Ok(CliCommand::Help(Some("visualize".to_string()))),
                "-o" | "--output" => {
                    output_path = PathBuf::from(take_value(flag, inline_val, args, &mut i)?);
                }
                "--octaves" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    octaves = parse_typed(flag, &val)?;
                }
                "--scales" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    scales = parse_typed(flag, &val)?;
                }
                "--threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    threshold = parse_typed(flag, &val)?;
                }
                _ => {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown option `{flag}` for `visualize`. Run `visualize --help` for usage."
                    )));
                }
            }
        } else if image_path.is_none() {
            image_path = Some(PathBuf::from(arg));
        } else {
            return Err(VisionError::InvalidInput(format!(
                "Unexpected extra positional argument `{arg}` for `visualize`"
            )));
        }

        i += 1;
    }

    let Some(img) = image_path else {
        return Err(VisionError::InvalidInput(
            "Missing required `<IMAGE>` path for `visualize`. Run `visualize --help` for usage."
                .to_string(),
        ));
    };

    validate_pipeline_params(octaves, scales, threshold)?;

    Ok(CliCommand::Visualize(VisualizeArgs {
        image_path: img,
        output_path,
        octaves,
        scales,
        threshold,
    }))
}

fn parse_pair_visual_args(
    args: &[String],
    cmd_name: &str,
    default_output: &str,
    allow_no_ransac: bool,
) -> Result<(MatchArgs, PathBuf), VisionError> {
    let mut query_path: Option<PathBuf> = None;
    let mut target_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from(default_output);
    let mut octaves = DEFAULT_OCTAVES;
    let mut scales = DEFAULT_SCALES;
    let mut threshold = DEFAULT_CONTRAST_THRESHOLD;
    let mut ratio = DEFAULT_MATCH_RATIO;
    let mut ransac_enabled = true;
    let mut ransac_threshold = DEFAULT_RANSAC_REPROJ_THRESHOLD;
    let mut positional_only = false;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        if !positional_only && arg == "--" {
            positional_only = true;
            i += 1;
            continue;
        }

        if !positional_only && arg.starts_with('-') && arg != "-" {
            let (flag, inline_val) = split_flag_value(arg);
            match flag {
                "-h" | "--help" => {
                    return Err(VisionError::Io(format!("__HELP__:{cmd_name}")));
                }
                "-o" | "--output" => {
                    output_path = PathBuf::from(take_value(flag, inline_val, args, &mut i)?);
                }
                "--octaves" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    octaves = parse_typed(flag, &val)?;
                }
                "--scales" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    scales = parse_typed(flag, &val)?;
                }
                "--threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    threshold = parse_typed(flag, &val)?;
                }
                "--ratio" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    ratio = parse_typed(flag, &val)?;
                }
                "--ransac-threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    ransac_threshold = parse_typed(flag, &val)?;
                }
                "--no-ransac" if allow_no_ransac => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    ransac_enabled = false;
                }
                _ => {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown option `{flag}` for `{cmd_name}`. Run `{cmd_name} --help` for usage."
                    )));
                }
            }
        } else if query_path.is_none() {
            query_path = Some(PathBuf::from(arg));
        } else if target_path.is_none() {
            target_path = Some(PathBuf::from(arg));
        } else {
            return Err(VisionError::InvalidInput(format!(
                "Unexpected extra positional argument `{arg}` for `{cmd_name}`"
            )));
        }

        i += 1;
    }

    let (Some(q_path), Some(t_path)) = (query_path, target_path) else {
        return Err(VisionError::InvalidInput(format!(
            "Missing required `<QUERY>` and `<TARGET>` image paths for `{cmd_name}`. Run `{cmd_name} --help` for usage."
        )));
    };

    validate_pipeline_params(octaves, scales, threshold)?;
    validate_match_params(ratio, ransac_threshold)?;

    Ok((
        MatchArgs {
            query_path: q_path,
            target_path: t_path,
            output_path: Some(output_path.clone()),
            octaves,
            scales,
            threshold,
            ratio,
            ransac_enabled,
            ransac_threshold,
            pretty: false,
        },
        output_path,
    ))
}

fn parse_visualize_matches_args(args: &[String]) -> Result<CliCommand, VisionError> {
    match parse_pair_visual_args(args, "visualize-matches", "matches.png", true) {
        Ok((m, out)) => Ok(CliCommand::VisualizeMatches(VisualizeMatchesArgs {
            query_path: m.query_path,
            target_path: m.target_path,
            output_path: out,
            octaves: m.octaves,
            scales: m.scales,
            threshold: m.threshold,
            ratio: m.ratio,
            ransac_enabled: m.ransac_enabled,
            ransac_threshold: m.ransac_threshold,
        })),
        Err(VisionError::Io(ref s)) if s == "__HELP__:visualize-matches" => {
            Ok(CliCommand::Help(Some("visualize-matches".to_string())))
        }
        Err(e) => Err(e),
    }
}

fn parse_align_args(args: &[String]) -> Result<CliCommand, VisionError> {
    match parse_pair_visual_args(args, "align", "aligned.png", false) {
        Ok((m, out)) => Ok(CliCommand::Align(AlignArgs {
            query_path: m.query_path,
            target_path: m.target_path,
            output_path: out,
            octaves: m.octaves,
            scales: m.scales,
            threshold: m.threshold,
            ratio: m.ratio,
            ransac_threshold: m.ransac_threshold,
        })),
        Err(VisionError::Io(ref s)) if s == "__HELP__:align" => {
            Ok(CliCommand::Help(Some("align".to_string())))
        }
        Err(e) => Err(e),
    }
}

fn parse_catalog_build_args(args: &[String]) -> Result<CliCommand, VisionError> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut output_path = PathBuf::from("catalog.json");
    let mut octaves = DEFAULT_OCTAVES;
    let mut scales = DEFAULT_SCALES;
    let mut threshold = DEFAULT_CONTRAST_THRESHOLD;
    let mut pretty = false;
    let mut positional_only = false;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        if !positional_only && arg == "--" {
            positional_only = true;
            i += 1;
            continue;
        }

        if !positional_only && arg.starts_with('-') && arg != "-" {
            let (flag, inline_val) = split_flag_value(arg);
            match flag {
                "-h" | "--help" => {
                    return Ok(CliCommand::Help(Some("catalog-build".to_string())));
                }
                "-o" | "--output" => {
                    output_path = PathBuf::from(take_value(flag, inline_val, args, &mut i)?);
                }
                "--octaves" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    octaves = parse_typed(flag, &val)?;
                }
                "--scales" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    scales = parse_typed(flag, &val)?;
                }
                "--threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    threshold = parse_typed(flag, &val)?;
                }
                "--pretty" => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    pretty = true;
                }
                _ => {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown option `{flag}` for `catalog-build`. Run `catalog-build --help` for usage."
                    )));
                }
            }
        } else {
            inputs.push(PathBuf::from(arg));
        }

        i += 1;
    }

    if inputs.is_empty() {
        return Err(VisionError::InvalidInput(
            "Missing required `<PATH...>` image or directory inputs for `catalog-build`."
                .to_string(),
        ));
    }

    validate_pipeline_params(octaves, scales, threshold)?;

    Ok(CliCommand::CatalogBuild(CatalogBuildArgs {
        inputs,
        output_path,
        octaves,
        scales,
        threshold,
        pretty,
    }))
}

fn parse_catalog_search_args(args: &[String]) -> Result<CliCommand, VisionError> {
    let mut catalog_path: Option<PathBuf> = None;
    let mut query_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut top_k = DEFAULT_CATALOG_TOP_K;
    let mut shortlist_k = DEFAULT_CATALOG_SHORTLIST;
    let mut ratio = DEFAULT_MATCH_RATIO;
    let mut ransac_threshold = DEFAULT_RANSAC_REPROJ_THRESHOLD;
    let mut pretty = false;
    let mut positional_only = false;
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];
        if !positional_only && arg == "--" {
            positional_only = true;
            i += 1;
            continue;
        }

        if !positional_only && arg.starts_with('-') && arg != "-" {
            let (flag, inline_val) = split_flag_value(arg);
            match flag {
                "-h" | "--help" => {
                    return Ok(CliCommand::Help(Some("catalog-search".to_string())));
                }
                "-o" | "--output" => {
                    output_path = Some(PathBuf::from(take_value(flag, inline_val, args, &mut i)?));
                }
                "--top-k" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    top_k = parse_typed(flag, &val)?;
                }
                "--shortlist" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    shortlist_k = parse_typed(flag, &val)?;
                }
                "--ratio" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    ratio = parse_typed(flag, &val)?;
                }
                "--ransac-threshold" => {
                    let val = take_value(flag, inline_val, args, &mut i)?;
                    ransac_threshold = parse_typed(flag, &val)?;
                }
                "--pretty" => {
                    if inline_val.is_some() {
                        return Err(VisionError::InvalidInput(format!(
                            "Flag `{flag}` does not take a value"
                        )));
                    }
                    pretty = true;
                }
                _ => {
                    return Err(VisionError::InvalidInput(format!(
                        "Unknown option `{flag}` for `catalog-search`. Run `catalog-search --help` for usage."
                    )));
                }
            }
        } else if catalog_path.is_none() {
            catalog_path = Some(PathBuf::from(arg));
        } else if query_path.is_none() {
            query_path = Some(PathBuf::from(arg));
        } else {
            return Err(VisionError::InvalidInput(format!(
                "Unexpected extra positional argument `{arg}` for `catalog-search`"
            )));
        }

        i += 1;
    }

    let (Some(cat), Some(q)) = (catalog_path, query_path) else {
        return Err(VisionError::InvalidInput(
            "Missing required `<CATALOG>` and `<QUERY>` paths for `catalog-search`.".to_string(),
        ));
    };

    if top_k == 0 || shortlist_k == 0 {
        return Err(VisionError::InvalidInput(
            "`--top-k` and `--shortlist` must be at least 1".to_string(),
        ));
    }
    validate_match_params(ratio, ransac_threshold)?;

    Ok(CliCommand::CatalogSearch(CatalogSearchArgs {
        catalog_path: cat,
        query_path: q,
        output_path,
        top_k,
        shortlist_k,
        ratio,
        ransac_threshold,
        pretty,
    }))
}

fn is_supported_image_ext(path: &std::path::Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "webp" | "tif" | "tiff" | "bmp"
    )
}

fn collect_image_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, VisionError> {
    let mut files = Vec::new();
    for path in inputs {
        if path.is_dir() {
            let entries = fs::read_dir(path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read directory `{}`: {e}",
                    path.display()
                ))
            })?;
            for entry in entries {
                let entry = entry.map_err(|e| {
                    VisionError::Io(format!(
                        "Directory traversal error in `{}`: {e}",
                        path.display()
                    ))
                })?;
                let p = entry.path();
                if p.is_file() && is_supported_image_ext(&p) {
                    files.push(p);
                }
            }
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

/// Searches a `FeatureCatalog` using raw image bytes for the query image.
#[allow(clippy::too_many_arguments)]
pub fn run_catalog_search_from_bytes(
    catalog: &FeatureCatalog,
    query_bytes: &[u8],
    top_k: usize,
    shortlist_k: usize,
    ratio_threshold: f32,
    ransac_threshold: f32,
    enforce_pixel_cap: bool,
) -> Result<CatalogSearchResponse, VisionError> {
    if top_k == 0 || shortlist_k == 0 {
        return Err(VisionError::InvalidInput(
            "`top_k` and `shortlist` must be at least 1".to_string(),
        ));
    }
    validate_match_params(ratio_threshold, ransac_threshold)?;

    let query_luma = decode_validated_luma8(query_bytes, enforce_pixel_cap)?;
    let query_arr = luma_to_array2(&query_luma);
    let (q_h, q_w) = query_arr.dim();

    let query_features = extract_sift_features(
        &query_arr,
        catalog.octaves,
        catalog.scales,
        catalog.threshold,
    );
    let hits = catalog.search_features(
        &query_features,
        top_k,
        shortlist_k,
        ratio_threshold,
        ransac_threshold,
    );

    Ok(CatalogSearchResponse {
        query_dimensions: (q_w, q_h),
        query_feature_count: query_features.len(),
        catalog_size: catalog.entries.len(),
        top_k,
        shortlist_k,
        hit_count: hits.len(),
        hits,
    })
}

fn ensure_parent_dir(path: &std::path::Path) -> Result<(), VisionError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            VisionError::Io(format!(
                "Failed to create output directory `{}`: {e}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

pub fn save_rgba_output(img: &RgbaImage, path: &PathBuf) -> Result<(), VisionError> {
    ensure_parent_dir(path)?;
    let format = ImageFormat::from_path(path).unwrap_or(ImageFormat::Png);
    let dyn_img = match format {
        ImageFormat::Jpeg => {
            DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(img.clone()).to_rgb8())
        }
        _ => DynamicImage::ImageRgba8(img.clone()),
    };
    dyn_img
        .save_with_format(path, format)
        .map_err(|e| VisionError::Io(format!("Failed to write image `{}`: {e}", path.display())))
}

pub fn serialize_json<T: Serialize>(value: &T, pretty: bool) -> Result<String, VisionError> {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|e| VisionError::Io(format!("Failed to serialize JSON output: {e}")))
}

pub fn write_json_output(
    json: &str,
    output_path: Option<&PathBuf>,
    stdout: &mut impl Write,
) -> Result<(), VisionError> {
    match output_path {
        Some(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent).map_err(|e| {
                    VisionError::Io(format!(
                        "Failed to create output directory `{}`: {e}",
                        parent.display()
                    ))
                })?;
            }
            fs::write(path, format!("{json}\n")).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to write output file `{}`: {e}",
                    path.display()
                ))
            })
        }
        None => writeln!(stdout, "{json}")
            .map_err(|e| VisionError::Io(format!("Failed to write to stdout: {e}"))),
    }
}

/// Executes non-server CLI commands (`extract`, `match`, `help`, `version`), writing output to `stdout`.
pub fn execute_cli_command(cmd: &CliCommand, stdout: &mut impl Write) -> Result<(), VisionError> {
    match cmd {
        CliCommand::Help(sub) => {
            let text = sub
                .as_deref()
                .and_then(subcommand_help_text)
                .map(str::to_string)
                .unwrap_or_else(general_help_text);
            write!(stdout, "{text}")
                .map_err(|e| VisionError::Io(format!("Failed to write help output: {e}")))
        }
        CliCommand::Version => writeln!(stdout, "pure-sift {}", env!("CARGO_PKG_VERSION"))
            .map_err(|e| VisionError::Io(format!("Failed to write version output: {e}"))),
        CliCommand::Extract(args) => {
            let raw_bytes = fs::read(&args.image_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read image `{}`: {e}",
                    args.image_path.display()
                ))
            })?;
            let response = run_extract_from_bytes(
                &raw_bytes,
                args.octaves,
                args.scales,
                args.threshold,
                args.include_descriptors,
                false,
            )?;
            let json = serialize_json(&response, args.pretty)?;
            write_json_output(&json, args.output_path.as_ref(), stdout)?;
            if let Some(out) = &args.output_path {
                eprintln!(
                    "Extracted {} keypoints ({}x{}) -> {}",
                    response.keypoint_count,
                    response.width,
                    response.height,
                    out.display()
                );
            }
            Ok(())
        }
        CliCommand::Match(args) => {
            let query_bytes = fs::read(&args.query_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read query image `{}`: {e}",
                    args.query_path.display()
                ))
            })?;
            let target_bytes = fs::read(&args.target_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read target image `{}`: {e}",
                    args.target_path.display()
                ))
            })?;
            let response = run_match_from_bytes(
                &query_bytes,
                &target_bytes,
                args.octaves,
                args.scales,
                args.threshold,
                args.ratio,
                args.ransac_enabled,
                args.ransac_threshold,
                false,
            )?;
            let json = serialize_json(&response, args.pretty)?;
            write_json_output(&json, args.output_path.as_ref(), stdout)?;
            if let Some(out) = &args.output_path {
                eprintln!(
                    "Matched {} inliers (of {} raw matches) -> {}",
                    response.tracking.match_count,
                    response.tracking.raw_match_count,
                    out.display()
                );
            }
            Ok(())
        }
        CliCommand::Visualize(args) => {
            let raw_bytes = fs::read(&args.image_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read image `{}`: {e}",
                    args.image_path.display()
                ))
            })?;
            let (rendered, count) = run_visualize_features_rgba(
                &raw_bytes,
                args.octaves,
                args.scales,
                args.threshold,
                false,
            )?;
            save_rgba_output(&rendered, &args.output_path)?;
            writeln!(
                stdout,
                "Rendered {} keypoints ({}x{}) -> {}",
                count,
                rendered.width(),
                rendered.height(),
                args.output_path.display()
            )
            .map_err(|e| VisionError::Io(format!("Failed to write to stdout: {e}")))
        }
        CliCommand::VisualizeMatches(args) => {
            let query_bytes = fs::read(&args.query_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read query image `{}`: {e}",
                    args.query_path.display()
                ))
            })?;
            let target_bytes = fs::read(&args.target_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read target image `{}`: {e}",
                    args.target_path.display()
                ))
            })?;
            let (canvas, shown, raw_count) = run_visualize_matches_rgba(
                &query_bytes,
                &target_bytes,
                args.octaves,
                args.scales,
                args.threshold,
                args.ratio,
                args.ransac_enabled,
                args.ransac_threshold,
                false,
            )?;
            save_rgba_output(&canvas, &args.output_path)?;
            writeln!(
                stdout,
                "Rendered {} matches (of {} raw) ({}x{}) -> {}",
                shown,
                raw_count,
                canvas.width(),
                canvas.height(),
                args.output_path.display()
            )
            .map_err(|e| VisionError::Io(format!("Failed to write to stdout: {e}")))
        }
        CliCommand::Align(args) => {
            let query_bytes = fs::read(&args.query_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read query image `{}`: {e}",
                    args.query_path.display()
                ))
            })?;
            let target_bytes = fs::read(&args.target_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read target image `{}`: {e}",
                    args.target_path.display()
                ))
            })?;
            let (stitched, inliers) = run_align_images_rgba(
                &query_bytes,
                &target_bytes,
                args.octaves,
                args.scales,
                args.threshold,
                args.ratio,
                args.ransac_threshold,
                false,
            )?;
            save_rgba_output(&stitched, &args.output_path)?;
            writeln!(
                stdout,
                "Aligned & stitched ({} inliers) ({}x{}) -> {}",
                inliers,
                stitched.width(),
                stitched.height(),
                args.output_path.display()
            )
            .map_err(|e| VisionError::Io(format!("Failed to write to stdout: {e}")))
        }
        CliCommand::CatalogBuild(args) => {
            let image_paths = collect_image_files(&args.inputs)?;
            if image_paths.is_empty() {
                return Err(VisionError::InvalidInput(
                    "No supported image files found in the provided paths".to_string(),
                ));
            }

            let octaves = args.octaves;
            let scales = args.scales;
            let threshold = args.threshold;

            type IndexedEntryTuple = (String, usize, usize, Vec<SiftFeature>);
            // Extract SIFT features across all images in parallel using Rayon
            let indexed_entries: Result<Vec<IndexedEntryTuple>, VisionError> = image_paths
                .par_iter()
                .map(|path| {
                    let raw_bytes = fs::read(path).map_err(|e| {
                        VisionError::Io(format!("Failed to read `{}`: {e}", path.display()))
                    })?;
                    let resp = run_extract_from_bytes(
                        &raw_bytes, octaves, scales, threshold, true, false,
                    )?;
                    let feats = resp.features.unwrap_or_default();
                    let id = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| path.display().to_string());
                    Ok((id, resp.width, resp.height, feats))
                })
                .collect();

            let mut catalog = FeatureCatalog::new(octaves, scales, threshold);
            for (id, w, h, feats) in indexed_entries? {
                catalog.upsert_entry(id, w, h, feats);
            }

            catalog
                .save_json(&args.output_path, args.pretty)
                .map_err(VisionError::Io)?;
            writeln!(
                stdout,
                "Indexed {} images -> {}",
                catalog.entries.len(),
                args.output_path.display()
            )
            .map_err(|e| VisionError::Io(format!("Failed to write to stdout: {e}")))
        }
        CliCommand::CatalogSearch(args) => {
            let catalog = FeatureCatalog::load_json(&args.catalog_path).map_err(VisionError::Io)?;
            let query_bytes = fs::read(&args.query_path).map_err(|e| {
                VisionError::Io(format!(
                    "Failed to read query image `{}`: {e}",
                    args.query_path.display()
                ))
            })?;
            let response = run_catalog_search_from_bytes(
                &catalog,
                &query_bytes,
                args.top_k,
                args.shortlist_k,
                args.ratio,
                args.ransac_threshold,
                false,
            )?;
            let json = serialize_json(&response, args.pretty)?;
            write_json_output(&json, args.output_path.as_ref(), stdout)?;
            if let Some(out) = &args.output_path {
                eprintln!(
                    "Found {} catalog hits (searched {} images) -> {}",
                    response.hit_count,
                    response.catalog_size,
                    out.display()
                );
            }
            Ok(())
        }
        CliCommand::Serve { .. } => Ok(()),
    }
}

/// Convenience entry helper writing CLI output directly to standard output.
pub fn execute_cli_to_stdout(cmd: &CliCommand) -> Result<(), VisionError> {
    let mut stdout = io::stdout().lock();
    execute_cli_command(cmd, &mut stdout)
}
