//! Pure-Rust SIFT feature extraction (Lowe 2004, expired patent `US6711293B1`),
//! multi-view geometry estimation, RANSAC homography alignment, multi-image
//! panorama stitching, and compact Hamming/MIH visual catalog search with zero
//! `unsafe` code (`#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Binary & JSON visual feature catalog with 16x8-bit Multi-Index Hashing search.
pub mod catalog;
/// Internal CLI runner & payload validation helpers (used by the binary, WASM demo, and fuzz targets).
#[doc(hidden)]
#[allow(missing_docs)]
pub mod cli;
/// Structured error types powered by `thiserror`.
pub mod error;
/// Fundamental/Essential matrix RANSAC solvers, linear triangulation, pose recovery, and GRIC selection.
pub mod geometry;
/// Core DoG scale-space SIFT feature extraction, descriptor matching, and 2D Similarity/Homography RANSAC.
pub mod pipeline;
/// Multi-image panorama stitching, cylindrical warp, exposure compensation, and multi-band blending.
pub mod stitch;
/// Image alignment warping, feather blending, and keypoint/match PNG overlays.
pub mod visualize;

pub use catalog::{
    CatalogEntry, CatalogEntrySummary, CatalogSearchHit, CompactSiftFeature,
    DEFAULT_CATALOG_SHORTLIST, DEFAULT_CATALOG_TOP_K, FeatureCatalog, quantize_descriptor_u128,
};
pub use error::VisionError;
pub use geometry::{
    CameraIntrinsics, EssentialMatrix, FundamentalMatrix, FundamentalRansacOptions, GricComparison,
    ModelSelection, RelativePose, essential_from_fundamental, estimate_essential_ransac,
    estimate_fundamental_ransac, estimate_fundamental_ransac_with_options, fit_fundamental_7point,
    fit_fundamental_8point, recover_pose, rotation_error_deg, sampson_distance_sq,
    select_model_gric, triangulate_point_linear,
};
pub use pipeline::{
    ASSUMED_INPUT_SIGMA, BASE_SIGMA, DEFAULT_CONTRAST_THRESHOLD, DEFAULT_HOMOGRAPHY_RANSAC_SEED,
    DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD, DEFAULT_SIMILARITY_RANSAC_SEED,
    DEFAULT_UPSAMPLE_FIRST_OCTAVE, DescriptorKind, FeatureMatch, Homography, Homography2D,
    Keypoint, MAX_FEATURES_PER_IMAGE, MatchOptions, Matcher, Octave, Pyramid, RansacOptions,
    SIFT_DESCRIPTOR_LEN, Sift, SiftConfig, SiftFeature, SimilarityTransform2D, TrackingSummary,
    compute_sift_descriptors, compute_sift_descriptors_with_kind, dequantize_descriptor_u8,
    detect_keypoints, detect_keypoints_with_options, downsample_2x, estimate_homography_ransac,
    estimate_homography_ransac_with_seed, extract_extrema, extract_extrema_with_base_multiplier,
    extract_sift_features, extract_sift_features_with_kind, extract_sift_features_with_options,
    filter_matches_ransac, filter_matches_ransac_with_seed, fit_homography_dlt, gaussian_blur,
    gaussian_kernel_1d, gaussian_kernel_2d, generate_dog_space, generate_dog_space_with_options,
    luma_to_array2, match_descriptors, match_descriptors_with_options, quantize_descriptor_u8,
    summarize_tracking, to_rootsift_descriptor, upsample_2x_bilinear,
};
pub use stitch::{
    BlendMode, PanoramaOptions, PanoramaResult, ProjectionMode, solve_exposure_gains,
    stitch_panorama, warp_image_cylindrical,
};
pub use visualize::{
    MAX_WARP_DIM, MAX_WARP_PIXELS, blend_pixel, draw_keypoints_overlay, draw_matches_side_by_side,
    encode_rgba_png, invert_matrix_3x3, is_sane_homography_for_dims, project_f64,
    resolve_alignment_homography, resolve_safe_alignment_homography, sample_bilinear_rgba,
    warp_and_stitch,
};
