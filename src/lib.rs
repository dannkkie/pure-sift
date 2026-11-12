#![forbid(unsafe_code)]

pub mod catalog;
pub mod cli;
pub mod pipeline;
pub mod visualize;

pub use catalog::{
    CatalogEntry, CatalogEntrySummary, CatalogSearchHit, DEFAULT_CATALOG_SHORTLIST,
    DEFAULT_CATALOG_TOP_K, FeatureCatalog, quantize_descriptor_u128,
};
pub use cli::{
    AlignArgs, CatalogBuildArgs, CatalogSearchArgs, CatalogSearchResponse, CliCommand,
    DEFAULT_OCTAVES, DEFAULT_SCALES, ExtractArgs, FeatureResponse, MAX_IMAGE_DIM, MAX_IMAGE_PIXELS,
    MAX_OCTAVES, MAX_SCALES, MIN_IMAGE_DIM, MIN_OCTAVES, MIN_SCALES, MatchArgs,
    MatchFeaturesResponse, VisionError, VisualizeArgs, VisualizeMatchesArgs,
    decode_validated_dynamic_image, decode_validated_luma8, execute_cli_command,
    execute_cli_to_stdout, map_image_prefix, parse_cli_args, run_align_images_rgba,
    run_catalog_search_from_bytes, run_extract_from_bytes, run_match_from_bytes,
    run_visualize_features_rgba, run_visualize_matches_rgba, save_rgba_output, serialize_json,
    validate_match_params, validate_pipeline_params, write_json_output,
};
pub use pipeline::{
    BASE_SIGMA, DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD,
    FeatureMatch, Homography2D, Keypoint, MAX_FEATURES_PER_IMAGE, SIFT_DESCRIPTOR_LEN, SiftFeature,
    SimilarityTransform2D, TrackingSummary, compute_sift_descriptors, detect_keypoints,
    downsample_2x, estimate_homography_ransac, extract_extrema, extract_sift_features,
    filter_matches_ransac, fit_homography_dlt, gaussian_blur, gaussian_kernel_1d,
    gaussian_kernel_2d, generate_dog_space, luma_to_array2, match_descriptors, summarize_tracking,
};
pub use visualize::{
    MAX_WARP_DIM, MAX_WARP_PIXELS, blend_pixel, draw_keypoints_overlay, draw_matches_side_by_side,
    encode_rgba_png, invert_matrix_3x3, is_sane_homography_for_dims, project_f64,
    resolve_alignment_homography, resolve_safe_alignment_homography, sample_bilinear_rgba,
    warp_and_stitch,
};
