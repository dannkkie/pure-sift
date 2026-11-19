#![forbid(unsafe_code)]

pub mod catalog;
pub mod cli;
pub mod pipeline;
pub mod stitch;
pub mod visualize;

pub use catalog::{
    CatalogEntry, CatalogEntrySummary, CatalogSearchHit, CompactSiftFeature,
    DEFAULT_CATALOG_SHORTLIST, DEFAULT_CATALOG_TOP_K, FeatureCatalog, quantize_descriptor_u128,
};
pub use cli::{
    AlignArgs, CatalogBuildArgs, CatalogSearchArgs, CatalogSearchOptions, CatalogSearchResponse,
    CliCommand, DEFAULT_OCTAVES, DEFAULT_SCALES, ExtractArgs, FeatureResponse, MAX_IMAGE_DIM,
    MAX_IMAGE_PIXELS, MAX_OCTAVES, MAX_SCALES, MIN_IMAGE_DIM, MIN_OCTAVES, MIN_SCALES, MatchArgs,
    MatchFeaturesResponse, OutputFormat, PairPipelineOptions, PanoramaArgs, VisionError,
    VisualizeArgs, VisualizeMatchesArgs, decode_validated_dynamic_image, decode_validated_luma8,
    execute_cli_command, execute_cli_to_stdout, map_image_prefix, parse_cli_args,
    run_align_images_rgba, run_catalog_search_from_bytes, run_extract_from_bytes,
    run_match_from_bytes, run_visualize_features_rgba, run_visualize_matches_rgba,
    save_rgba_output, serialize_extract_csv, serialize_json, serialize_match_csv,
    validate_match_params, validate_pipeline_params, write_json_output,
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
