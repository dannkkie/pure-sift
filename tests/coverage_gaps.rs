use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use pure_sift::catalog::{DEFAULT_CATALOG_SHORTLIST, FeatureCatalog};
use pure_sift::cli::{CliCommand, VisionError, parse_cli_args};
use pure_sift::pipeline::{
    DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD, Homography2D,
    SimilarityTransform2D, extract_sift_features, luma_to_array2,
};
use pure_sift::visualize::{resolve_safe_alignment_homography, warp_and_stitch};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn make_textured_pattern(seed: u32, w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let block = ((x / 8 + seed) ^ (y / 8 + seed * 3)) & 1;
            let grad = ((x * (seed + 3) + y * (seed + 5)) % 97) as u8;
            let val = if block == 0 {
                30u8.saturating_add(grad)
            } else {
                200u8.saturating_sub(grad / 2)
            };
            img.put_pixel(x, y, Luma([val]));
        }
    }
    img
}

#[test]
fn test_catalog_search_ranking_ranks_matching_image_first() {
    let mut catalog = FeatureCatalog::new(3, 3, DEFAULT_CONTRAST_THRESHOLD);
    let img_a = make_textured_pattern(1, 128, 128);
    let img_b = make_textured_pattern(7, 128, 128);
    let img_c = make_textured_pattern(19, 128, 128);

    let feats_a = extract_sift_features(&luma_to_array2(&img_a), 3, 3, DEFAULT_CONTRAST_THRESHOLD);
    let feats_b = extract_sift_features(&luma_to_array2(&img_b), 3, 3, DEFAULT_CONTRAST_THRESHOLD);
    let feats_c = extract_sift_features(&luma_to_array2(&img_c), 3, 3, DEFAULT_CONTRAST_THRESHOLD);

    assert!(!feats_b.is_empty());
    catalog.upsert_entry("alpha", 128, 128, feats_a);
    catalog.upsert_entry("beta", 128, 128, feats_b.clone());
    catalog.upsert_entry("gamma", 128, 128, feats_c);

    let hits = catalog.search_features(
        &feats_b,
        3,
        DEFAULT_CATALOG_SHORTLIST,
        DEFAULT_MATCH_RATIO,
        DEFAULT_RANSAC_REPROJ_THRESHOLD,
    );
    assert!(!hits.is_empty(), "Expected catalog search hits");
    assert_eq!(
        hits[0].id, "beta",
        "Expected exact/best matching entry 'beta' to rank first, got {:?}",
        hits
    );
    assert!(hits[0].tracking.match_count > 0);
}

#[test]
fn test_catalog_atomic_save_and_load_leaves_no_temp_files() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("pure_sift_atomic_test_{nonce}"));
    fs::create_dir_all(&dir).unwrap();

    let img = make_textured_pattern(5, 96, 96);
    let feats = extract_sift_features(&luma_to_array2(&img), 3, 3, DEFAULT_CONTRAST_THRESHOLD);

    let mut catalog = FeatureCatalog::new(3, 3, DEFAULT_CONTRAST_THRESHOLD);
    catalog.upsert_entry("item-1", 96, 96, feats);

    let bin_path = dir.join("catalog.bin");
    catalog
        .save_binary(&bin_path)
        .expect("save_binary should succeed");

    let json_path = dir.join("catalog.json");
    catalog
        .save_json(&json_path, false)
        .expect("save_json should succeed");

    let loaded_bin = FeatureCatalog::load_auto(&bin_path).expect("load_auto bin should succeed");
    let loaded_json = FeatureCatalog::load_auto(&json_path).expect("load_auto json should succeed");
    assert_eq!(loaded_bin.entries.len(), 1);
    assert_eq!(loaded_json.entries.len(), 1);
    assert_eq!(loaded_bin.entries[0].id, "item-1");
    assert_eq!(loaded_json.entries[0].id, "item-1");

    // Ensure atomic write cleaned up any temporary files
    let leftover_tmp: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
        .collect();
    assert!(
        leftover_tmp.is_empty(),
        "Atomic save should not leave temporary files behind: {leftover_tmp:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_resolve_safe_alignment_homography_fallbacks() {
    let sane_h = Homography2D {
        matrix: [[1.0, 0.0, 12.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]],
    };
    let sim = SimilarityTransform2D {
        a: 1.0,
        b: 0.0,
        tx: 5.0,
        ty: -3.0,
        scale: 1.0,
        rotation_deg: 0.0,
    };

    // 1. Sane homography with >= 6 inliers is preserved
    let resolved = resolve_safe_alignment_homography(Some(sane_h), 6, Some(sim), 200, 200);
    assert_eq!(resolved, Some(sane_h));

    // 2. Too few homography inliers (< 6) falls back to similarity
    let fallback_low_inliers =
        resolve_safe_alignment_homography(Some(sane_h), 5, Some(sim), 200, 200);
    let resolved_sim = fallback_low_inliers.expect("Expected similarity fallback");
    assert_eq!(resolved_sim.matrix[0][2], 5.0);
    assert_eq!(resolved_sim.matrix[1][2], -3.0);

    // 3. Geometrically degenerate homography (projective horizon blow-up) falls back to similarity
    let degenerate_h = Homography2D {
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.05, 0.05, 1.0]],
    };
    let fallback_degenerate =
        resolve_safe_alignment_homography(Some(degenerate_h), 20, Some(sim), 200, 200);
    let resolved_deg = fallback_degenerate.expect("Expected similarity fallback for degenerate H");
    assert_eq!(resolved_deg.matrix[0][2], 5.0);

    // 4. Both absent -> None
    assert_eq!(
        resolve_safe_alignment_homography(None, 0, None, 200, 200),
        None
    );
}

#[test]
fn test_warp_and_stitch_size_limit_and_singular_errors() {
    let q = DynamicImage::ImageRgba8(RgbaImage::from_pixel(64, 64, Rgba([255, 0, 0, 255])));
    let t = DynamicImage::ImageRgba8(RgbaImage::from_pixel(64, 64, Rgba([0, 255, 0, 255])));

    // Singular homography matrix
    let singular_h = Homography2D {
        matrix: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let err_singular = warp_and_stitch(&q, &t, &singular_h).unwrap_err();
    assert!(
        err_singular.contains("singular"),
        "Expected singular matrix error, got: {err_singular}"
    );

    // Huge translation exceeding MAX_WARP_DIM / MAX_WARP_PIXELS
    let huge_h = Homography2D {
        matrix: [[1.0, 0.0, 50_000.0], [0.0, 1.0, 50_000.0], [0.0, 0.0, 1.0]],
    };
    let err_size = warp_and_stitch(&q, &t, &huge_h).unwrap_err();
    assert!(
        err_size.contains("limit") || err_size.contains("exceeds") || err_size.contains("max"),
        "Expected canvas size limit error, got: {err_size}"
    );
}

#[test]
fn test_cli_argument_parsing_edge_cases() {
    assert_eq!(parse_cli_args(["--help"]).unwrap(), CliCommand::Help(None));
    assert_eq!(parse_cli_args(["--version"]).unwrap(), CliCommand::Version);

    // `--` stops flag parsing so filenames starting with `-` work
    let cmd = parse_cli_args(["extract", "--", "-weird-name.png"]).unwrap();
    assert!(
        matches!(cmd, CliCommand::Extract(args) if args.image_path.to_str() == Some("-weird-name.png"))
    );

    // Out-of-range parameters are rejected
    assert!(matches!(
        parse_cli_args(["extract", "img.png", "--octaves=0"]),
        Err(VisionError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_cli_args(["extract", "img.png", "--scales=99"]),
        Err(VisionError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_cli_args(["extract", "img.png", "--threshold=0.0"]),
        Err(VisionError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_cli_args(["match", "a.png", "b.png", "--ratio=1.5"]),
        Err(VisionError::InvalidInput(_))
    ));

    // Missing required args / unknown flags
    assert!(matches!(
        parse_cli_args(["extract"]),
        Err(VisionError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_cli_args(["unknown-cmd"]),
        Err(VisionError::InvalidInput(_))
    ));
}
