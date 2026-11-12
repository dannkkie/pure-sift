use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use image::{GrayImage, ImageFormat, Luma};
use ndarray::Array2;
use pure_sift::{
    catalog::{FeatureCatalog, quantize_descriptor_u128},
    cli::{
        CatalogSearchResponse, CliCommand, FeatureResponse, MatchFeaturesResponse, parse_cli_args,
    },
    pipeline::{
        DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD,
        FeatureMatch, Homography2D, SIFT_DESCRIPTOR_LEN, SiftFeature, SimilarityTransform2D,
        detect_keypoints, estimate_homography_ransac, extract_extrema, extract_sift_features,
        filter_matches_ransac, gaussian_blur, gaussian_kernel_1d, gaussian_kernel_2d,
        generate_dog_space, luma_to_array2, match_descriptors, summarize_tracking,
    },
    visualize::{is_sane_homography_for_dims, resolve_safe_alignment_homography},
};

/// Constructs a deterministic 512x512 high-contrast geometric test pattern
/// featuring a centered checkerboard of squares with isolated corner and blob landmarks.
fn build_mock_checkerboard_512x512() -> Array2<f32> {
    let mut img = Array2::<f32>::zeros((512, 512));
    let cell_size = 32usize;

    for y in 64..448 {
        for x in 64..448 {
            let cell_y = (y - 64) / cell_size;
            let cell_x = (x - 64) / cell_size;
            if (cell_x + cell_y).is_multiple_of(2) {
                // Inset each active square slightly to form crisp 2D corners and scale-space extrema
                let local_y = (y - 64) % cell_size;
                let local_x = (x - 64) % cell_size;
                if (4..cell_size - 4).contains(&local_y) && (4..cell_size - 4).contains(&local_x) {
                    img[[y, x]] = 1.0;
                }
            }
        }
    }

    img
}

#[test]
fn test_checkerboard_512x512_under_50ms_and_deterministic() {
    let mock_frame = build_mock_checkerboard_512x512();
    let octaves = 3;
    let scales = 3;

    // Measure full pyramidal DoG generation + 3D extrema extraction on the 512x512 frame
    let start = Instant::now();
    let dog_space = generate_dog_space(&mock_frame, octaves, scales);
    let keypoints_first = extract_extrema(&dog_space, DEFAULT_CONTRAST_THRESHOLD);
    let elapsed = start.elapsed();

    println!(
        "512x512 DoG pipeline finished in {:?} with {} keypoints",
        elapsed,
        keypoints_first.len()
    );

    // 1. Assert sub-50ms execution budget
    assert!(
        elapsed < Duration::from_millis(50),
        "Expected 512x512 pipeline execution under 50ms, but took {:?}",
        elapsed
    );

    // 2. Assert keypoints were discovered on the high-contrast checkerboard geometry
    assert!(
        !keypoints_first.is_empty(),
        "Expected non-empty keypoints for 512x512 high-contrast checkerboard square"
    );

    // 3. Assert all keypoint coordinates and scale values lie within valid image/scale bounds
    for kp in &keypoints_first {
        assert!(
            kp.x >= 0.0 && kp.x < 512.0,
            "Keypoint x coordinate out of 512x512 bounds: {}",
            kp.x
        );
        assert!(
            kp.y >= 0.0 && kp.y < 512.0,
            "Keypoint y coordinate out of 512x512 bounds: {}",
            kp.y
        );
        assert!(
            kp.scale > 0.0,
            "Keypoint scale must be positive: {}",
            kp.scale
        );
        assert!(
            kp.response.abs() > DEFAULT_CONTRAST_THRESHOLD,
            "Keypoint response {} did not exceed threshold {}",
            kp.response,
            DEFAULT_CONTRAST_THRESHOLD
        );
    }

    // 4. Assert deterministic coordinate space across repeated executions
    let keypoints_second =
        detect_keypoints(&mock_frame, octaves, scales, DEFAULT_CONTRAST_THRESHOLD);
    assert_eq!(
        keypoints_first, keypoints_second,
        "Pipeline keypoint extraction must be strictly deterministic across runs"
    );

    // 5. Assert interior scale values across octaves follow sigma_0 * (2^octave) * k^(1 + ds)
    // with 3D Taylor sub-scale offset ds in [-0.5, 0.5].
    let k = 2.0f32.powf(1.0 / (scales as f32));
    let expected_scales = [1.0 * k, 2.0 * k, 4.0 * k];
    let min_factor = k.powf(-0.5) - 1e-4;
    let max_factor = k.powf(0.5) + 1e-4;
    for kp in &keypoints_first {
        assert!(
            expected_scales
                .iter()
                .any(|&s| kp.scale >= s * min_factor && kp.scale <= s * max_factor),
            "Unexpected keypoint scale {} (expected within sub-scale band of {:?})",
            kp.scale,
            expected_scales
        );
    }
}

#[test]
fn test_dog_pyramid_octave_downsampling_dimensions() {
    let mock_frame = build_mock_checkerboard_512x512();
    let octaves = 3;
    let scales = 3;

    let dog_space = generate_dog_space(&mock_frame, octaves, scales);
    assert_eq!(dog_space.len(), octaves * scales);

    // Octave 0: 512x512
    for slice in &dog_space[0..3] {
        assert_eq!(slice.dim(), (512, 512));
    }
    // Octave 1: 2x downsampled -> 256x256
    for slice in &dog_space[3..6] {
        assert_eq!(slice.dim(), (256, 256));
    }
    // Octave 2: 4x downsampled -> 128x128
    for slice in &dog_space[6..9] {
        assert_eq!(slice.dim(), (128, 128));
    }
}

#[test]
fn test_gaussian_kernel_and_separable_blur_properties() {
    let k1d = gaussian_kernel_1d(1.6);
    assert_eq!(k1d.len() % 2, 1, "1D Gaussian kernel must have odd length");

    let sum_1d: f32 = k1d.iter().sum();
    assert!(
        (sum_1d - 1.0).abs() < 1e-5,
        "1D Gaussian kernel weights must sum to 1.0, got {sum_1d}"
    );

    // Verify symmetry of 1D kernel
    for i in 0..k1d.len() / 2 {
        assert!(
            (k1d[i] - k1d[k1d.len() - 1 - i]).abs() < 1e-6,
            "1D Gaussian kernel must be symmetric"
        );
    }

    let k2d = gaussian_kernel_2d(1.6);
    let sum_2d: f32 = k2d.iter().sum();
    assert!(
        (sum_2d - 1.0).abs() < 1e-5,
        "2D Gaussian kernel weights must sum to 1.0, got {sum_2d}"
    );

    // Constant image must remain invariant under normalized Gaussian blur
    let uniform = Array2::from_elem((64, 64), 0.75f32);
    let blurred = gaussian_blur(&uniform, 1.6);
    for &val in blurred.iter() {
        assert!(
            (val - 0.75).abs() < 1e-4,
            "Constant image should remain constant after Gaussian blur, got {val}"
        );
    }
}

#[test]
fn test_png_decode_luma_to_array2_pipeline() {
    let mut gray = GrayImage::new(128, 128);
    for y in 32..96 {
        for x in 32..96 {
            if ((x / 16) + (y / 16)) % 2 == 0 {
                gray.put_pixel(x, y, Luma([255u8]));
            }
        }
    }

    let mut png_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_bytes);
    gray.write_to(&mut cursor, ImageFormat::Png)
        .expect("Failed to encode in-memory PNG");

    let decoded = image::load_from_memory(&png_bytes)
        .expect("Failed to decode in-memory PNG")
        .into_luma8();
    let arr = luma_to_array2(&decoded);

    assert_eq!(arr.dim(), (128, 128));
    let keypoints = detect_keypoints(&arr, 2, 3, DEFAULT_CONTRAST_THRESHOLD);
    assert!(!keypoints.is_empty());
}

#[test]
fn test_sift_128d_descriptor_invariants() {
    let mock_frame = build_mock_checkerboard_512x512();
    let features = extract_sift_features(&mock_frame, 3, 3, DEFAULT_CONTRAST_THRESHOLD);
    assert!(!features.is_empty());

    let two_pi = 2.0 * std::f32::consts::PI;
    for feat in &features {
        assert_eq!(feat.descriptor.len(), SIFT_DESCRIPTOR_LEN);
        assert!(
            feat.orientation >= 0.0 && feat.orientation < two_pi,
            "Orientation {} out of [0, 2*pi)",
            feat.orientation
        );

        let l2_norm: f32 = feat.descriptor.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (l2_norm - 1.0).abs() < 1e-4,
            "Expected unit L2 descriptor norm, got {l2_norm}"
        );
        assert!(
            feat.descriptor.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "Descriptor elements must be non-negative and normalized"
        );
    }
}

#[test]
fn test_sift_matching_and_tracking_under_translation() {
    // Construct an asymmetric multi-landmark scene on a 256x256 canvas,
    // and a translated target frame shifted by (dx = 24, dy = 16).
    let mut query_img = Array2::<f32>::zeros((256, 256));
    let mut target_img = Array2::<f32>::zeros((256, 256));
    let shift_x = 24usize;
    let shift_y = 16usize;

    // Asymmetric L-shapes, stepped blocks, and rings with distinct local gradient topologies
    let shapes: &[(usize, usize, usize, usize, f32)] = &[
        (40, 40, 28, 10, 1.0),
        (40, 40, 10, 28, 1.0),
        (110, 50, 20, 20, 0.9),
        (118, 58, 18, 8, 0.4),
        (55, 130, 14, 32, 0.85),
        (69, 148, 22, 14, 1.0),
        (140, 120, 30, 12, 0.95),
        (158, 132, 12, 26, 0.75),
    ];

    for &(x0, y0, w, h, val) in shapes {
        for y in y0..(y0 + h) {
            for x in x0..(x0 + w) {
                query_img[[y, x]] = val;
                target_img[[y + shift_y, x + shift_x]] = val;
            }
        }
    }

    let query_features = extract_sift_features(&query_img, 3, 3, DEFAULT_CONTRAST_THRESHOLD);
    let target_features = extract_sift_features(&target_img, 3, 3, DEFAULT_CONTRAST_THRESHOLD);

    assert!(!query_features.is_empty());
    assert!(!target_features.is_empty());

    let matches = match_descriptors(&query_features, &target_features, DEFAULT_MATCH_RATIO);
    assert!(
        !matches.is_empty(),
        "Expected unambiguous SIFT descriptor matches across translated frames"
    );

    let summary = summarize_tracking(&query_features, &target_features, &matches);
    assert!(
        (summary.mean_displacement.0 - (shift_x as f32)).abs() < 1.5,
        "Expected mean x displacement ~{}, got {}",
        shift_x,
        summary.mean_displacement.0
    );
    assert!(
        (summary.mean_displacement.1 - (shift_y as f32)).abs() < 1.5,
        "Expected mean y displacement ~{}, got {}",
        shift_y,
        summary.mean_displacement.1
    );
    assert!(
        (summary.mean_scale_ratio - 1.0).abs() < 0.1,
        "Expected scale ratio ~1.0, got {}",
        summary.mean_scale_ratio
    );
}

#[test]
fn test_ransac_outlier_rejection_and_similarity_recovery() {
    // Construct 10 true inlier correspondences for a crop offset of (-341.0, 0.0)
    // mixed with 4 gross repetitive-texture outliers.
    let mut query_feats = Vec::new();
    let mut target_feats = Vec::new();
    let mut raw_matches = Vec::new();

    for i in 0..10usize {
        let qx = 400.0 + (i as f32) * 85.0;
        let qy = 50.0 + ((i * 73) % 600) as f32;
        let tx = qx - 341.0;
        let ty = qy;

        query_feats.push(SiftFeature {
            x: qx,
            y: qy,
            scale: 1.26,
            response: 0.04,
            orientation: 1.0,
            descriptor: vec![0.0; SIFT_DESCRIPTOR_LEN],
        });
        target_feats.push(SiftFeature {
            x: tx,
            y: ty,
            scale: 1.26,
            response: 0.04,
            orientation: 1.0,
            descriptor: vec![0.0; SIFT_DESCRIPTOR_LEN],
        });
        raw_matches.push(FeatureMatch {
            query_index: i,
            target_index: i,
            query_point: (qx, qy),
            target_point: (tx, ty),
            distance: 0.02,
            ratio: 0.10,
            displacement: (-341.0, 0.0),
        });
    }

    // Add 4 gross outliers (+220, +180), (-90, -310), etc.
    let outlier_offsets = [
        (220.0f32, 180.0f32),
        (-90.0, -310.0),
        (410.0, -95.0),
        (150.0, 260.0),
    ];
    for (k, &(ox, oy)) in outlier_offsets.iter().enumerate() {
        let idx = 10 + k;
        let qx = 500.0 + (k as f32) * 60.0;
        let qy = 200.0 + (k as f32) * 50.0;
        let tx = qx + ox;
        let ty = qy + oy;

        query_feats.push(SiftFeature {
            x: qx,
            y: qy,
            scale: 1.26,
            response: 0.03,
            orientation: 2.5,
            descriptor: vec![0.0; SIFT_DESCRIPTOR_LEN],
        });
        target_feats.push(SiftFeature {
            x: tx,
            y: ty,
            scale: 2.52,
            response: 0.03,
            orientation: 0.2,
            descriptor: vec![0.0; SIFT_DESCRIPTOR_LEN],
        });
        raw_matches.push(FeatureMatch {
            query_index: idx,
            target_index: idx,
            query_point: (qx, qy),
            target_point: (tx, ty),
            distance: 0.28,
            ratio: 0.71,
            displacement: (ox, oy),
        });
    }

    let (inliers, transform) = filter_matches_ransac(&raw_matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
    assert_eq!(
        inliers.len(),
        10,
        "RANSAC must retain all 10 true inliers and reject all 4 outliers"
    );

    let model = transform.expect("Expected valid similarity transform");
    assert!((model.tx - (-341.0)).abs() < 1e-3);
    assert!(model.ty.abs() < 1e-3);
    assert!((model.scale - 1.0).abs() < 1e-3);
    assert!(model.rotation_deg.abs() < 1e-3);

    let summary = summarize_tracking(&query_feats, &target_feats, &raw_matches);
    assert_eq!(summary.raw_match_count, 14);
    assert_eq!(summary.match_count, 10);
    assert!((summary.mean_displacement.0 - (-341.0)).abs() < 1e-3);
    assert!(summary.mean_displacement.1.abs() < 1e-3);
    assert!((summary.median_displacement.0 - (-341.0)).abs() < 1e-3);
    assert!(summary.median_displacement.1.abs() < 1e-3);
}

#[test]
fn test_perspective_homography_ransac_recovery() {
    // True projective homography with non-zero perspective warp terms (h20, h21)
    let true_h = Homography2D {
        matrix: [
            [1.08, -0.04, 35.0],
            [0.03, 0.96, -18.0],
            [0.00012, -0.00008, 1.0],
        ],
    };

    let sample_pts = [
        (50.0f32, 60.0f32),
        (320.0, 45.0),
        (480.0, 110.0),
        (95.0, 240.0),
        (260.0, 275.0),
        (440.0, 310.0),
        (70.0, 460.0),
        (290.0, 490.0),
        (495.0, 470.0),
        (180.0, 170.0),
    ];

    let mut matches = Vec::new();
    for (i, &(qx, qy)) in sample_pts.iter().enumerate() {
        let (tx, ty) = true_h.project(qx, qy).expect("valid projection");
        matches.push(FeatureMatch {
            query_index: i,
            target_index: i,
            query_point: (qx, qy),
            target_point: (tx, ty),
            distance: 0.015,
            ratio: 0.08,
            displacement: (tx - qx, ty - qy),
        });
    }

    // Add 3 gross outliers that violate the projective homography
    for (k, &(qx, qy, tx, ty)) in [
        (120.0f32, 130.0f32, 490.0f32, 20.0f32),
        (350.0, 220.0, 15.0, 480.0),
        (200.0, 400.0, 420.0, 90.0),
    ]
    .iter()
    .enumerate()
    {
        let idx = sample_pts.len() + k;
        matches.push(FeatureMatch {
            query_index: idx,
            target_index: idx,
            query_point: (qx, qy),
            target_point: (tx, ty),
            distance: 0.31,
            ratio: 0.72,
            displacement: (tx - qx, ty - qy),
        });
    }

    let (inliers, homography_opt) =
        estimate_homography_ransac(&matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
    assert_eq!(
        inliers.len(),
        10,
        "Homography RANSAC must retain all 10 perspective inliers and reject all 3 outliers"
    );

    let est_h = homography_opt.expect("Expected valid 3x3 homography matrix");
    for &(qx, qy) in &sample_pts {
        let (true_tx, true_ty) = true_h.project(qx, qy).unwrap();
        let (est_tx, est_ty) = est_h.project(qx, qy).unwrap();
        let err = ((true_tx - est_tx).powi(2) + (true_ty - est_ty).powi(2)).sqrt();
        assert!(
            err < 0.05,
            "Homography reprojection error {err} exceeded 0.05 px at ({qx}, {qy})"
        );
    }
}

#[test]
fn test_cli_argument_parsing() {
    // Default (no args) => serve
    assert_eq!(
        parse_cli_args::<_, &str>([]).unwrap(),
        CliCommand::Serve { bind_addr: None }
    );

    // serve --bind
    assert_eq!(
        parse_cli_args(["serve", "--bind", "127.0.0.1:8080"]).unwrap(),
        CliCommand::Serve {
            bind_addr: Some("127.0.0.1:8080".to_string())
        }
    );

    // extract with mixed flag placement and --flag=val syntax
    let cmd = parse_cli_args([
        "extract",
        "--descriptors",
        "--threshold=0.03",
        "photo.jpg",
        "-o",
        "out.json",
        "--pretty",
    ])
    .unwrap();
    match cmd {
        CliCommand::Extract(args) => {
            assert_eq!(args.image_path, PathBuf::from("photo.jpg"));
            assert_eq!(args.output_path, Some(PathBuf::from("out.json")));
            assert!((args.threshold - 0.03).abs() < 1e-6);
            assert!(args.include_descriptors);
            assert!(args.pretty);
        }
        other => panic!("Expected Extract command, got {other:?}"),
    }

    // match with --no-ransac and custom ratio
    let cmd =
        parse_cli_args(["match", "q.png", "t.png", "--ratio", "0.65", "--no-ransac"]).unwrap();
    match cmd {
        CliCommand::Match(args) => {
            assert_eq!(args.query_path, PathBuf::from("q.png"));
            assert_eq!(args.target_path, PathBuf::from("t.png"));
            assert!((args.ratio - 0.65).abs() < 1e-6);
            assert!(!args.ransac_enabled);
        }
        other => panic!("Expected Match command, got {other:?}"),
    }

    // Invalid scales rejected cleanly
    assert!(parse_cli_args(["extract", "photo.jpg", "--scales", "1"]).is_err());
    assert!(parse_cli_args(["unknown-cmd"]).is_err());
}

#[test]
fn test_cli_binary_extract_and_match_end_to_end() {
    let bin_path = env!("CARGO_BIN_EXE_pure-sift");

    // 1. Verify --help exits 0
    let help_out = Command::new(bin_path)
        .arg("--help")
        .output()
        .expect("Failed to execute binary --help");
    assert!(help_out.status.success());
    let help_text = String::from_utf8_lossy(&help_out.stdout);
    assert!(help_text.contains("extract <IMAGE>"));
    assert!(help_text.contains("match <QUERY> <TARGET>"));

    // 2. Create deterministic query and shifted target PNG files in temp_dir
    let temp_dir = std::env::temp_dir().join("pure_sift_cli_test");
    fs::create_dir_all(&temp_dir).expect("Failed to create temp test dir");

    let query_path = temp_dir.join("query.png");
    let target_path = temp_dir.join("target.png");
    let extract_json_path = temp_dir.join("extract.json");
    let match_json_path = temp_dir.join("match.json");

    let mut q_gray = GrayImage::new(200, 200);
    let mut t_gray = GrayImage::new(200, 200);
    let shift_x = 18u32;
    let shift_y = 12u32;

    let shapes: &[(u32, u32, u32, u32, u8)] = &[
        (30, 30, 26, 10, 255),
        (30, 30, 10, 26, 255),
        (95, 42, 18, 18, 220),
        (45, 110, 14, 28, 235),
        (59, 124, 20, 12, 255),
        (115, 100, 24, 10, 240),
        (129, 110, 10, 24, 200),
    ];

    for &(x0, y0, w, h, val) in shapes {
        for y in y0..(y0 + h) {
            for x in x0..(x0 + w) {
                q_gray.put_pixel(x, y, Luma([val]));
                t_gray.put_pixel(x + shift_x, y + shift_y, Luma([val]));
            }
        }
    }

    q_gray.save(&query_path).expect("Failed to save query PNG");
    t_gray
        .save(&target_path)
        .expect("Failed to save target PNG");

    // 3. Run CLI `extract` writing JSON file with descriptors
    let extract_out = Command::new(bin_path)
        .arg("extract")
        .arg(&query_path)
        .arg("--descriptors")
        .arg("--pretty")
        .arg("-o")
        .arg(&extract_json_path)
        .output()
        .expect("Failed to execute CLI extract");
    assert!(
        extract_out.status.success(),
        "CLI extract failed: {}",
        String::from_utf8_lossy(&extract_out.stderr)
    );

    let extract_raw = fs::read_to_string(&extract_json_path).expect("Missing extract output JSON");
    let feat_resp: FeatureResponse =
        serde_json::from_str(&extract_raw).expect("Invalid FeatureResponse JSON");
    assert_eq!((feat_resp.width, feat_resp.height), (200, 200));
    assert!(feat_resp.keypoint_count > 0);
    let feats = feat_resp
        .features
        .expect("Expected 128-D descriptors with --descriptors");
    assert!(
        feats.len() >= feat_resp.keypoint_count,
        "Expected feature count ({}) >= keypoint count ({}) with >=80% secondary orientation splitting",
        feats.len(),
        feat_resp.keypoint_count
    );

    // 4. Run CLI `match` writing JSON file
    let match_out = Command::new(bin_path)
        .arg("match")
        .arg(&query_path)
        .arg(&target_path)
        .arg("-o")
        .arg(&match_json_path)
        .output()
        .expect("Failed to execute CLI match");
    assert!(
        match_out.status.success(),
        "CLI match failed: {}",
        String::from_utf8_lossy(&match_out.stderr)
    );

    let match_raw = fs::read_to_string(&match_json_path).expect("Missing match output JSON");
    let match_resp: MatchFeaturesResponse =
        serde_json::from_str(&match_raw).expect("Invalid MatchFeaturesResponse JSON");
    assert!(match_resp.tracking.match_count > 0);
    assert!(
        (match_resp.tracking.median_displacement.0 - (shift_x as f32)).abs() < 1.5,
        "Expected CLI match dx ~{shift_x}, got {}",
        match_resp.tracking.median_displacement.0
    );
    assert!(
        (match_resp.tracking.median_displacement.1 - (shift_y as f32)).abs() < 1.5,
        "Expected CLI match dy ~{shift_y}, got {}",
        match_resp.tracking.median_displacement.1
    );

    // 5. Run CLI `visualize`, `visualize-matches`, and `align`
    let vis_kp_path = temp_dir.join("keypoints.png");
    let vis_matches_path = temp_dir.join("matches.png");
    let aligned_path = temp_dir.join("aligned.png");

    let vis_out = Command::new(bin_path)
        .arg("visualize")
        .arg(&query_path)
        .arg("-o")
        .arg(&vis_kp_path)
        .output()
        .expect("Failed to execute CLI visualize");
    assert!(
        vis_out.status.success(),
        "CLI visualize failed: {}",
        String::from_utf8_lossy(&vis_out.stderr)
    );
    let vis_img = image::open(&vis_kp_path).expect("Failed to open rendered keypoints PNG");
    assert_eq!((vis_img.width(), vis_img.height()), (200, 200));

    let vis_m_out = Command::new(bin_path)
        .arg("visualize-matches")
        .arg(&query_path)
        .arg(&target_path)
        .arg("-o")
        .arg(&vis_matches_path)
        .output()
        .expect("Failed to execute CLI visualize-matches");
    assert!(
        vis_m_out.status.success(),
        "CLI visualize-matches failed: {}",
        String::from_utf8_lossy(&vis_m_out.stderr)
    );
    let vis_m_img = image::open(&vis_matches_path).expect("Failed to open matches PNG");
    assert_eq!((vis_m_img.width(), vis_m_img.height()), (400, 200));

    let align_out = Command::new(bin_path)
        .arg("align")
        .arg(&query_path)
        .arg(&target_path)
        .arg("-o")
        .arg(&aligned_path)
        .output()
        .expect("Failed to execute CLI align");
    assert!(
        align_out.status.success(),
        "CLI align failed: {}",
        String::from_utf8_lossy(&align_out.stderr)
    );
    let aligned_img = image::open(&aligned_path).expect("Failed to open aligned PNG");
    assert!(
        aligned_img.width() >= 200 && aligned_img.height() >= 200,
        "Expected stitched canvas >= 200x200, got {}x{}",
        aligned_img.width(),
        aligned_img.height()
    );

    // 6. Verify invalid flag returns non-zero exit code
    let bad_out = Command::new(bin_path)
        .args(["extract", "--scales", "1", query_path.to_str().unwrap()])
        .output()
        .expect("Failed to run invalid CLI args check");
    assert!(!bad_out.status.success());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_cli_visual_command_parsing() {
    let vis_cmd = parse_cli_args([
        "visualize",
        "img.png",
        "--threshold",
        "0.025",
        "-o",
        "kp.png",
    ])
    .expect("Failed to parse visualize args");
    match vis_cmd {
        CliCommand::Visualize(args) => {
            assert_eq!(args.image_path, PathBuf::from("img.png"));
            assert_eq!(args.output_path, PathBuf::from("kp.png"));
            assert!((args.threshold - 0.025).abs() < 1e-6);
        }
        other => panic!("Expected Visualize command, got {other:?}"),
    }

    let stitch_cmd = parse_cli_args(["stitch", "a.jpg", "b.jpg", "--ratio=0.7", "-o", "pano.png"])
        .expect("Failed to parse stitch alias args");
    match stitch_cmd {
        CliCommand::Align(args) => {
            assert_eq!(args.query_path, PathBuf::from("a.jpg"));
            assert_eq!(args.target_path, PathBuf::from("b.jpg"));
            assert_eq!(args.output_path, PathBuf::from("pano.png"));
            assert!((args.ratio - 0.7).abs() < 1e-6);
        }
        other => panic!("Expected Align command, got {other:?}"),
    }
}

fn render_synthetic_scene(seed: usize, shift_x: usize, shift_y: usize) -> GrayImage {
    let mut img = GrayImage::new(220, 220);
    let base_shapes: &[(usize, usize, usize, usize, u8)] = match seed {
        0 => &[
            (25, 25, 28, 10, 255),
            (25, 25, 10, 28, 255),
            (90, 40, 22, 22, 220),
            (40, 110, 14, 30, 240),
            (54, 126, 20, 12, 255),
            (120, 95, 24, 10, 235),
            (134, 105, 10, 24, 210),
        ],
        1 => &[
            (30, 35, 14, 14, 250),
            (60, 30, 12, 34, 215),
            (95, 65, 30, 12, 245),
            (135, 35, 16, 26, 230),
            (45, 130, 28, 12, 255),
            (115, 125, 22, 22, 200),
        ],
        _ => &[
            (20, 50, 32, 8, 240),
            (44, 58, 8, 32, 240),
            (100, 25, 16, 16, 255),
            (140, 75, 24, 24, 190),
            (70, 120, 26, 10, 220),
            (30, 150, 18, 18, 250),
        ],
    };

    for &(x0, y0, w, h, val) in base_shapes {
        for y in y0..(y0 + h) {
            for x in x0..(x0 + w) {
                let px = (x + shift_x) as u32;
                let py = (y + shift_y) as u32;
                if px < 220 && py < 220 {
                    img.put_pixel(px, py, Luma([val]));
                }
            }
        }
    }
    img
}

#[test]
fn test_feature_catalog_hamming_shortlist_and_ransac_ranking() {
    let mut catalog = FeatureCatalog::new(3, 3, DEFAULT_CONTRAST_THRESHOLD);

    for scene_id in 0..3usize {
        let gray = render_synthetic_scene(scene_id, 0, 0);
        let arr = luma_to_array2(&gray);
        let feats = extract_sift_features(&arr, 3, 3, DEFAULT_CONTRAST_THRESHOLD);
        assert!(!feats.is_empty());
        assert_ne!(quantize_descriptor_u128(&feats[0].descriptor), 0);
        catalog.upsert_entry(format!("scene_{scene_id}.png"), 220, 220, feats);
    }

    // Query is scene 0 shifted by (16, 14); use shortlist_k=2 to force 128-bit Hamming prefiltering
    let query_gray = render_synthetic_scene(0, 16, 14);
    let query_arr = luma_to_array2(&query_gray);
    let query_feats = extract_sift_features(&query_arr, 3, 3, DEFAULT_CONTRAST_THRESHOLD);

    let hits = catalog.search_features(
        &query_feats,
        3,
        2,
        DEFAULT_MATCH_RATIO,
        DEFAULT_RANSAC_REPROJ_THRESHOLD,
    );
    assert!(
        !hits.is_empty(),
        "Expected catalog search to return matches"
    );
    assert_eq!(
        hits[0].id,
        "scene_0.png",
        "Expected scene_0.png as top-ranked match, got {:?}",
        hits.iter()
            .map(|h| (&h.id, h.tracking.match_count, h.coarse_votes))
            .collect::<Vec<_>>()
    );
    assert!(hits[0].tracking.match_count > 0);
}

#[test]
fn test_cli_catalog_build_and_search_end_to_end() {
    let bin_path = env!("CARGO_BIN_EXE_pure-sift");
    let temp_dir = std::env::temp_dir().join("pure_sift_catalog_cli_test");
    let images_dir = temp_dir.join("images");
    fs::create_dir_all(&images_dir).expect("Failed to create temp catalog dir");

    for scene_id in 0..3usize {
        let img = render_synthetic_scene(scene_id, 0, 0);
        img.save(images_dir.join(format!("item_{scene_id}.png")))
            .expect("Failed to save catalog item");
    }

    let query_path = temp_dir.join("query_shifted.png");
    render_synthetic_scene(1, 14, 10)
        .save(&query_path)
        .expect("Failed to save query image");

    let catalog_path = temp_dir.join("catalog.json");
    let search_out_path = temp_dir.join("search_results.json");

    let build_out = Command::new(bin_path)
        .arg("catalog-build")
        .arg(&images_dir)
        .arg("-o")
        .arg(&catalog_path)
        .output()
        .expect("Failed to run catalog-build");
    assert!(
        build_out.status.success(),
        "catalog-build failed: {}",
        String::from_utf8_lossy(&build_out.stderr)
    );
    assert!(catalog_path.exists());

    let search_out = Command::new(bin_path)
        .arg("catalog-search")
        .arg(&catalog_path)
        .arg(&query_path)
        .arg("--top-k")
        .arg("2")
        .arg("--pretty")
        .arg("-o")
        .arg(&search_out_path)
        .output()
        .expect("Failed to run catalog-search");
    assert!(
        search_out.status.success(),
        "catalog-search failed: {}",
        String::from_utf8_lossy(&search_out.stderr)
    );

    let search_json = fs::read_to_string(&search_out_path).expect("Missing search output JSON");
    let resp: CatalogSearchResponse =
        serde_json::from_str(&search_json).expect("Invalid CatalogSearchResponse JSON");
    assert_eq!(resp.catalog_size, 3);
    assert!(!resp.hits.is_empty());
    assert_eq!(resp.hits[0].id, "item_1.png");

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_3d_taylor_subpixel_extremum_refinement() {
    // Construct a synthetic continuous 2D Gaussian blob centered at sub-pixel coordinate (32.35, 28.65)
    // with characteristic scale sigma = sqrt(2) * BASE_SIGMA = 1.4142 so its scale-space DoG peak
    // lands directly in the interior DoG slice s = 1 of Octave 0.
    let true_cx = 32.35f32;
    let true_cy = 28.65f32;
    let blob_sigma = std::f32::consts::SQRT_2;
    let two_sigma_sq = 2.0 * blob_sigma * blob_sigma;

    let mut img = Array2::<f32>::zeros((64, 64));
    for y in 0..64 {
        for x in 0..64 {
            let dx = (x as f32) - true_cx;
            let dy = (y as f32) - true_cy;
            img[[y, x]] = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
        }
    }

    let keypoints = detect_keypoints(&img, 1, 3, DEFAULT_CONTRAST_THRESHOLD);
    assert!(
        !keypoints.is_empty(),
        "Expected keypoint at Gaussian blob peak"
    );

    // Find the keypoint closest to (true_cx, true_cy)
    let best_kp = keypoints
        .iter()
        .min_by(|a, b| {
            let da = (a.x - true_cx).hypot(a.y - true_cy);
            let db = (b.x - true_cx).hypot(b.y - true_cy);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    let subpixel_err = (best_kp.x - true_cx).hypot(best_kp.y - true_cy);
    let integer_grid_err = (32.0f32 - true_cx).hypot(29.0f32 - true_cy);

    assert!(
        subpixel_err < 0.12,
        "Expected 3D Taylor sub-pixel localization error < 0.12 px (vs integer error {:.3} px), got {:.4} px at ({:.3}, {:.3})",
        integer_grid_err,
        subpixel_err,
        best_kp.x,
        best_kp.y
    );
}

#[test]
fn test_secondary_orientation_splitting_80_percent_peak() {
    // Construct a symmetric 4-fold cross pattern whose corners produce multiple equal-strength
    // orthogonal gradient orientation peaks (>= 80% of dominant peak).
    let mut img = Array2::<f32>::zeros((96, 96));
    for y in 24..72 {
        for x in 42..54 {
            img[[y, x]] = 1.0;
        }
    }
    for y in 42..54 {
        for x in 24..72 {
            img[[y, x]] = 1.0;
        }
    }

    let keypoints = detect_keypoints(&img, 2, 3, DEFAULT_CONTRAST_THRESHOLD);
    let features = extract_sift_features(&img, 2, 3, DEFAULT_CONTRAST_THRESHOLD);

    assert!(!keypoints.is_empty());
    assert!(
        features.len() > keypoints.len(),
        "Expected secondary orientation splitting (>=80% peak) to emit more SiftFeatures ({}) than Keypoints ({})",
        features.len(),
        keypoints.len()
    );

    // Verify at least one keypoint coordinate has >= 2 distinct canonical orientations
    let has_split_orientations = keypoints.iter().any(|kp| {
        let matching_feats: Vec<&SiftFeature> = features
            .iter()
            .filter(|f| {
                (f.x - kp.x).abs() < 1e-5
                    && (f.y - kp.y).abs() < 1e-5
                    && (f.scale - kp.scale).abs() < 1e-5
            })
            .collect();
        matching_feats.len() >= 2
            && (matching_feats[0].orientation - matching_feats[1].orientation).abs() > 0.1
    });
    assert!(
        has_split_orientations,
        "Expected at least one keypoint to split into multiple distinct canonical orientations"
    );
}

#[test]
fn test_safe_alignment_homography_rejects_degenerate_rank1_warp() {
    // Rank-1 / horizon-crossing homography similar to low-texture collinear collapse
    let degenerate = Homography2D {
        matrix: [
            [-0.7689, -0.2804, 1340.6],
            [-0.6761, -0.2462, 1178.4],
            [-0.000574, -0.000209, 1.0],
        ],
    };
    assert!(!is_sane_homography_for_dims(&degenerate, 1024, 1024));

    let valid_sim = SimilarityTransform2D {
        a: 0.99,
        b: -0.05,
        tx: 12.0,
        ty: -8.0,
        scale: 0.99126,
        rotation_deg: -2.89,
    };

    // Should reject the degenerate Homography and fall back cleanly to SimilarityTransform2D
    let chosen =
        resolve_safe_alignment_homography(Some(degenerate), 8, Some(valid_sim), 1024, 1024)
            .expect("Expected similarity fallback");
    assert!((chosen.matrix[2][0]).abs() < 1e-6);
    assert!((chosen.matrix[0][2] - 12.0).abs() < 1e-4);
}
