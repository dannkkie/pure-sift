use std::time::Instant;

use image::{GrayImage, Luma, imageops::FilterType};
use pure_sift::{
    DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD,
    extract_sift_features, filter_matches_ransac, luma_to_array2, match_descriptors,
};

const SAMPLE_PATH: &str = "docs/samples/aerial_west.jpg";
const SWEEP_OCTAVES: usize = 4;
const SWEEP_SCALES: usize = 3;

/// Baseline minimum RANSAC similarity inliers for the scale sweep on `aerial_west.jpg`.
const SCALE_SWEEP_BASELINES: &[(f32, usize)] = &[
    (0.25, 150),
    (0.35, 340),
    (0.50, 750),
    (0.70, 1400),
    (1.40, 900),
    (2.00, 800),
];

/// Baseline minimum RANSAC similarity inliers for the rotation sweep on `aerial_west.jpg`.
const ROTATION_SWEEP_BASELINES: &[(f32, usize)] = &[
    (0.0, 3000),
    (15.0, 1900),
    (30.0, 1500),
    (45.0, 1500),
    (90.0, 2500),
    (180.0, 2500),
];

/// Rotates a grayscale image counter-clockwise around its center by `angle_deg` using bilinear
/// sampling, preserving dimensions.
fn rotate_bilinear(src: &GrayImage, angle_deg: f32) -> GrayImage {
    let w = src.width();
    let h = src.height();
    let cx = 0.5 * ((w as f32) - 1.0);
    let cy = 0.5 * ((h as f32) - 1.0);
    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let mut dst = GrayImage::new(w, h);
    for y in 0..h {
        let dy = (y as f32) - cy;
        for x in 0..w {
            let dx = (x as f32) - cx;
            // Inverse mapping from destination (x, y) to source (sx, sy) for CCW rotation
            let sx = cx + dx * cos_a + dy * sin_a;
            let sy = cy - dx * sin_a + dy * cos_a;

            if sx >= 0.0 && sx <= ((w - 1) as f32) && sy >= 0.0 && sy <= ((h - 1) as f32) {
                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);
                let fx = sx - (x0 as f32);
                let fy = sy - (y0 as f32);

                let p00 = src.get_pixel(x0, y0)[0] as f32;
                let p10 = src.get_pixel(x1, y0)[0] as f32;
                let p01 = src.get_pixel(x0, y1)[0] as f32;
                let p11 = src.get_pixel(x1, y1)[0] as f32;

                let val =
                    (1.0 - fy) * ((1.0 - fx) * p00 + fx * p10) + fy * ((1.0 - fx) * p01 + fx * p11);
                dst.put_pixel(x, y, Luma([val.round().clamp(0.0, 255.0) as u8]));
            }
        }
    }
    dst
}

fn angular_diff_deg(a: f32, b: f32) -> f32 {
    let mut d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    d.abs()
}

#[test]
fn test_scale_sweep_invariance() {
    let start = Instant::now();
    let base_luma = image::open(SAMPLE_PATH)
        .expect("Failed to open sample image")
        .into_luma8();
    let base_arr = luma_to_array2(&base_luma);
    let base_feats = extract_sift_features(
        &base_arr,
        SWEEP_OCTAVES,
        SWEEP_SCALES,
        DEFAULT_CONTRAST_THRESHOLD,
    );
    assert!(!base_feats.is_empty());

    for &(scale_factor, min_inliers) in SCALE_SWEEP_BASELINES {
        let new_w = ((base_luma.width() as f32) * scale_factor).round() as u32;
        let new_h = ((base_luma.height() as f32) * scale_factor).round() as u32;
        let resized = image::imageops::resize(&base_luma, new_w, new_h, FilterType::CatmullRom);
        let resized_arr = luma_to_array2(&resized);

        let target_feats = extract_sift_features(
            &resized_arr,
            SWEEP_OCTAVES,
            SWEEP_SCALES,
            DEFAULT_CONTRAST_THRESHOLD,
        );
        let raw_matches = match_descriptors(&base_feats, &target_feats, DEFAULT_MATCH_RATIO);
        let (inliers, sim_opt) =
            filter_matches_ransac(&raw_matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);

        let sim = sim_opt.unwrap_or_else(|| {
            panic!("Scale {scale_factor}x failed to produce a SimilarityTransform2D")
        });
        let rel_scale_err = (sim.scale - scale_factor).abs() / scale_factor;
        let rot_err = angular_diff_deg(sim.rotation_deg, 0.0);

        println!(
            "Scale {scale_factor:.2}x ({new_w}x{new_h}): inliers={} (min={min_inliers}), est_scale={:.4} (err={:.2}%), est_rot={:.2}deg",
            inliers.len(),
            sim.scale,
            rel_scale_err * 100.0,
            sim.rotation_deg
        );

        assert!(
            inliers.len() >= min_inliers,
            "Scale {scale_factor:.2}x inliers {} < baseline {min_inliers}",
            inliers.len()
        );
        assert!(
            rel_scale_err <= 0.03,
            "Scale {scale_factor:.2}x relative scale error {:.4} > 3%",
            rel_scale_err
        );
        assert!(
            rot_err <= 2.0,
            "Scale {scale_factor:.2}x rotation error {rot_err:.2} deg > 2 deg"
        );
    }

    if std::env::var_os("LLVM_PROFILE_FILE").is_none() {
        assert!(
            start.elapsed().as_secs_f32() < 30.0,
            "Scale sweep exceeded 30s CI budget"
        );
    }
}

#[test]
fn test_rotation_sweep_invariance() {
    let start = Instant::now();
    let base_luma = image::open(SAMPLE_PATH)
        .expect("Failed to open sample image")
        .into_luma8();
    let base_arr = luma_to_array2(&base_luma);
    let base_feats = extract_sift_features(
        &base_arr,
        SWEEP_OCTAVES,
        SWEEP_SCALES,
        DEFAULT_CONTRAST_THRESHOLD,
    );
    assert!(!base_feats.is_empty());

    for &(angle_deg, min_inliers) in ROTATION_SWEEP_BASELINES {
        let rotated = rotate_bilinear(&base_luma, angle_deg);
        let rotated_arr = luma_to_array2(&rotated);

        let target_feats = extract_sift_features(
            &rotated_arr,
            SWEEP_OCTAVES,
            SWEEP_SCALES,
            DEFAULT_CONTRAST_THRESHOLD,
        );
        let raw_matches = match_descriptors(&base_feats, &target_feats, DEFAULT_MATCH_RATIO);
        let (inliers, sim_opt) =
            filter_matches_ransac(&raw_matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);

        let sim = sim_opt.unwrap_or_else(|| {
            panic!("Rotation {angle_deg}deg failed to produce a SimilarityTransform2D")
        });
        let expected_rot = angle_deg.rem_euclid(360.0);
        let est_rot = sim.rotation_deg.rem_euclid(360.0);
        let rot_err = angular_diff_deg(est_rot, expected_rot);
        let rel_scale_err = (sim.scale - 1.0).abs();

        println!(
            "Rotation {angle_deg:.0}deg: inliers={} (min={min_inliers}), est_rot={:.2}deg (err={rot_err:.2}deg), est_scale={:.4}",
            inliers.len(),
            sim.rotation_deg,
            sim.scale
        );

        assert!(
            inliers.len() >= min_inliers,
            "Rotation {angle_deg:.0}deg inliers {} < baseline {min_inliers}",
            inliers.len()
        );
        assert!(
            rot_err <= 2.0,
            "Rotation {angle_deg:.0}deg rotation error {rot_err:.2} deg > 2 deg (est={est_rot:.2}, expected={expected_rot:.2})"
        );
        assert!(
            rel_scale_err <= 0.03,
            "Rotation {angle_deg:.0}deg scale error {rel_scale_err:.4} > 3%"
        );
    }

    if std::env::var_os("LLVM_PROFILE_FILE").is_none() {
        assert!(
            start.elapsed().as_secs_f32() < 30.0,
            "Rotation sweep exceeded 30s CI budget"
        );
    }
}
