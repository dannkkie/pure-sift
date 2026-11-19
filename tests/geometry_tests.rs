use pure_sift::geometry::{
    CameraIntrinsics, FundamentalRansacOptions, ModelSelection, estimate_essential_ransac,
    estimate_fundamental_ransac, fit_fundamental_7point, fit_fundamental_8point, recover_pose,
    rotation_error_deg, sampson_distance_sq, select_model_gric, triangulate_point_linear,
};
use pure_sift::pipeline::{
    DEFAULT_RANSAC_REPROJ_THRESHOLD, FeatureMatch, estimate_homography_ransac,
};

fn euler_zyx_rotation(yaw_deg: f64, pitch_deg: f64, roll_deg: f64) -> [[f64; 3]; 3] {
    let (sz, cz) = yaw_deg.to_radians().sin_cos();
    let (sy, cy) = pitch_deg.to_radians().sin_cos();
    let (sx, cx) = roll_deg.to_radians().sin_cos();

    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];

    let mut ryx = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            ryx[r][c] = ry[r][0] * rx[0][c] + ry[r][1] * rx[1][c] + ry[r][2] * rx[2][c];
        }
    }
    let mut r_out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            r_out[r][c] = rz[r][0] * ryx[0][c] + rz[r][1] * ryx[1][c] + rz[r][2] * ryx[2][c];
        }
    }
    r_out
}

fn project_point(k: &CameraIntrinsics, r: &[[f64; 3]; 3], t: [f64; 3], pt: [f64; 3]) -> (f64, f64) {
    let x_c = r[0][0] * pt[0] + r[0][1] * pt[1] + r[0][2] * pt[2] + t[0];
    let y_c = r[1][0] * pt[0] + r[1][1] * pt[1] + r[1][2] * pt[2] + t[1];
    let z_c = r[2][0] * pt[0] + r[2][1] * pt[1] + r[2][2] * pt[2] + t[2];
    (k.fx * (x_c / z_c) + k.cx, k.fy * (y_c / z_c) + k.cy)
}

fn make_3d_correspondences(
    n_inliers: usize,
    n_outliers: usize,
    k: &CameraIntrinsics,
    r: &[[f64; 3]; 3],
    t: [f64; 3],
) -> (Vec<[f64; 3]>, Vec<FeatureMatch>) {
    let mut pts_3d = Vec::with_capacity(n_inliers);
    let mut matches = Vec::with_capacity(n_inliers + n_outliers);
    let r_id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let mut state = 0x1234_ABCD_5678_90EFu64;
    let mut next_u01 = || -> f64 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for i in 0..n_inliers {
        let x = (next_u01() - 0.5) * 4.0;
        let y = (next_u01() - 0.5) * 3.5;
        let z = 4.0 + next_u01() * 5.0;
        let pt = [x, y, z];
        pts_3d.push(pt);
        let p1 = project_point(k, &r_id, [0.0, 0.0, 0.0], pt);
        let p2 = project_point(k, r, t, pt);
        matches.push(FeatureMatch {
            query_index: i,
            target_index: i,
            query_point: (p1.0 as f32, p1.1 as f32),
            target_point: (p2.0 as f32, p2.1 as f32),
            distance: 0.1,
            ratio: 0.45,
            displacement: ((p2.0 - p1.0) as f32, (p2.1 - p1.1) as f32),
        });
    }

    for k_idx in 0..n_outliers {
        let i = n_inliers + k_idx;
        let x1 = next_u01() * 800.0;
        let y1 = next_u01() * 600.0;
        let x2 = next_u01() * 800.0;
        let y2 = next_u01() * 600.0;
        matches.push(FeatureMatch {
            query_index: i,
            target_index: i,
            query_point: (x1 as f32, y1 as f32),
            target_point: (x2 as f32, y2 as f32),
            distance: 0.4,
            ratio: 0.78,
            displacement: ((x2 - x1) as f32, (y2 - y1) as f32),
        });
    }

    (pts_3d, matches)
}

#[test]
fn test_essential_and_pose_recovery_under_1_deg_error() {
    let k = CameraIntrinsics {
        fx: 720.0,
        fy: 720.0,
        cx: 400.0,
        cy: 300.0,
    };
    let r_true = euler_zyx_rotation(12.0, -6.0, 4.0);
    let t_true: [f64; 3] = [0.6, -0.15, 0.2];
    let t_true_norm = {
        let n = (t_true[0] * t_true[0] + t_true[1] * t_true[1] + t_true[2] * t_true[2]).sqrt();
        [t_true[0] / n, t_true[1] / n, t_true[2] / n]
    };

    let (_, matches) = make_3d_correspondences(80, 26, &k, &r_true, t_true);

    // Direct normalized 8-point fit on clean subset works cleanly
    let f8 = fit_fundamental_8point(&matches[..12]).expect("8-point fit should succeed");
    let mean_sampson_clean: f64 = matches[..80]
        .iter()
        .map(|m| {
            sampson_distance_sq(
                &f8,
                (m.query_point.0 as f64, m.query_point.1 as f64),
                (m.target_point.0 as f64, m.target_point.1 as f64),
            )
            .sqrt()
        })
        .sum::<f64>()
        / 80.0;
    assert!(
        mean_sampson_clean < 0.05,
        "Mean Sampson error too high: {mean_sampson_clean}"
    );

    // Essential matrix RANSAC with 25% outliers
    let (inliers, e_opt) =
        estimate_essential_ransac(&matches, &k, &k, FundamentalRansacOptions::default());
    let e = e_opt.expect("Essential matrix should be estimated");
    assert!(
        inliers.len() >= 76,
        "Expected >= 76 inliers, got {}",
        inliers.len()
    );

    let pose = recover_pose(&e, &inliers, &k, &k).expect("Pose recovery should succeed");
    assert!(
        pose.cheirality_inliers >= 75,
        "Expected >= 75 cheirality-valid points, got {}",
        pose.cheirality_inliers
    );

    let rot_err = rotation_error_deg(&pose.rotation, &r_true);
    assert!(
        rot_err < 1.0,
        "Expected rotation error < 1.0 deg, got {rot_err:.4} deg"
    );

    let cos_t = (pose.translation[0] * t_true_norm[0]
        + pose.translation[1] * t_true_norm[1]
        + pose.translation[2] * t_true_norm[2])
        .clamp(-1.0, 1.0);
    let t_err_deg = cos_t.acos().to_degrees();
    assert!(
        t_err_deg < 2.0,
        "Expected translation direction error < 2.0 deg, got {t_err_deg:.4} deg"
    );

    // Verify linear triangulation recovers a point at positive depth proportional to X_0
    let p1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let p2 = [
        [
            pose.rotation[0][0],
            pose.rotation[0][1],
            pose.rotation[0][2],
            pose.translation[0],
        ],
        [
            pose.rotation[1][0],
            pose.rotation[1][1],
            pose.rotation[1][2],
            pose.translation[1],
        ],
        [
            pose.rotation[2][0],
            pose.rotation[2][1],
            pose.rotation[2][2],
            pose.translation[2],
        ],
    ];
    let xn1 = k.normalize_point(
        matches[0].query_point.0 as f64,
        matches[0].query_point.1 as f64,
    );
    let xn2 = k.normalize_point(
        matches[0].target_point.0 as f64,
        matches[0].target_point.1 as f64,
    );
    let tri = triangulate_point_linear(&p1, &p2, xn1, xn2).expect("Triangulation should succeed");
    assert!(
        tri[2] > 0.0,
        "Triangulated depth should be positive: {tri:?}"
    );
}

#[test]
fn test_7point_fundamental_exact() {
    let k = CameraIntrinsics {
        fx: 720.0,
        fy: 720.0,
        cx: 400.0,
        cy: 300.0,
    };
    let r_true = euler_zyx_rotation(8.0, -4.0, 3.0);
    let t_true = [0.5, 0.1, -0.15];
    let (_, matches) = make_3d_correspondences(12, 0, &k, &r_true, t_true);

    let sample7: [FeatureMatch; 7] = matches[..7].try_into().unwrap();
    let candidates = fit_fundamental_7point(&sample7);
    assert!(
        !candidates.is_empty(),
        "Expected at least 1 real 7-point candidate"
    );

    // At least one root must fit both the 7 sample points and the 5 held-out correspondences
    let best_heldout_err = candidates
        .iter()
        .map(|f| {
            matches[7..]
                .iter()
                .map(|m| {
                    sampson_distance_sq(
                        f,
                        (m.query_point.0 as f64, m.query_point.1 as f64),
                        (m.target_point.0 as f64, m.target_point.1 as f64),
                    )
                    .sqrt()
                })
                .sum::<f64>()
                / 5.0
        })
        .fold(f64::INFINITY, f64::min);

    assert!(
        best_heldout_err < 0.05,
        "7-point solver failed to generalize to held-out points: err={best_heldout_err}"
    );
}

#[test]
fn test_gric_selects_homography_for_planar_and_fundamental_for_3d() {
    let k = CameraIntrinsics {
        fx: 700.0,
        fy: 700.0,
        cx: 350.0,
        cy: 280.0,
    };
    let r_id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let r_tilt = euler_zyx_rotation(5.0, 14.0, -3.0);
    let t_plane = [0.35, -0.1, 0.05];

    // Case 1: Planar scene (Z = 6.0 for all points) -> GRIC must pick Homography
    let mut planar_matches = Vec::new();
    for i in 0..40 {
        let x = -2.0 + ((i % 8) as f64) * 0.55;
        let y = -1.5 + ((i / 8) as f64) * 0.7;
        let pt = [x, y, 6.0];
        let p1 = project_point(&k, &r_id, [0.0, 0.0, 0.0], pt);
        let p2 = project_point(&k, &r_tilt, t_plane, pt);
        planar_matches.push(FeatureMatch {
            query_index: i,
            target_index: i,
            query_point: (p1.0 as f32, p1.1 as f32),
            target_point: (p2.0 as f32, p2.1 as f32),
            distance: 0.1,
            ratio: 0.4,
            displacement: ((p2.0 - p1.0) as f32, (p2.1 - p1.1) as f32),
        });
    }
    let (_, h_planar) =
        estimate_homography_ransac(&planar_matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
    let (_, f_planar) = estimate_fundamental_ransac(&planar_matches, 1.5);
    let gric_planar = select_model_gric(&planar_matches, h_planar.as_ref(), f_planar.as_ref(), 1.0)
        .expect("GRIC comparison should succeed for planar scene");
    assert_eq!(
        gric_planar.selected,
        ModelSelection::Homography,
        "Expected Homography on planar scene, got {gric_planar:?}"
    );

    // Case 2: Strong 3D parallax scene (widely varying depths) -> GRIC must pick Fundamental
    let (_, nonplanar_matches) = make_3d_correspondences(
        60,
        0,
        &k,
        &euler_zyx_rotation(10.0, -8.0, 2.0),
        [0.9, 0.25, 0.35],
    );
    let (_, h_3d) = estimate_homography_ransac(&nonplanar_matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
    let (_, f_3d) = estimate_fundamental_ransac(&nonplanar_matches, 1.5);
    let gric_3d = select_model_gric(&nonplanar_matches, h_3d.as_ref(), f_3d.as_ref(), 1.0)
        .expect("GRIC comparison should succeed for 3D scene");
    assert_eq!(
        gric_3d.selected,
        ModelSelection::Fundamental,
        "Expected Fundamental on 3D parallax scene, got {gric_3d:?}"
    );
}
