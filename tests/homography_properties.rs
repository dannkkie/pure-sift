use proptest::{collection::vec, prelude::*, sample::select};
use pure_sift::pipeline::{
    DEFAULT_RANSAC_REPROJ_THRESHOLD, FeatureMatch, Homography2D, estimate_homography_ransac,
};

type Point = (f32, f32);

fn correspondences(pairs: impl IntoIterator<Item = (Point, Point)>) -> Vec<FeatureMatch> {
    pairs
        .into_iter()
        .enumerate()
        .map(|(index, (query, target))| FeatureMatch {
            query_index: index,
            target_index: index,
            query_point: query,
            target_point: target,
            distance: 0.0,
            ratio: 0.5,
            displacement: (target.0 - query.0, target.1 - query.1),
        })
        .collect()
}

fn point() -> impl Strategy<Value = Point> {
    (-100_000.0f32..=100_000.0, -100_000.0f32..=100_000.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn ransac_accepts_arbitrary_correspondences(pairs in vec((point(), point()), 4..=20)) {
        let matches = correspondences(pairs);
        let (inliers, _) = estimate_homography_ransac(&matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
        prop_assert!(inliers.len() <= matches.len());
    }

    #[test]
    fn ransac_handles_collinear_and_repeated_points(query_y in -1_000.0f32..=1_000.0,
                                                   target_x in -1_000.0f32..=1_000.0,
                                                   query in point(), target in point()) {
        let collinear = correspondences([0.0, 1.0, 2.0, 3.0].map(|i| {
            ((i, query_y), (target_x, i))
        }));
        let repeated = correspondences([(query, target); 4]);

        for matches in [collinear, repeated] {
            // Similarity fallback can still return inliers for a degenerate DLT sample.
            let (inliers, _) = estimate_homography_ransac(&matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
            prop_assert!(inliers.len() <= matches.len());
        }
    }

    #[test]
    fn ransac_handles_extreme_finite_coordinates(
        pairs in vec((
            select(vec![f32::MIN, -1.0e30, 1.0e30, f32::MAX]),
            select(vec![f32::MIN, -1.0e30, 1.0e30, f32::MAX]),
            point(),
        ), 4..=12),
    ) {
        let matches = correspondences(pairs.into_iter().map(|(x, y, target)| ((x, y), target)));
        let (inliers, _) = estimate_homography_ransac(&matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
        prop_assert!(inliers.len() <= matches.len());
    }
}

#[test]
fn ransac_recovers_homography_at_80_percent_outliers_and_with_collinear_distractors() {
    let true_h = Homography2D {
        matrix: [
            [1.12, -0.06, 42.0],
            [0.05, 0.94, -24.0],
            [0.00015, -0.0001, 1.0],
        ],
    };

    let inlier_pts = [
        (40.0f32, 50.0f32),
        (200.0, 40.0),
        (420.0, 75.0),
        (560.0, 110.0),
        (80.0, 210.0),
        (270.0, 240.0),
        (490.0, 280.0),
        (60.0, 410.0),
        (230.0, 450.0),
        (440.0, 430.0),
        (580.0, 470.0),
        (330.0, 340.0),
    ];

    let mut matches = Vec::new();
    for (i, &(qx, qy)) in inlier_pts.iter().enumerate() {
        let (tx, ty) = true_h.project(qx, qy).unwrap();
        matches.push(FeatureMatch {
            query_index: i,
            target_index: i,
            query_point: (qx, qy),
            target_point: (tx, ty),
            distance: 0.02,
            ratio: 0.25,
            displacement: (tx - qx, ty - qy),
        });
    }

    // Add 48 gross outliers (80% outlier ratio out of 60 total matches), including collinear points
    for k in 0..48usize {
        let idx = inlier_pts.len() + k;
        let qx = if k < 8 {
            100.0 + (k as f32) * 30.0
        } else {
            ((k * 97) % 600) as f32
        };
        let qy = if k < 8 {
            100.0
        } else {
            ((k * 131) % 480) as f32
        };
        let tx = ((k * 173 + 50) % 620) as f32;
        let ty = ((k * 211 + 80) % 490) as f32;
        matches.push(FeatureMatch {
            query_index: idx,
            target_index: idx,
            query_point: (qx, qy),
            target_point: (tx, ty),
            distance: 0.35,
            ratio: 0.72,
            displacement: (tx - qx, ty - qy),
        });
    }

    let (inliers, est_opt) = estimate_homography_ransac(&matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
    assert_eq!(
        inliers.len(),
        inlier_pts.len(),
        "LO-RANSAC must recover all {} true inliers under 80% outliers",
        inlier_pts.len()
    );
    let est_h = est_opt.expect("Expected valid homography");
    for &(qx, qy) in &inlier_pts {
        let (tx, ty) = true_h.project(qx, qy).unwrap();
        let (px, py) = est_h.project(qx, qy).unwrap();
        let err = (tx - px).hypot(ty - py);
        assert!(err < 0.05, "Reprojection error {err} >= 0.05 px");
    }
}
