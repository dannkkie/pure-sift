use proptest::{collection::vec, prelude::*, sample::select};
use pure_sift::pipeline::{
    DEFAULT_RANSAC_REPROJ_THRESHOLD, FeatureMatch, estimate_homography_ransac,
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
