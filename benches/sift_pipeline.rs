//! Criterion micro-benchmarks for the core SIFT pipeline stages (#1):
//! feature extraction on a synthetic image, descriptor matching between two
//! synthetic feature sets, and RANSAC homography estimation on synthetic
//! correspondences. All inputs are deterministic so runs are comparable.
//!
//! Run with `cargo bench --bench sift_pipeline`.

use criterion::{Criterion, criterion_group, criterion_main};
use image::{GrayImage, Luma};
use pure_sift::{
    DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD, FeatureMatch,
    SIFT_DESCRIPTOR_LEN, SiftFeature, estimate_homography_ransac, extract_sift_features,
    luma_to_array2, match_descriptors,
};

/// Deterministic xorshift64* generator so every bench run sees identical data.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as u32
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Synthetic 512x512 luma8 image: two overlaid sinusoid textures so the DoG
/// detector finds a stable, non-trivial set of extrema on every run.
fn synthetic_luma8(size: u32) -> GrayImage {
    GrayImage::from_fn(size, size, |x, y| {
        let fx = x as f32 / size as f32;
        let fy = y as f32 / size as f32;
        let wave_a =
            (fx * std::f32::consts::TAU * 8.0).sin() * (fy * std::f32::consts::TAU * 8.0).cos();
        let wave_b = (fx * std::f32::consts::TAU * 17.0 + 1.3).sin()
            * (fy * std::f32::consts::TAU * 11.0 + 0.7).sin();
        let value = (0.5 + 0.22 * wave_a + 0.13 * wave_b).clamp(0.0, 1.0);
        Luma([(value * 255.0) as u8])
    })
}

/// Deterministic synthetic SIFT features with 128-D descriptors.
fn synthetic_features(count: usize, seed: u64) -> Vec<SiftFeature> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| SiftFeature {
            x: rng.next_f32() * 512.0,
            y: rng.next_f32() * 512.0,
            scale: 1.0 + rng.next_f32(),
            response: 0.05 + rng.next_f32() * 0.5,
            orientation: rng.next_f32() * std::f32::consts::TAU,
            descriptor: (0..SIFT_DESCRIPTOR_LEN).map(|_| rng.next_f32()).collect(),
        })
        .collect()
}

/// Deterministic synthetic correspondences: 80% inliers around a known
/// projective transform with small noise, 20% uniform outliers.
fn synthetic_correspondences(count: usize) -> Vec<FeatureMatch> {
    let mut rng = Rng(7);
    (0..count)
        .map(|i| {
            let qx = rng.next_f32() * 512.0;
            let qy = rng.next_f32() * 512.0;
            let (tx, ty) = if i % 5 == 0 {
                (rng.next_f32() * 512.0, rng.next_f32() * 512.0)
            } else {
                let w = 1.0 + 0.0004 * qx;
                (
                    (1.04 * qx - 0.08 * qy + 12.0) / w + (rng.next_f32() - 0.5) * 1.5,
                    (0.08 * qx + 1.04 * qy - 7.0) / w + (rng.next_f32() - 0.5) * 1.5,
                )
            };
            FeatureMatch {
                query_index: i,
                target_index: i,
                query_point: (qx, qy),
                target_point: (tx, ty),
                distance: 0.1,
                ratio: 0.3 + (i % 7) as f32 * 0.05,
                displacement: (tx - qx, ty - qy),
            }
        })
        .collect()
}

/// End-to-end SIFT extraction on a synthetic 512x512 luma8 image.
fn bench_extract_sift_features(c: &mut Criterion) {
    let img = luma_to_array2(&synthetic_luma8(512));
    c.bench_function("extract_sift_features/512x512_luma8", |b| {
        b.iter(|| {
            extract_sift_features(std::hint::black_box(&img), 4, 3, DEFAULT_CONTRAST_THRESHOLD)
        })
    });
}

/// Ratio-test matching between two synthetic 500-feature sets.
fn bench_match_descriptors(c: &mut Criterion) {
    let query = synthetic_features(500, 42);
    let target = synthetic_features(500, 43);
    c.bench_function("match_descriptors/500x500_features", |b| {
        b.iter(|| {
            match_descriptors(
                std::hint::black_box(&query),
                std::hint::black_box(&target),
                DEFAULT_MATCH_RATIO,
            )
        })
    });
}

/// RANSAC homography estimation on a synthetic 200-correspondence set.
fn bench_estimate_homography_ransac(c: &mut Criterion) {
    let matches = synthetic_correspondences(200);
    c.bench_function("estimate_homography_ransac/200_correspondences", |b| {
        b.iter(|| {
            estimate_homography_ransac(
                std::hint::black_box(&matches),
                DEFAULT_RANSAC_REPROJ_THRESHOLD,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_extract_sift_features,
    bench_match_descriptors,
    bench_estimate_homography_ransac
);
criterion_main!(benches);
