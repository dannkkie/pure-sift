#![forbid(unsafe_code)]

use image::GrayImage;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Base standard deviation (\sigma_0) for the initial Gaussian scale-space level.
pub const BASE_SIGMA: f32 = 1.0;

/// Default contrast threshold for 3D DoG extrema extraction on normalized [0.0, 1.0] images.
pub const DEFAULT_CONTRAST_THRESHOLD: f32 = 0.015;

/// Represents a detected scale-invariant feature point (keypoint) in 2D image coordinates
/// and scale space, following US6711293B1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    /// Horizontal coordinate in the base image domain.
    pub x: f32,
    /// Vertical coordinate in the base image domain.
    pub y: f32,
    /// Effective Gaussian scale (\sigma) at which the extremum was localized.
    pub scale: f32,
    /// Difference-of-Gaussians (DoG) response amplitude at the extremum.
    pub response: f32,
}

/// Generates a normalized 1D Gaussian convolution kernel for a given standard deviation `sigma`.
///
/// Uses the analytical 1D Gaussian distribution formula:
/// `g(x) = (1 / (sqrt(2 * pi) * sigma)) * exp(-(x^2) / (2 * sigma^2))`
/// normalized so that all discrete weights sum to `1.0`.
pub fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let clamped_sigma = sigma.max(1e-3);
    // Truncate kernel at 2.5 * sigma (clamped to [1, 6] for real-time sub-50ms throughput)
    let radius = ((2.5 * clamped_sigma).ceil() as isize).clamp(1, 6);
    let two_sigma_sq = 2.0 * clamped_sigma * clamped_sigma;
    let norm_factor = 1.0 / ((2.0 * std::f32::consts::PI).sqrt() * clamped_sigma);

    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|x| {
            let x_f = x as f32;
            norm_factor * (-((x_f * x_f) / two_sigma_sq)).exp()
        })
        .collect();

    let sum: f32 = kernel.iter().copied().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for w in &mut kernel {
            *w *= inv_sum;
        }
    }
    kernel
}

/// Generates a normalized 2D isotropic Gaussian kernel as an `Array2<f32>`
/// via the outer product of the 1D Gaussian kernel.
pub fn gaussian_kernel_2d(sigma: f32) -> Array2<f32> {
    let k1d = gaussian_kernel_1d(sigma);
    let size = k1d.len();
    let mut k2d = Array2::<f32>::zeros((size, size));
    for (y, row_w) in k1d.iter().copied().enumerate() {
        for (x, col_w) in k1d.iter().copied().enumerate() {
            k2d[[y, x]] = row_w * col_w;
        }
    }
    k2d
}

/// Applies a separable 2D Gaussian blur to an `Array2<f32>` image using safe slice iterators
/// structured for automatic LLVM SIMD vectorization.
pub fn gaussian_blur(img: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let (height, width) = img.dim();
    if height == 0 || width == 0 {
        return Array2::zeros((height, width));
    }

    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    // Ensure standard row-major contiguous layout for slice-based SIMD iteration
    let src_cow = img.as_standard_layout();
    let src_slice = src_cow
        .as_slice()
        .expect("standard layout guarantees contiguous slice");

    // Pass 1: Horizontal 1D convolution (row-wise contiguous scan)
    let mut temp = vec![0.0f32; height * width];
    for (y, out_row) in temp.chunks_exact_mut(width).enumerate() {
        let in_row = &src_slice[y * width..(y + 1) * width];

        if width > 2 * radius {
            // Left border (clamped boundary)
            for (x, out_val) in out_row[..radius].iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k_idx, &weight) in kernel.iter().enumerate() {
                    let sample_x = (x + k_idx).saturating_sub(radius).min(width - 1);
                    acc += in_row[sample_x] * weight;
                }
                *out_val = acc;
            }

            // Interior columns: sliding window amenable to compiler SIMD vectorization
            let k_len = kernel.len();
            for (out_val, window) in out_row[radius..width - radius]
                .iter_mut()
                .zip(in_row.windows(k_len))
            {
                let mut acc = 0.0f32;
                for (&pix, &weight) in window.iter().zip(kernel.iter()) {
                    acc += pix * weight;
                }
                *out_val = acc;
            }

            // Right border (clamped boundary)
            for (offset, out_val) in out_row[width - radius..].iter_mut().enumerate() {
                let x = width - radius + offset;
                let mut acc = 0.0f32;
                for (k_idx, &weight) in kernel.iter().enumerate() {
                    let sample_x = (x + k_idx).saturating_sub(radius).min(width - 1);
                    acc += in_row[sample_x] * weight;
                }
                *out_val = acc;
            }
        } else {
            for (x, out_val) in out_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k_idx, &weight) in kernel.iter().enumerate() {
                    let sample_x = (x + k_idx).saturating_sub(radius).min(width - 1);
                    acc += in_row[sample_x] * weight;
                }
                *out_val = acc;
            }
        }
    }

    // Pass 2: Vertical 1D convolution implemented as row-wise AXPY accumulations.
    // Iterating across contiguous rows `out_row[x] += weight * temp_row[x]` guarantees
    // unit-stride memory access and direct SIMD auto-vectorization.
    let mut dst = vec![0.0f32; height * width];
    for (y, out_row) in dst.chunks_exact_mut(width).enumerate() {
        for (k_idx, &weight) in kernel.iter().enumerate() {
            let sample_y = (y + k_idx).saturating_sub(radius).min(height - 1);
            let in_row = &temp[sample_y * width..(sample_y + 1) * width];
            for (out_val, &in_val) in out_row.iter_mut().zip(in_row.iter()) {
                *out_val += in_val * weight;
            }
        }
    }

    Array2::from_shape_vec((height, width), dst).expect("output shape matches height * width")
}

/// Downsamples an `Array2<f32>` by a factor of 2 along both spatial axes,
/// selecting every second pixel `(2*y, 2*x)` as specified in US6711293B1.
pub fn downsample_2x(img: &Array2<f32>) -> Array2<f32> {
    let (height, width) = img.dim();
    let new_h = height / 2;
    let new_w = width / 2;
    if new_h == 0 || new_w == 0 {
        return Array2::zeros((0, 0));
    }

    let src_cow = img.as_standard_layout();
    let src_slice = src_cow
        .as_slice()
        .expect("standard layout guarantees contiguous slice");

    let mut out = Vec::with_capacity(new_h * new_w);
    for y in 0..new_h {
        let row = &src_slice[(2 * y) * width..(2 * y + 1) * width];
        out.extend(row.iter().step_by(2).take(new_w).copied());
    }

    Array2::from_shape_vec((new_h, new_w), out).expect("downsampled dimensions match buffer length")
}

/// Generates the pyramidal Difference-of-Gaussians (DoG) scale-space representation
/// across `octaves` octaves and `scales` scale intervals per octave.
///
/// Within each octave:
/// - `scales + 1` Gaussian-smoothed images are computed with exponentially spaced scales
///   `\sigma_i = \sigma_0 * k^i`, where `k = 2^(1 / scales)`.
/// - By the Gaussian semigroup property, each consecutive level blurs the previous Gaussian image
///   with incremental standard deviation `\Delta\sigma_i = \sqrt(\sigma_i^2 - \sigma_{i-1}^2)`.
/// - Adjacent Gaussian levels are subtracted element-wise (`L_{i+1} - L_i`) to produce `scales`
///   DoG slices per octave.
/// - Between octaves, the smoothed image is decimated by 2x along each spatial dimension.
pub fn generate_dog_space(img: &Array2<f32>, octaves: usize, scales: usize) -> Vec<Array2<f32>> {
    if img.is_empty() || octaves == 0 || scales == 0 {
        return Vec::new();
    }

    let k = 2.0f32.powf(1.0 / (scales as f32));
    let mut dog_space = Vec::with_capacity(octaves * scales);
    let mut octave_base = gaussian_blur(img, BASE_SIGMA);

    for octave in 0..octaves {
        let (h, w) = octave_base.dim();
        if h < 3 || w < 3 {
            break;
        }

        let mut prev_gaussian = octave_base;
        let mut prev_sigma = BASE_SIGMA;

        for s in 1..=scales {
            let target_sigma = BASE_SIGMA * k.powi(s as i32);
            let delta_sigma = (target_sigma * target_sigma - prev_sigma * prev_sigma)
                .max(1e-4)
                .sqrt();

            let next_gaussian = gaussian_blur(&prev_gaussian, delta_sigma);

            // Element-wise subtraction: D(x, y, sigma) = L(x, y, k * sigma) - L(x, y, sigma)
            let prev_slice = prev_gaussian
                .as_slice()
                .expect("Gaussian level is contiguous");
            let next_slice = next_gaussian
                .as_slice()
                .expect("Gaussian level is contiguous");

            let diff_vec: Vec<f32> = next_slice
                .iter()
                .zip(prev_slice.iter())
                .map(|(&n, &p)| n - p)
                .collect();

            let dog_img = Array2::from_shape_vec((h, w), diff_vec)
                .expect("DoG slice shape matches octave dimensions");
            dog_space.push(dog_img);

            prev_gaussian = next_gaussian;
            prev_sigma = target_sigma;
        }

        // Prepare next octave base by 2x spatial downsampling of the doubled-sigma Gaussian image
        if octave + 1 < octaves {
            octave_base = downsample_2x(&prev_gaussian);
        } else {
            octave_base = prev_gaussian;
        }
    }

    dog_space
}

/// Evaluates the 2D Hessian principal curvature ratio test at `(x, y)` in `curr_slice`
/// to reject poorly localized responses along straight 1D step edges.
#[inline]
fn passes_hessian_edge_test(
    curr_slice: &[f32],
    width: usize,
    x: usize,
    y: usize,
    center: f32,
) -> bool {
    let row_prev = &curr_slice[(y - 1) * width..y * width];
    let row_curr = &curr_slice[y * width..(y + 1) * width];
    let row_next = &curr_slice[(y + 1) * width..(y + 2) * width];

    let dxx = row_curr[x + 1] - 2.0 * center + row_curr[x - 1];
    let dyy = row_next[x] - 2.0 * center + row_prev[x];
    let dxy = 0.25 * ((row_next[x + 1] - row_next[x - 1]) - (row_prev[x + 1] - row_prev[x - 1]));

    let trace = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;

    // Principal curvature ratio r = 10.0 => (r + 1)^2 / r = 12.1
    const EDGE_RATIO: f32 = 10.0;
    const THRESHOLD_RATIO: f32 = ((EDGE_RATIO + 1.0) * (EDGE_RATIO + 1.0)) / EDGE_RATIO;

    det > 1e-9 && (trace * trace) < THRESHOLD_RATIO * det
}

/// Refines a discrete 3D Difference-of-Gaussians (DoG) extremum at `(x, y)` in `curr_slice`
/// using a second-order 3D Taylor expansion across `(x, y, s)` (US6711293B1 / Lowe 2004):
///
/// `D(x + dx) = D(x) + g^T * dx + 0.5 * dx^T * H_3x3 * dx`
/// `dx_hat = -H_3x3^-1 * g`, `D(dx_hat) = D(x) + 0.5 * g^T * dx_hat`
///
/// Returns `(dx, dy, ds, interpolated_response)` with offsets bounded to `[-0.5, 0.5]`.
#[inline]
pub fn refine_extremum_3d(
    prev_slice: &[f32],
    curr_slice: &[f32],
    next_slice: &[f32],
    width: usize,
    x: usize,
    y: usize,
) -> (f32, f32, f32, f32) {
    let c_prev = &curr_slice[(y - 1) * width..y * width];
    let c_curr = &curr_slice[y * width..(y + 1) * width];
    let c_next = &curr_slice[(y + 1) * width..(y + 2) * width];

    let p_prev = &prev_slice[(y - 1) * width..y * width];
    let p_curr = &prev_slice[y * width..(y + 1) * width];
    let p_next = &prev_slice[(y + 1) * width..(y + 2) * width];

    let n_prev = &next_slice[(y - 1) * width..y * width];
    let n_curr = &next_slice[y * width..(y + 1) * width];
    let n_next = &next_slice[(y + 1) * width..(y + 2) * width];

    let center = c_curr[x];

    // 3D first-order central finite-difference gradient g = [dx, dy, ds]^T
    let gx = 0.5 * (c_curr[x + 1] - c_curr[x - 1]);
    let gy = 0.5 * (c_next[x] - c_prev[x]);
    let gs = 0.5 * (n_curr[x] - p_curr[x]);

    // 3x3 symmetric scale-space Hessian entries
    let dxx = c_curr[x + 1] - 2.0 * center + c_curr[x - 1];
    let dyy = c_next[x] - 2.0 * center + c_prev[x];
    let dss = n_curr[x] - 2.0 * center + p_curr[x];

    let dxy = 0.25 * ((c_next[x + 1] - c_next[x - 1]) - (c_prev[x + 1] - c_prev[x - 1]));
    let dxs = 0.25 * ((n_curr[x + 1] - n_curr[x - 1]) - (p_curr[x + 1] - p_curr[x - 1]));
    let dys = 0.25 * ((n_next[x] - n_prev[x]) - (p_next[x] - p_prev[x]));

    // Symmetric 3x3 cofactor matrix C = adj(H_3x3)
    let c00 = dyy * dss - dys * dys;
    let c01 = dys * dxs - dxy * dss;
    let c02 = dxy * dys - dyy * dxs;
    let c11 = dxx * dss - dxs * dxs;
    let c12 = dxy * dxs - dxx * dys;
    let c22 = dxx * dyy - dxy * dxy;

    let det3 = dxx * c00 + dxy * c01 + dxs * c02;

    if det3.abs() > 1e-9 {
        let inv_det = -1.0 / det3;
        let ox = (c00 * gx + c01 * gy + c02 * gs) * inv_det;
        let oy = (c01 * gx + c11 * gy + c12 * gs) * inv_det;
        let os = (c02 * gx + c12 * gy + c22 * gs) * inv_det;

        if ox.abs() <= 0.5 && oy.abs() <= 0.5 && os.abs() <= 0.5 {
            let interp = center + 0.5 * (gx * ox + gy * oy + gs * os);
            return (ox, oy, os, interp);
        }
    }

    // Fallback to 2D spatial quadratic refinement if the scale axis is nearly flat/degenerate
    if c22.abs() > 1e-9 {
        let inv_det2 = -1.0 / c22;
        let ox = ((dyy * gx - dxy * gy) * inv_det2).clamp(-0.5, 0.5);
        let oy = ((-dxy * gx + dxx * gy) * inv_det2).clamp(-0.5, 0.5);
        let os = if dss.abs() > 1e-6 {
            (-gs / dss).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let interp = center + 0.5 * (gx * ox + gy * oy + gs * os);
        return (ox, oy, os, interp);
    }

    (0.0, 0.0, 0.0, center)
}

/// Checks whether `val` at `(x, y)` is strictly greater than all 9 values in the 3x3 patch
/// centered at `(x, y)` within `plane`.
#[inline]
fn is_greater_than_patch3x3(plane: &[f32], width: usize, x: usize, y: usize, val: f32) -> bool {
    let r0 = &plane[(y - 1) * width..y * width];
    let r1 = &plane[y * width..(y + 1) * width];
    let r2 = &plane[(y + 1) * width..(y + 2) * width];

    val > r0[x - 1]
        && val > r0[x]
        && val > r0[x + 1]
        && val > r1[x - 1]
        && val > r1[x]
        && val > r1[x + 1]
        && val > r2[x - 1]
        && val > r2[x]
        && val > r2[x + 1]
}

/// Checks whether `val` at `(x, y)` is strictly less than all 9 values in the 3x3 patch
/// centered at `(x, y)` within `plane`.
#[inline]
fn is_less_than_patch3x3(plane: &[f32], width: usize, x: usize, y: usize, val: f32) -> bool {
    let r0 = &plane[(y - 1) * width..y * width];
    let r1 = &plane[y * width..(y + 1) * width];
    let r2 = &plane[(y + 1) * width..(y + 2) * width];

    val < r0[x - 1]
        && val < r0[x]
        && val < r0[x + 1]
        && val < r1[x - 1]
        && val < r1[x]
        && val < r1[x + 1]
        && val < r2[x - 1]
        && val < r2[x]
        && val < r2[x + 1]
}

/// Scans consecutive Difference-of-Gaussians (DoG) scale slices to extract 3D local
/// minima and maxima across the 26-connected `(3 x 3 x 3)` scale-space neighborhood.
///
/// Consecutive triplets `(prev, curr, next)` sharing identical spatial dimensions belong
/// to the same octave and are evaluated for strict local extrema exceeding `threshold`.
pub fn extract_extrema(dog_space: &[Array2<f32>], threshold: f32) -> Vec<Keypoint> {
    if dog_space.len() < 3 {
        return Vec::new();
    }

    let base_dim = dog_space[0].dim();
    if base_dim.0 < 3 || base_dim.1 < 3 {
        return Vec::new();
    }

    // Count how many slices per octave exist in the first octave to compute k = 2^(1/scales)
    let scales_per_octave = dog_space
        .iter()
        .take_while(|slice| slice.dim() == base_dim)
        .count()
        .max(1);
    let k = 2.0f32.powf(1.0 / (scales_per_octave as f32));

    let mut keypoints = Vec::new();
    let mut octave_idx = 0usize;
    let mut scale_idx_in_octave = 1usize;

    for i in 1..(dog_space.len() - 1) {
        let prev_dim = dog_space[i - 1].dim();
        let curr_dim = dog_space[i].dim();
        let next_dim = dog_space[i + 1].dim();

        // Detect octave transition when spatial dimensions change
        if curr_dim != prev_dim {
            octave_idx += 1;
            scale_idx_in_octave = 1;
            continue;
        }
        if curr_dim != next_dim {
            continue;
        }

        let (height, width) = curr_dim;
        if height < 3 || width < 3 {
            continue;
        }

        let prev_cow = dog_space[i - 1].as_standard_layout();
        let curr_cow = dog_space[i].as_standard_layout();
        let next_cow = dog_space[i + 1].as_standard_layout();

        let prev_slice = prev_cow.as_slice().expect("standard layout slice");
        let curr_slice = curr_cow.as_slice().expect("standard layout slice");
        let next_slice = next_cow.as_slice().expect("standard layout slice");

        let octave_multiplier = (1usize << octave_idx) as f32;
        let effective_scale = BASE_SIGMA * octave_multiplier * k.powi(scale_idx_in_octave as i32);

        for y in 1..(height - 1) {
            let r_above = &curr_slice[(y - 1) * width..y * width];
            let r_curr = &curr_slice[y * width..(y + 1) * width];
            let r_below = &curr_slice[(y + 1) * width..(y + 2) * width];

            for x in 1..(width - 1) {
                let val = r_curr[x];
                if val.abs() <= threshold {
                    continue;
                }

                let is_extremum = if val > 0.0 {
                    // Check 8 spatial neighbors in current scale plane first (fast short-circuit)
                    val > r_curr[x - 1]
                        && val > r_curr[x + 1]
                        && val > r_above[x - 1]
                        && val > r_above[x]
                        && val > r_above[x + 1]
                        && val > r_below[x - 1]
                        && val > r_below[x]
                        && val > r_below[x + 1]
                        && is_greater_than_patch3x3(prev_slice, width, x, y, val)
                        && is_greater_than_patch3x3(next_slice, width, x, y, val)
                } else {
                    val < r_curr[x - 1]
                        && val < r_curr[x + 1]
                        && val < r_above[x - 1]
                        && val < r_above[x]
                        && val < r_above[x + 1]
                        && val < r_below[x - 1]
                        && val < r_below[x]
                        && val < r_below[x + 1]
                        && is_less_than_patch3x3(prev_slice, width, x, y, val)
                        && is_less_than_patch3x3(next_slice, width, x, y, val)
                };

                if is_extremum && passes_hessian_edge_test(curr_slice, width, x, y, val) {
                    let (dx, dy, ds, interp_response) =
                        refine_extremum_3d(prev_slice, curr_slice, next_slice, width, x, y);
                    if interp_response.abs() > threshold {
                        let refined_scale = effective_scale * k.powf(ds);
                        keypoints.push(Keypoint {
                            x: ((x as f32) + dx) * octave_multiplier,
                            y: ((y as f32) + dy) * octave_multiplier,
                            scale: refined_scale.max(0.5 * BASE_SIGMA),
                            response: interp_response,
                        });
                    }
                }
            }
        }

        scale_idx_in_octave += 1;
    }

    keypoints
}

/// Converts a grayscale `image::GrayImage` into a normalized `ndarray::Array2<f32>`
/// with pixel intensities mapped to `[0.0, 1.0]`.
pub fn luma_to_array2(gray: &GrayImage) -> Array2<f32> {
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    let raw = gray.as_raw();
    let inv_255 = 1.0f32 / 255.0f32;

    let data: Vec<f32> = raw.iter().map(|&px| (px as f32) * inv_255).collect();
    Array2::from_shape_vec((height, width), data)
        .expect("GrayImage buffer length matches width * height")
}

/// Convenience end-to-end pipeline execution: generates the DoG scale-space pyramid
/// and extracts 3D local extrema keypoints.
pub fn detect_keypoints(
    img: &Array2<f32>,
    octaves: usize,
    scales: usize,
    threshold: f32,
) -> Vec<Keypoint> {
    let dog_space = generate_dog_space(img, octaves, scales);
    extract_extrema(&dog_space, threshold)
}

/// Dimensionality of the canonical SIFT descriptor (4 x 4 spatial subregions x 8 orientation bins).
pub const SIFT_DESCRIPTOR_LEN: usize = 128;

/// Default Lowe's nearest-neighbor distance ratio threshold (d1 / d2) for feature matching.
pub const DEFAULT_MATCH_RATIO: f32 = 0.75;

/// Default RANSAC reprojection error threshold in pixels for 2D similarity inlier filtering.
pub const DEFAULT_RANSAC_REPROJ_THRESHOLD: f32 = 3.0;

/// Maximum number of highest-contrast keypoints retained per image for O(N * M) descriptor matching
/// to bound worst-case latency on noisy high-resolution frames.
pub const MAX_FEATURES_PER_IMAGE: usize = 4096;

/// Complete SIFT feature combining a scale-space keypoint with its canonical orientation
/// and 128-dimensional rotation- and illumination-invariant descriptor (US6711293B1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiftFeature {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub response: f32,
    /// Dominant local gradient orientation in radians `[0.0, 2 * pi)`.
    pub orientation: f32,
    /// 128-element L2-normalized, saturation-clamped (<= 0.2), and re-normalized descriptor vector.
    pub descriptor: Vec<f32>,
}

/// Matched feature correspondence between a query image and a target image.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeatureMatch {
    pub query_index: usize,
    pub target_index: usize,
    pub query_point: (f32, f32),
    pub target_point: (f32, f32),
    /// Euclidean (L2) distance between the two 128-D descriptors.
    pub distance: f32,
    /// Ambiguity ratio `d1 / d2` between best and second-best match distances.
    pub ratio: f32,
    /// 2D spatial displacement `(target_x - query_x, target_y - query_y)`.
    pub displacement: (f32, f32),
}

/// 2D similarity transformation mapping query `(x, y)` to target `(a*x - b*y + tx, b*x + a*y + ty)`,
/// where `a = scale * cos(theta)` and `b = scale * sin(theta)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimilarityTransform2D {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub ty: f32,
    pub scale: f32,
    pub rotation_deg: f32,
}

/// 3x3 planar projective homography matrix `H` mapping query `(x, y, 1)^T` to target
/// `(x', y', w')^T = H * (x, y, 1)^T` under 3D perspective tilt.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Homography2D {
    pub matrix: [[f32; 3]; 3],
}

impl Homography2D {
    /// Projects a 2D point `(x, y)` through the 3x3 homography matrix.
    #[inline]
    pub fn project(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let h = &self.matrix;
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        if w.abs() < 1e-6 {
            return None;
        }
        let inv_w = 1.0 / w;
        let px = (h[0][0] * x + h[0][1] * y + h[0][2]) * inv_w;
        let py = (h[1][0] * x + h[1][1] * y + h[1][2]) * inv_w;
        Some((px, py))
    }
}

/// Aggregated kinematic / alignment estimate across matched SIFT features after RANSAC inlier filtering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackingSummary {
    pub raw_match_count: usize,
    pub match_count: usize,
    pub inlier_ratio: f32,
    pub mean_displacement: (f32, f32),
    pub median_displacement: (f32, f32),
    pub mean_scale_ratio: f32,
    pub mean_rotation_deg: f32,
}

/// Secondary orientation peak splitting ratio (`0.80 * dominant_peak`) per US6711293B1.
pub const SECONDARY_ORI_PEAK_RATIO: f32 = 0.80;

/// Quantizes a continuous keypoint scale `sigma` to the nearest 1/6-octave bucket
/// so sub-scale refined keypoints share cached Gaussian gradient planes.
#[inline]
fn scale_bucket_key(sigma: f32) -> (i32, f32) {
    let clamped = sigma.max(0.5);
    let key = (clamped.log2() * 6.0).round() as i32;
    let bucket_sigma = 2.0f32.powf((key as f32) / 6.0);
    (key, bucket_sigma)
}

/// Precomputed gradient magnitude and orientation planes for a single Gaussian scale level.
struct ScaleGradientPlanes {
    scale_key: i32,
    mag: Vec<f32>,
    angle: Vec<f32>,
}

impl ScaleGradientPlanes {
    fn compute(img: &Array2<f32>, scale_key: i32, sigma: f32) -> Self {
        let smoothed = gaussian_blur(img, sigma);
        let (height, width) = smoothed.dim();
        let slice = smoothed
            .as_slice()
            .expect("smoothed Gaussian image is contiguous");

        let mut mag = vec![0.0f32; height * width];
        let mut angle = vec![0.0f32; height * width];
        let two_pi = 2.0 * std::f32::consts::PI;

        if height >= 3 && width >= 3 {
            for y in 1..(height - 1) {
                let row_above = &slice[(y - 1) * width..y * width];
                let row_curr = &slice[y * width..(y + 1) * width];
                let row_below = &slice[(y + 1) * width..(y + 2) * width];
                let out_mag = &mut mag[y * width..(y + 1) * width];
                let out_ang = &mut angle[y * width..(y + 1) * width];

                for x in 1..(width - 1) {
                    let dx = row_curr[x + 1] - row_curr[x - 1];
                    let dy = row_below[x] - row_above[x];
                    out_mag[x] = (dx * dx + dy * dy).sqrt();
                    out_ang[x] = dy.atan2(dx).rem_euclid(two_pi);
                }
            }
        }

        Self {
            scale_key,
            mag,
            angle,
        }
    }
}

/// Computes all canonical orientations in `[0, 2*pi)` for a keypoint using a smoothed
/// 36-bin Gaussian-weighted gradient histogram, parabolic peak refinement, and
/// secondary orientation splitting for any local peak >= `80%` (`SECONDARY_ORI_PEAK_RATIO`)
/// of the dominant orientation peak (US6711293B1).
pub fn compute_keypoint_orientations(
    mag: &[f32],
    angle: &[f32],
    width: usize,
    height: usize,
    kp: &Keypoint,
) -> Vec<f32> {
    const NUM_BINS: usize = 36;
    const MAX_ORIENTATIONS_PER_KEYPOINT: usize = 4;
    let two_pi = 2.0 * std::f32::consts::PI;
    let sigma_ori = (1.5 * kp.scale).max(1.0);
    let two_sigma_sq = 2.0 * sigma_ori * sigma_ori;
    let radius = ((3.0 * sigma_ori).ceil() as isize).clamp(2, 18);

    let cx = kp.x.round() as isize;
    let cy = kp.y.round() as isize;

    let mut hist = [0.0f32; NUM_BINS];
    let bin_scale = (NUM_BINS as f32) / two_pi;

    let y_min = (cy - radius).max(1) as usize;
    let y_max = (cy + radius).min((height as isize) - 2).max(1) as usize;
    let x_min = (cx - radius).max(1) as usize;
    let x_max = (cx + radius).min((width as isize) - 2).max(1) as usize;

    let r_sq_max = (radius * radius) as f32;

    for y in y_min..=y_max {
        let dy = (y as isize - cy) as f32;
        let row_offset = y * width;
        for x in x_min..=x_max {
            let dx = (x as isize - cx) as f32;
            let r_sq = dx * dx + dy * dy;
            if r_sq > r_sq_max {
                continue;
            }
            let m = mag[row_offset + x];
            if m <= 1e-7 {
                continue;
            }
            let weight = (-r_sq / two_sigma_sq).exp();
            let ang = angle[row_offset + x];
            let bin = ((ang * bin_scale).floor() as usize) % NUM_BINS;
            hist[bin] += m * weight;
        }
    }

    // Circular [1/3, 1/3, 1/3] smoothing pass
    let mut smoothed = [0.0f32; NUM_BINS];
    for b in 0..NUM_BINS {
        let prev = hist[(b + NUM_BINS - 1) % NUM_BINS];
        let curr = hist[b];
        let next = hist[(b + 1) % NUM_BINS];
        smoothed[b] = (prev + curr + next) * (1.0 / 3.0);
    }

    // Locate global maximum peak bin
    let mut best_bin = 0usize;
    let mut best_val = smoothed[0];
    for (b, &val) in smoothed.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_bin = b;
        }
    }

    if best_val <= 1e-7 {
        return vec![0.0];
    }

    let split_threshold = SECONDARY_ORI_PEAK_RATIO * best_val;
    let mut peak_bins: Vec<(usize, f32)> = Vec::with_capacity(MAX_ORIENTATIONS_PER_KEYPOINT);

    for b in 0..NUM_BINS {
        let val = smoothed[b];
        if val < split_threshold {
            continue;
        }
        let left = smoothed[(b + NUM_BINS - 1) % NUM_BINS];
        let right = smoothed[(b + 1) % NUM_BINS];
        // Strict local maximum (with tie-breaking on the right side for flat-topped peaks)
        if val > left && val >= right {
            peak_bins.push((b, val));
        }
    }

    if peak_bins.is_empty() {
        peak_bins.push((best_bin, best_val));
    } else {
        // Order dominant peak first, followed by secondary peaks in descending strength
        peak_bins.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        peak_bins.truncate(MAX_ORIENTATIONS_PER_KEYPOINT);
    }

    peak_bins
        .into_iter()
        .map(|(bin, val)| {
            let left = smoothed[(bin + NUM_BINS - 1) % NUM_BINS];
            let right = smoothed[(bin + 1) % NUM_BINS];
            let denom = left - 2.0 * val + right;
            let offset = if denom.abs() > 1e-6 {
                0.5 * (left - right) / denom
            } else {
                0.0
            }
            .clamp(-0.5, 0.5);

            (((bin as f32) + 0.5 + offset) * (two_pi / (NUM_BINS as f32))).rem_euclid(two_pi)
        })
        .collect()
}

/// Computes the 128-element SIFT descriptor (4 x 4 spatial subregions x 8 orientation bins)
/// rotated to the keypoint's canonical `orientation`, with trilinear interpolation and
/// two-stage L2 normalization + 0.2 clamping.
fn compute_128d_descriptor(
    mag: &[f32],
    angle: &[f32],
    width: usize,
    height: usize,
    kp: &Keypoint,
    orientation: f32,
) -> Vec<f32> {
    const GRID_WIDTH: usize = 4;
    const NUM_ORI_BINS: usize = 8;
    let two_pi = 2.0 * std::f32::consts::PI;

    let mut desc = vec![0.0f32; SIFT_DESCRIPTOR_LEN];
    if width < 3 || height < 3 {
        return desc;
    }

    let cos_t = orientation.cos();
    let sin_t = orientation.sin();

    // Subregion spacing in pixels proportional to keypoint scale
    let hist_width = (3.0 * kp.scale).max(1.5);
    let inv_hist_width = 1.0 / hist_width;

    // Gaussian weighting window across the 4x4 subregion grid (sigma = 0.5 * GRID_WIDTH = 2.0 cells)
    let exp_scale = -1.0 / (2.0 * 2.0 * 2.0);

    let max_radius = ((hist_width * std::f32::consts::SQRT_2 * (GRID_WIDTH as f32 + 1.0) * 0.5)
        .ceil() as isize)
        .clamp(4, 24);

    let cx = kp.x.round() as isize;
    let cy = kp.y.round() as isize;

    let y_min = (cy - max_radius).max(1) as usize;
    let y_max = (cy + max_radius).min((height as isize) - 2).max(1) as usize;
    let x_min = (cx - max_radius).max(1) as usize;
    let x_max = (cx + max_radius).min((width as isize) - 2).max(1) as usize;

    let bin_per_rad = (NUM_ORI_BINS as f32) / two_pi;

    for y in y_min..=y_max {
        let dy = (y as f32) - kp.y;
        let row_offset = y * width;
        for x in x_min..=x_max {
            let dx = (x as f32) - kp.x;

            // Rotate offset into canonical keypoint frame normalized by subregion width
            let r_rot = (dx * cos_t + dy * sin_t) * inv_hist_width;
            let c_rot = (-dx * sin_t + dy * cos_t) * inv_hist_width;

            // Map into [0..4) subregion grid coordinates (centered at 1.5, 1.5)
            let r_bin = r_rot + (GRID_WIDTH as f32) * 0.5 - 0.5;
            let c_bin = c_rot + (GRID_WIDTH as f32) * 0.5 - 0.5;

            if r_bin <= -1.0
                || r_bin >= (GRID_WIDTH as f32)
                || c_bin <= -1.0
                || c_bin >= (GRID_WIDTH as f32)
            {
                continue;
            }

            let m = mag[row_offset + x];
            if m <= 1e-7 {
                continue;
            }

            let w_gauss = ((r_rot * r_rot + c_rot * c_rot) * exp_scale).exp();
            let weighted_mag = m * w_gauss;

            let rel_angle = (angle[row_offset + x] - orientation).rem_euclid(two_pi);
            let o_bin = rel_angle * bin_per_rad;

            let r0 = r_bin.floor() as isize;
            let c0 = c_bin.floor() as isize;
            let o0 = o_bin.floor() as isize;

            let dr = r_bin - (r0 as f32);
            let dc = c_bin - (c0 as f32);
            let do_frac = o_bin - (o0 as f32);

            // Trilinear distribution across adjacent (row, col, orientation) bins
            for ir in 0..=1isize {
                let rb = r0 + ir;
                if rb < 0 || rb >= (GRID_WIDTH as isize) {
                    continue;
                }
                let wr = if ir == 0 { 1.0 - dr } else { dr };

                for ic in 0..=1isize {
                    let cb = c0 + ic;
                    if cb < 0 || cb >= (GRID_WIDTH as isize) {
                        continue;
                    }
                    let wc = if ic == 0 { 1.0 - dc } else { dc };

                    for io in 0..=1isize {
                        let ob = (o0 + io).rem_euclid(NUM_ORI_BINS as isize) as usize;
                        let wo = if io == 0 { 1.0 - do_frac } else { do_frac };

                        let idx = ((rb as usize) * GRID_WIDTH + (cb as usize)) * NUM_ORI_BINS + ob;
                        desc[idx] += weighted_mag * wr * wc * wo;
                    }
                }
            }
        }
    }

    // Stage 1: L2 normalization
    let norm1 = desc.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm1 > 1e-7 {
        let inv_norm1 = 1.0 / norm1;
        for v in &mut desc {
            // Stage 2: Non-linear illumination saturation clamp at 0.2 (US6711293B1)
            *v = (*v * inv_norm1).min(0.2);
        }
        // Stage 3: Final L2 re-normalization
        let norm2 = desc.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm2 > 1e-7 {
            let inv_norm2 = 1.0 / norm2;
            for v in &mut desc {
                *v *= inv_norm2;
            }
        }
    }

    desc
}

/// Computes canonical orientations (including >=80% secondary orientation splitting) and 128-D SIFT
/// descriptors for a slice of detected keypoints in parallel across CPU cores via `rayon`.
///
/// Gradient magnitude and orientation planes are cached per 1/6-octave scale bucket (and computed
/// in parallel across distinct buckets) so multi-thousand keypoint sets require only a small fixed
/// number of Gaussian blurs while preserving deterministic output order.
pub fn compute_sift_descriptors(img: &Array2<f32>, keypoints: &[Keypoint]) -> Vec<SiftFeature> {
    let (height, width) = img.dim();
    if height < 3 || width < 3 || keypoints.is_empty() {
        return Vec::new();
    }

    // Deduplicate unique keypoint scale buckets in deterministic order
    let mut unique_scales: Vec<(i32, f32)> = Vec::new();
    for kp in keypoints {
        let (key, bucket_sigma) = scale_bucket_key(kp.scale);
        if !unique_scales.iter().any(|&(k, _)| k == key) {
            unique_scales.push((key, bucket_sigma));
        }
    }

    // Compute gradient magnitude & orientation planes in parallel across distinct scale buckets
    let scale_cache: Vec<ScaleGradientPlanes> = unique_scales
        .par_iter()
        .map(|&(key, sigma)| ScaleGradientPlanes::compute(img, key, sigma))
        .collect();

    // Compute primary + secondary (>=80% peak) orientations and 128-D descriptors in parallel
    keypoints
        .par_iter()
        .flat_map_iter(|kp| {
            let (key, _) = scale_bucket_key(kp.scale);
            let planes = scale_cache
                .iter()
                .find(|p| p.scale_key == key)
                .expect("scale bucket key was pre-indexed");

            let orientations =
                compute_keypoint_orientations(&planes.mag, &planes.angle, width, height, kp);

            orientations.into_iter().map(move |orientation| {
                let descriptor = compute_128d_descriptor(
                    &planes.mag,
                    &planes.angle,
                    width,
                    height,
                    kp,
                    orientation,
                );

                SiftFeature {
                    x: kp.x,
                    y: kp.y,
                    scale: kp.scale,
                    response: kp.response,
                    orientation,
                    descriptor,
                }
            })
        })
        .collect()
}

/// End-to-end SIFT feature extraction pipeline: detects 3D DoG keypoints with sub-pixel/sub-scale
/// 3D Taylor refinement, retains up to `MAX_FEATURES_PER_IMAGE` highest-contrast keypoints, and
/// computes their canonical orientations (with >=80% secondary peak splitting) and 128-D descriptors.
pub fn extract_sift_features(
    img: &Array2<f32>,
    octaves: usize,
    scales: usize,
    threshold: f32,
) -> Vec<SiftFeature> {
    let mut keypoints = detect_keypoints(img, octaves, scales, threshold);
    if keypoints.len() > MAX_FEATURES_PER_IMAGE {
        keypoints.sort_by(|a, b| {
            b.response
                .abs()
                .partial_cmp(&a.response.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        keypoints.truncate(MAX_FEATURES_PER_IMAGE);
    }
    let mut features = compute_sift_descriptors(img, &keypoints);
    if features.len() > MAX_FEATURES_PER_IMAGE {
        features.truncate(MAX_FEATURES_PER_IMAGE);
    }
    features
}

/// Computes squared Euclidean distance between two 128-D descriptors with early exit
/// as soon as the partial sum exceeds `cutoff_sq`.
#[inline]
fn descriptor_dist_sq_early_exit(a: &[f32], b: &[f32], cutoff_sq: f32) -> f32 {
    let mut sum = 0.0f32;
    for (chunk_a, chunk_b) in a
        .as_chunks::<32>()
        .0
        .iter()
        .zip(b.as_chunks::<32>().0.iter())
    {
        for (&va, &vb) in chunk_a.iter().zip(chunk_b.iter()) {
            let diff = va - vb;
            sum += diff * diff;
        }
        if sum >= cutoff_sq {
            return sum;
        }
    }
    sum
}

/// Matches SIFT features from `query` to `target` in parallel across CPU cores via `rayon`
/// using Euclidean (L2) descriptor distance with 32-element chunked early-exit pruning and
/// Lowe's nearest-neighbor ambiguity ratio test (`d1 / d2 < ratio_threshold`).
pub fn match_descriptors(
    query: &[SiftFeature],
    target: &[SiftFeature],
    ratio_threshold: f32,
) -> Vec<FeatureMatch> {
    if query.is_empty() || target.len() < 2 {
        return Vec::new();
    }

    let query_slice = if query.len() > MAX_FEATURES_PER_IMAGE {
        &query[..MAX_FEATURES_PER_IMAGE]
    } else {
        query
    };
    let target_slice = if target.len() > MAX_FEATURES_PER_IMAGE {
        &target[..MAX_FEATURES_PER_IMAGE]
    } else {
        target
    };

    query_slice
        .par_iter()
        .enumerate()
        .filter_map(|(q_idx, q_feat)| {
            if q_feat.descriptor.len() != SIFT_DESCRIPTOR_LEN {
                return None;
            }

            let mut best_dist_sq = f32::INFINITY;
            let mut second_dist_sq = f32::INFINITY;
            let mut best_t_idx = 0usize;

            for (t_idx, t_feat) in target_slice.iter().enumerate() {
                if t_feat.descriptor.len() != SIFT_DESCRIPTOR_LEN {
                    continue;
                }

                let dist_sq = descriptor_dist_sq_early_exit(
                    &q_feat.descriptor,
                    &t_feat.descriptor,
                    second_dist_sq,
                );

                if dist_sq < best_dist_sq {
                    second_dist_sq = best_dist_sq;
                    best_dist_sq = dist_sq;
                    best_t_idx = t_idx;
                } else if dist_sq < second_dist_sq {
                    second_dist_sq = dist_sq;
                }
            }

            if second_dist_sq > 1e-12 && best_dist_sq.is_finite() {
                let d1 = best_dist_sq.sqrt();
                let d2 = second_dist_sq.sqrt();
                let ratio = d1 / d2;

                if ratio < ratio_threshold {
                    let t_feat = &target_slice[best_t_idx];
                    return Some(FeatureMatch {
                        query_index: q_idx,
                        target_index: best_t_idx,
                        query_point: (q_feat.x, q_feat.y),
                        target_point: (t_feat.x, t_feat.y),
                        distance: d1,
                        ratio,
                        displacement: (t_feat.x - q_feat.x, t_feat.y - q_feat.y),
                    });
                }
            }

            None
        })
        .collect()
}

/// Deterministic 64-bit SplitMix64 generator for reproducible RANSAC pair sampling.
#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Evaluates how many correspondences in `matches` agree with similarity transform `(a, b, tx, ty)`
/// within `thresh_sq`, returning `(inlier_count, total_inlier_error_sq)`.
#[inline]
fn evaluate_similarity_hypothesis(
    matches: &[FeatureMatch],
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
    thresh_sq: f32,
) -> (usize, f32) {
    let mut count = 0usize;
    let mut err_sum = 0.0f32;
    for m in matches {
        let (qx, qy) = m.query_point;
        let (tx_pt, ty_pt) = m.target_point;
        let pred_x = a * qx - b * qy + tx;
        let pred_y = b * qx + a * qy + ty;
        let dx = pred_x - tx_pt;
        let dy = pred_y - ty_pt;
        let err_sq = dx * dx + dy * dy;
        if err_sq <= thresh_sq {
            count += 1;
            err_sum += err_sq;
        }
    }
    (count, err_sum)
}

/// Closed-form 2D Orthogonal Procrustes least-squares similarity fit over a slice of inlier matches.
fn fit_procrustes_similarity(inliers: &[FeatureMatch]) -> Option<SimilarityTransform2D> {
    if inliers.is_empty() {
        return None;
    }
    if inliers.len() == 1 {
        let m = &inliers[0];
        return Some(SimilarityTransform2D {
            a: 1.0,
            b: 0.0,
            tx: m.displacement.0,
            ty: m.displacement.1,
            scale: 1.0,
            rotation_deg: 0.0,
        });
    }

    let inv_n = 1.0 / (inliers.len() as f32);
    let mut mean_qx = 0.0f32;
    let mut mean_qy = 0.0f32;
    let mut mean_tx = 0.0f32;
    let mut mean_ty = 0.0f32;

    for m in inliers {
        mean_qx += m.query_point.0;
        mean_qy += m.query_point.1;
        mean_tx += m.target_point.0;
        mean_ty += m.target_point.1;
    }
    mean_qx *= inv_n;
    mean_qy *= inv_n;
    mean_tx *= inv_n;
    mean_ty *= inv_n;

    let mut s_qq = 0.0f32;
    let mut num_a = 0.0f32;
    let mut num_b = 0.0f32;

    for m in inliers {
        let qx = m.query_point.0 - mean_qx;
        let qy = m.query_point.1 - mean_qy;
        let tx = m.target_point.0 - mean_tx;
        let ty = m.target_point.1 - mean_ty;

        s_qq += qx * qx + qy * qy;
        num_a += qx * tx + qy * ty;
        num_b += qx * ty - qy * tx;
    }

    let (a, b) = if s_qq > 1e-5 {
        let a_fit = num_a / s_qq;
        let b_fit = num_b / s_qq;
        let scale_sq = a_fit * a_fit + b_fit * b_fit;
        if (0.04..=25.0).contains(&scale_sq) {
            (a_fit, b_fit)
        } else {
            (1.0, 0.0)
        }
    } else {
        (1.0, 0.0)
    };

    let tx = mean_tx - (a * mean_qx - b * mean_qy);
    let ty = mean_ty - (b * mean_qx + a * mean_qy);
    let scale = (a * a + b * b).sqrt();
    let rotation_deg = b.atan2(a).to_degrees();

    Some(SimilarityTransform2D {
        a,
        b,
        tx,
        ty,
        scale,
        rotation_deg,
    })
}

/// Filters raw SIFT correspondences using deterministic 2D Similarity RANSAC
/// (translation + rotation + uniform scale) followed by Orthogonal Procrustes refinement.
///
/// Returns `(inlier_matches, best_similarity_transform)`.
pub fn filter_matches_ransac(
    matches: &[FeatureMatch],
    reproj_threshold: f32,
) -> (Vec<FeatureMatch>, Option<SimilarityTransform2D>) {
    if matches.is_empty() {
        return (Vec::new(), None);
    }
    if matches.len() == 1 {
        let transform = fit_procrustes_similarity(matches);
        return (matches.to_vec(), transform);
    }

    let thresh = reproj_threshold.max(0.5);
    let thresh_sq = thresh * thresh;

    // Prioritize highest-confidence matches (lowest Lowe ambiguity ratio) when proposing hypotheses
    let mut sorted_indices: Vec<usize> = (0..matches.len()).collect();
    sorted_indices.sort_by(|&i, &j| {
        matches[i]
            .ratio
            .partial_cmp(&matches[j].ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best_model = (
        1.0f32,
        0.0f32,
        matches[sorted_indices[0]].displacement.0,
        matches[sorted_indices[0]].displacement.1,
    );
    let (mut best_count, mut best_err_sq) = evaluate_similarity_hypothesis(
        matches,
        best_model.0,
        best_model.1,
        best_model.2,
        best_model.3,
        thresh_sq,
    );

    // Phase A: Evaluate top-32 1-point pure-translation hypotheses (zero angular lever-arm noise)
    for &idx in sorted_indices.iter().take(32) {
        let (tx, ty) = matches[idx].displacement;
        let (count, err_sq) = evaluate_similarity_hypothesis(matches, 1.0, 0.0, tx, ty, thresh_sq);
        if count > best_count || (count == best_count && err_sq < best_err_sq) {
            best_count = count;
            best_err_sq = err_sq;
            best_model = (1.0, 0.0, tx, ty);
        }
    }

    // Phase B: Evaluate up to 256 deterministic 2-point similarity (scale + rotation + translation) hypotheses
    let pool_len = sorted_indices.len().min(128);
    let mut rng_state = 0xD1B5_4A32_D192_ED03u64;
    let max_iters = 256usize;

    for _ in 0..max_iters {
        let i1 = (splitmix64_next(&mut rng_state) as usize) % pool_len;
        let mut i2 = (splitmix64_next(&mut rng_state) as usize) % (pool_len - 1);
        if i2 >= i1 {
            i2 += 1;
        }

        let m1 = &matches[sorted_indices[i1]];
        let m2 = &matches[sorted_indices[i2]];

        let dx_q = m2.query_point.0 - m1.query_point.0;
        let dy_q = m2.query_point.1 - m1.query_point.1;
        let dq_sq = dx_q * dx_q + dy_q * dy_q;
        if dq_sq < 16.0 {
            continue;
        }

        let dx_t = m2.target_point.0 - m1.target_point.0;
        let dy_t = m2.target_point.1 - m1.target_point.1;

        let a = (dx_q * dx_t + dy_q * dy_t) / dq_sq;
        let b = (dx_q * dy_t - dy_q * dx_t) / dq_sq;
        let scale_sq = a * a + b * b;
        if !(0.04..=25.0).contains(&scale_sq) {
            continue;
        }

        let tx = m1.target_point.0 - (a * m1.query_point.0 - b * m1.query_point.1);
        let ty = m1.target_point.1 - (b * m1.query_point.0 + a * m1.query_point.1);

        let (count, err_sq) = evaluate_similarity_hypothesis(matches, a, b, tx, ty, thresh_sq);
        if count > best_count || (count == best_count && err_sq < best_err_sq) {
            best_count = count;
            best_err_sq = err_sq;
            best_model = (a, b, tx, ty);
        }
    }

    // Collect initial consensus inliers
    let (a0, b0, tx0, ty0) = best_model;
    let mut inliers: Vec<FeatureMatch> = matches
        .iter()
        .copied()
        .filter(|m| {
            let (qx, qy) = m.query_point;
            let (tx_pt, ty_pt) = m.target_point;
            let dx = (a0 * qx - b0 * qy + tx0) - tx_pt;
            let dy = (b0 * qx + a0 * qy + ty0) - ty_pt;
            dx * dx + dy * dy <= thresh_sq
        })
        .collect();

    // Refine model via Procrustes least-squares over the consensus set
    let refined = fit_procrustes_similarity(&inliers);
    if let Some(ref model) = refined {
        let refined_inliers: Vec<FeatureMatch> = matches
            .iter()
            .copied()
            .filter(|m| {
                let (qx, qy) = m.query_point;
                let (tx_pt, ty_pt) = m.target_point;
                let dx = (model.a * qx - model.b * qy + model.tx) - tx_pt;
                let dy = (model.b * qx + model.a * qy + model.ty) - ty_pt;
                dx * dx + dy * dy <= thresh_sq
            })
            .collect();
        if refined_inliers.len() >= inliers.len() {
            inliers = refined_inliers;
        }
    }

    let final_transform = fit_procrustes_similarity(&inliers).or(refined);
    (inliers, final_transform)
}

/// Computes the statistical median of a mutable slice of `f32` values.
fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

/// Computes an aggregate frame-to-frame tracking / alignment summary (mean & median translation,
/// scale ratio, and circular mean rotation in degrees) from matched SIFT correspondences,
/// automatically rejecting outlier correspondences via 2D Similarity RANSAC.
pub fn summarize_tracking(
    query: &[SiftFeature],
    target: &[SiftFeature],
    matches: &[FeatureMatch],
) -> TrackingSummary {
    if matches.is_empty() {
        return TrackingSummary {
            raw_match_count: 0,
            match_count: 0,
            inlier_ratio: 0.0,
            mean_displacement: (0.0, 0.0),
            median_displacement: (0.0, 0.0),
            mean_scale_ratio: 1.0,
            mean_rotation_deg: 0.0,
        };
    }

    let (inliers, _) = filter_matches_ransac(matches, DEFAULT_RANSAC_REPROJ_THRESHOLD);
    let active = if inliers.is_empty() {
        matches
    } else {
        inliers.as_slice()
    };

    let inv_n = 1.0 / (active.len() as f32);
    let mut sum_dx = 0.0f32;
    let mut sum_dy = 0.0f32;
    let mut sum_scale_ratio = 0.0f32;
    let mut sum_sin = 0.0f32;
    let mut sum_cos = 0.0f32;
    let mut dxs = Vec::with_capacity(active.len());
    let mut dys = Vec::with_capacity(active.len());

    for m in active {
        sum_dx += m.displacement.0;
        sum_dy += m.displacement.1;
        dxs.push(m.displacement.0);
        dys.push(m.displacement.1);

        let q = &query[m.query_index];
        let t = &target[m.target_index];
        if q.scale > 1e-6 {
            sum_scale_ratio += t.scale / q.scale;
        }
        let d_theta = t.orientation - q.orientation;
        sum_sin += d_theta.sin();
        sum_cos += d_theta.cos();
    }

    let mean_rot_rad = sum_sin.atan2(sum_cos);
    let median_dx = median_f32(&mut dxs);
    let median_dy = median_f32(&mut dys);

    TrackingSummary {
        raw_match_count: matches.len(),
        match_count: active.len(),
        inlier_ratio: (active.len() as f32) / (matches.len() as f32),
        mean_displacement: (sum_dx * inv_n, sum_dy * inv_n),
        median_displacement: (median_dx, median_dy),
        mean_scale_ratio: sum_scale_ratio * inv_n,
        mean_rotation_deg: mean_rot_rad.to_degrees(),
    }
}

/// Solves an 8x8 linear system `A * x = b` in `f64` using Gaussian elimination with partial pivoting.
fn solve_linear_8x8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut pivot_row = col;
        let mut max_abs = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            let v = a_row[col].abs();
            if v > max_abs {
                max_abs = v;
                pivot_row = row;
            }
        }
        if max_abs < 1e-11 {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..8 {
            let factor = a[row][col] / pivot;
            if factor.abs() > 0.0 {
                a[row][col] = 0.0;
                let (row_col_slice, row_below_slice) = a.split_at_mut(row);
                for (k, cell) in row_below_slice[0].iter_mut().enumerate().skip(col + 1) {
                    *cell -= factor * row_col_slice[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
    }

    let mut x = [0.0f64; 8];
    for row in (0..8).rev() {
        let mut sum = b[row];
        for (k, &xk) in x.iter().enumerate().skip(row + 1) {
            sum -= a[row][k] * xk;
        }
        let diag = a[row][row];
        if diag.abs() < 1e-11 {
            return None;
        }
        x[row] = sum / diag;
    }
    Some(x)
}

/// Fits a 3x3 planar perspective homography `H` over `sample` correspondences (len >= 4)
/// using Hartley isotropic coordinate normalization and Normalized DLT normal equations.
pub fn fit_homography_dlt(sample: &[FeatureMatch]) -> Option<Homography2D> {
    if sample.len() < 4 {
        return None;
    }

    let inv_n = 1.0 / (sample.len() as f64);
    let mut cx_q = 0.0f64;
    let mut cy_q = 0.0f64;
    let mut cx_t = 0.0f64;
    let mut cy_t = 0.0f64;

    for m in sample {
        cx_q += m.query_point.0 as f64;
        cy_q += m.query_point.1 as f64;
        cx_t += m.target_point.0 as f64;
        cy_t += m.target_point.1 as f64;
    }
    cx_q *= inv_n;
    cy_q *= inv_n;
    cx_t *= inv_n;
    cy_t *= inv_n;

    let mut mean_dist_q = 0.0f64;
    let mut mean_dist_t = 0.0f64;
    for m in sample {
        let dx_q = (m.query_point.0 as f64) - cx_q;
        let dy_q = (m.query_point.1 as f64) - cy_q;
        let dx_t = (m.target_point.0 as f64) - cx_t;
        let dy_t = (m.target_point.1 as f64) - cy_t;
        mean_dist_q += (dx_q * dx_q + dy_q * dy_q).sqrt();
        mean_dist_t += (dx_t * dx_t + dy_t * dy_t).sqrt();
    }
    mean_dist_q *= inv_n;
    mean_dist_t *= inv_n;

    if mean_dist_q < 1e-6 || mean_dist_t < 1e-6 {
        return None;
    }

    let scale_q = std::f64::consts::SQRT_2 / mean_dist_q;
    let scale_t = std::f64::consts::SQRT_2 / mean_dist_t;

    let mut ata = [[0.0f64; 8]; 8];
    let mut atb = [0.0f64; 8];

    for m in sample {
        let x = ((m.query_point.0 as f64) - cx_q) * scale_q;
        let y = ((m.query_point.1 as f64) - cy_q) * scale_q;
        let xp = ((m.target_point.0 as f64) - cx_t) * scale_t;
        let yp = ((m.target_point.1 as f64) - cy_t) * scale_t;

        let r1 = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp];
        let r2 = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp];

        for i in 0..8 {
            atb[i] += r1[i] * xp + r2[i] * yp;
            for j in 0..8 {
                ata[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
            }
        }
    }

    let h_norm = solve_linear_8x8(ata, atb)?;
    let hn = [
        [h_norm[0], h_norm[1], h_norm[2]],
        [h_norm[3], h_norm[4], h_norm[5]],
        [h_norm[6], h_norm[7], 1.0],
    ];

    // Denormalize: H = T_t^{-1} * H_norm * T_q
    // where T_q = [[scale_q, 0, -scale_q * cx_q], [0, scale_q, -scale_q * cy_q], [0, 0, 1]]
    // and T_t^{-1} = [[1/scale_t, 0, cx_t], [0, 1/scale_t, cy_t], [0, 0, 1]]
    let t_q = [
        [scale_q, 0.0, -scale_q * cx_q],
        [0.0, scale_q, -scale_q * cy_q],
        [0.0, 0.0, 1.0],
    ];
    let inv_scale_t = 1.0 / scale_t;
    let t_t_inv = [
        [inv_scale_t, 0.0, cx_t],
        [0.0, inv_scale_t, cy_t],
        [0.0, 0.0, 1.0],
    ];

    let mut temp = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            temp[r][c] = hn[r][0] * t_q[0][c] + hn[r][1] * t_q[1][c] + hn[r][2] * t_q[2][c];
        }
    }

    let mut h_raw = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            h_raw[r][c] = t_t_inv[r][0] * temp[0][c]
                + t_t_inv[r][1] * temp[1][c]
                + t_t_inv[r][2] * temp[2][c];
        }
    }

    if h_raw[2][2].abs() < 1e-9 {
        return None;
    }
    let inv_h22 = 1.0 / h_raw[2][2];
    let matrix = [
        [
            (h_raw[0][0] * inv_h22) as f32,
            (h_raw[0][1] * inv_h22) as f32,
            (h_raw[0][2] * inv_h22) as f32,
        ],
        [
            (h_raw[1][0] * inv_h22) as f32,
            (h_raw[1][1] * inv_h22) as f32,
            (h_raw[1][2] * inv_h22) as f32,
        ],
        [
            (h_raw[2][0] * inv_h22) as f32,
            (h_raw[2][1] * inv_h22) as f32,
            1.0f32,
        ],
    ];

    Some(Homography2D { matrix })
}

/// Evaluates how many matches agree with `homography` within `thresh_sq`.
#[inline]
fn evaluate_homography_hypothesis(
    matches: &[FeatureMatch],
    homography: &Homography2D,
    thresh_sq: f32,
) -> (usize, f32) {
    let mut count = 0usize;
    let mut err_sum = 0.0f32;
    for m in matches {
        if let Some((px, py)) = homography.project(m.query_point.0, m.query_point.1) {
            let dx = px - m.target_point.0;
            let dy = py - m.target_point.1;
            let err_sq = dx * dx + dy * dy;
            if err_sq <= thresh_sq {
                count += 1;
                err_sum += err_sq;
            }
        }
    }
    (count, err_sum)
}

/// Estimates a 3x3 2D projective Homography (`Homography2D`) using deterministic 4-point
/// Normalized DLT RANSAC, handling full 3D planar perspective tilt in addition to translation,
/// rotation, and scale.
pub fn estimate_homography_ransac(
    matches: &[FeatureMatch],
    reproj_threshold: f32,
) -> (Vec<FeatureMatch>, Option<Homography2D>) {
    if matches.is_empty() {
        return (Vec::new(), None);
    }

    let thresh = reproj_threshold.max(0.5);
    let thresh_sq = thresh * thresh;

    // Seed initial candidate with the 2D similarity RANSAC model promoted to 3x3 homography
    let (sim_inliers, sim_opt) = filter_matches_ransac(matches, reproj_threshold);
    let mut best_h = sim_opt.map(|s| Homography2D {
        matrix: [[s.a, -s.b, s.tx], [s.b, s.a, s.ty], [0.0, 0.0, 1.0]],
    });
    let (mut best_count, mut best_err_sq) = if let Some(ref h) = best_h {
        evaluate_homography_hypothesis(matches, h, thresh_sq)
    } else {
        (0usize, f32::INFINITY)
    };

    if matches.len() < 4 {
        return (sim_inliers, best_h);
    }

    let mut sorted_indices: Vec<usize> = (0..matches.len()).collect();
    sorted_indices.sort_by(|&i, &j| {
        matches[i]
            .ratio
            .partial_cmp(&matches[j].ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let pool_len = sorted_indices.len().min(128);
    let mut rng_state = 0x517C_C1B7_2722_0A95u64;
    let max_iters = 200usize;

    for _ in 0..max_iters {
        // Sample 4 distinct indices from the highest-confidence pool
        let mut sample_idx = [0usize; 4];
        let mut chosen = 0usize;
        let mut attempts = 0usize;
        while chosen < 4 && attempts < 16 {
            attempts += 1;
            let cand = (splitmix64_next(&mut rng_state) as usize) % pool_len;
            if !sample_idx[..chosen].contains(&cand) {
                sample_idx[chosen] = cand;
                chosen += 1;
            }
        }
        if chosen < 4 {
            continue;
        }

        let quad = [
            matches[sorted_indices[sample_idx[0]]],
            matches[sorted_indices[sample_idx[1]]],
            matches[sorted_indices[sample_idx[2]]],
            matches[sorted_indices[sample_idx[3]]],
        ];

        if let Some(h_cand) = fit_homography_dlt(&quad) {
            let (count, err_sq) = evaluate_homography_hypothesis(matches, &h_cand, thresh_sq);
            if count > best_count || (count == best_count && err_sq < best_err_sq) {
                best_count = count;
                best_err_sq = err_sq;
                best_h = Some(h_cand);
            }
        }
    }

    let Some(initial_h) = best_h else {
        return (sim_inliers, None);
    };

    let mut inliers: Vec<FeatureMatch> = matches
        .iter()
        .copied()
        .filter(|m| {
            initial_h
                .project(m.query_point.0, m.query_point.1)
                .map(|(px, py)| {
                    let dx = px - m.target_point.0;
                    let dy = py - m.target_point.1;
                    dx * dx + dy * dy <= thresh_sq
                })
                .unwrap_or(false)
        })
        .collect();

    // Least-squares DLT refinement over all consensus inliers
    if inliers.len() >= 4
        && let Some(refined_h) = fit_homography_dlt(&inliers)
    {
        let (ref_count, ref_err) = evaluate_homography_hypothesis(matches, &refined_h, thresh_sq);
        if ref_count >= inliers.len() || (ref_count == best_count && ref_err <= best_err_sq) {
            inliers = matches
                .iter()
                .copied()
                .filter(|m| {
                    refined_h
                        .project(m.query_point.0, m.query_point.1)
                        .map(|(px, py)| {
                            let dx = px - m.target_point.0;
                            let dy = py - m.target_point.1;
                            dx * dx + dy * dy <= thresh_sq
                        })
                        .unwrap_or(false)
                })
                .collect();
            return (inliers, Some(refined_h));
        }
    }

    (inliers, Some(initial_h))
}
