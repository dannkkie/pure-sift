#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};

use image::{DynamicImage, Rgba, RgbaImage};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::pipeline::{
    DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD, Homography2D,
    SiftFeature, estimate_homography_ransac, extract_sift_features, filter_matches_ransac,
    luma_to_array2, match_descriptors,
};
use crate::visualize::{
    MAX_WARP_DIM, MAX_WARP_PIXELS, invert_matrix_3x3, project_f64,
    resolve_safe_alignment_homography, sample_bilinear_rgba,
};

/// Projection geometry before panorama alignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionMode {
    /// Planar pinhole projection directly via homographies.
    Planar,
    /// Cylindrical projection with focal length in pixels (`focal_px`).
    Cylindrical {
        /// Camera focal length in pixels used as cylinder radius.
        focal_px: f32,
    },
}

/// Composite blending mode for panorama stitching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Feather weighting proportional to distance from valid image borders (tiled memory-bounded pass).
    Feather,
    /// Burt & Adelson multi-band Laplacian pyramid blending over Voronoi seam masks.
    MultiBand {
        /// Number of Laplacian pyramid levels (`1..=6`).
        levels: usize,
    },
}

/// Configuration for multi-image panorama stitching (`stitch_panorama`).
#[derive(Debug, Clone, PartialEq)]
pub struct PanoramaOptions {
    /// Number of scale-space octaves (`1..=8`).
    pub octaves: usize,
    /// Searchable DoG scales per octave (`3..=10`).
    pub scales: usize,
    /// DoG contrast threshold in `(0.0, 1.0]`.
    pub threshold: f32,
    /// Lowe's descriptor ratio threshold in `(0.0, 1.0]`.
    pub ratio: f32,
    /// RANSAC reprojection error threshold in pixels.
    pub ransac_threshold: f32,
    /// Surface projection mode (`Planar` or `Cylindrical`).
    pub projection: ProjectionMode,
    /// Blending mode (`Feather` or `MultiBand`).
    pub blend: BlendMode,
    /// Enable Brown & Lowe regularized exposure gain compensation.
    pub exposure_compensation: bool,
    /// Maximum allowed output canvas width/height in pixels.
    pub max_canvas_dim: usize,
    /// Maximum allowed total output canvas area in pixels.
    pub max_canvas_pixels: usize,
    /// Tile size in pixels for memory-bounded feather rendering.
    pub tile_size: usize,
}

impl Default for PanoramaOptions {
    fn default() -> Self {
        Self {
            octaves: 3,
            scales: 3,
            threshold: DEFAULT_CONTRAST_THRESHOLD,
            ratio: DEFAULT_MATCH_RATIO,
            ransac_threshold: DEFAULT_RANSAC_REPROJ_THRESHOLD,
            projection: ProjectionMode::Planar,
            blend: BlendMode::MultiBand { levels: 4 },
            exposure_compensation: true,
            max_canvas_dim: MAX_WARP_DIM,
            max_canvas_pixels: MAX_WARP_PIXELS,
            tile_size: 512,
        }
    }
}

/// Output of `stitch_panorama` including the blended RGBA mosaic, registered indices,
/// solved per-image exposure gains, and global image-to-reference homographies.
#[derive(Debug, Clone)]
pub struct PanoramaResult {
    /// Composite stitched RGBA panorama image.
    pub image: RgbaImage,
    /// Index of the chosen reference anchor image in the input slice.
    pub reference_index: usize,
    /// Indices of input images successfully registered into the panorama.
    pub registered_indices: Vec<usize>,
    /// Solved multiplicative exposure gain per input image (`1.0` at reference).
    pub gains: Vec<f32>,
    /// Global homography mapping each input image into reference coordinates (`None` if unregistered).
    pub homographies: Vec<Option<Homography2D>>,
}

#[derive(Clone)]
struct PairEdge {
    from: usize,
    to: usize,
    h_from_to: [[f64; 3]; 3],
    inliers: usize,
    mean_i_from: f64,
    mean_i_to: f64,
}

#[inline]
fn compose_mat3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    if out[2][2].abs() > 1e-12 {
        let inv = 1.0 / out[2][2];
        for row in &mut out {
            for val in row.iter_mut() {
                *val *= inv;
            }
        }
    }
    out
}

#[inline]
fn h_to_f64(h: &Homography2D) -> [[f64; 3]; 3] {
    [
        [
            h.matrix[0][0] as f64,
            h.matrix[0][1] as f64,
            h.matrix[0][2] as f64,
        ],
        [
            h.matrix[1][0] as f64,
            h.matrix[1][1] as f64,
            h.matrix[1][2] as f64,
        ],
        [
            h.matrix[2][0] as f64,
            h.matrix[2][1] as f64,
            h.matrix[2][2] as f64,
        ],
    ]
}

#[inline]
fn f64_to_h(m: &[[f64; 3]; 3]) -> Homography2D {
    Homography2D {
        matrix: [
            [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
            [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
            [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
        ],
    }
}

/// Warps an RGBA image onto a cylinder of radius `focal_px` using inverse bilinear mapping.
pub fn warp_image_cylindrical(img: &DynamicImage, focal_px: f32) -> RgbaImage {
    let rgba = img.to_rgba8();
    let w = rgba.width();
    let h = rgba.height();
    let f = (focal_px as f64).max((w.max(h) as f64) * 0.25).max(16.0);
    let cx = (w as f64 - 1.0) * 0.5;
    let cy = (h as f64 - 1.0) * 0.5;

    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        let dy = y as f64 - cy;
        for x in 0..w {
            let theta = (x as f64 - cx) / f;
            if theta.abs() >= 1.45 {
                continue;
            }
            let cos_t = theta.cos();
            let src_x = f * theta.tan() + cx;
            let src_y = dy / cos_t + cy;
            if let Some((px, _)) = sample_bilinear_rgba(&rgba, src_x, src_y) {
                out.put_pixel(
                    x,
                    y,
                    Rgba([
                        px[0].round().clamp(0.0, 255.0) as u8,
                        px[1].round().clamp(0.0, 255.0) as u8,
                        px[2].round().clamp(0.0, 255.0) as u8,
                        255,
                    ]),
                );
            }
        }
    }
    out
}

/// Samples mean `[0, 255]` intensity over a small patch around `(x, y)` in a grayscale `Array2`.
fn sample_patch_intensity(gray: &ndarray::Array2<f32>, x: f32, y: f32) -> f64 {
    let (h, w) = gray.dim();
    if h == 0 || w == 0 || !x.is_finite() || !y.is_finite() {
        return 0.0;
    }
    let cx = (x.round() as isize).clamp(0, (w - 1) as isize) as usize;
    let cy = (y.round() as isize).clamp(0, (h - 1) as isize) as usize;
    let y0 = cy.saturating_sub(1);
    let y1 = (cy + 1).min(h - 1);
    let x0 = cx.saturating_sub(1);
    let x1 = (cx + 1).min(w - 1);

    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            sum += f64::from(gray[(yy, xx)]);
            cnt += 1;
        }
    }
    if cnt == 0 {
        0.0
    } else {
        (sum / cnt as f64) * 255.0
    }
}

/// Solves Brown & Lowe (2007) regularized linear least-squares gain compensation
/// `g_i` over pairwise overlaps `(i, j, n_ij, mean_i, mean_j)`, anchored at `gains[ref_idx] = 1.0`.
pub fn solve_exposure_gains(
    overlap_pairs: &[(usize, usize, f64, f64, f64)],
    n_images: usize,
    ref_idx: usize,
    sigma_n: f64,
    sigma_g: f64,
) -> Vec<f32> {
    if n_images == 0 {
        return Vec::new();
    }
    let mut a = vec![vec![0.0f64; n_images]; n_images];
    let mut b = vec![0.0f64; n_images];

    let inv_sn2 = 1.0 / sigma_n.max(1e-3).powi(2);
    let prior_w = 1.0 / sigma_g.max(1e-3).powi(2);

    for i in 0..n_images {
        a[i][i] += prior_w;
        b[i] += prior_w;
    }

    for &(i, j, n_ij, mean_i, mean_j) in overlap_pairs {
        if i >= n_images || j >= n_images || i == j || n_ij <= 0.0 {
            continue;
        }
        let w = n_ij * inv_sn2;
        a[i][i] += w * mean_i * mean_i;
        a[j][j] += w * mean_j * mean_j;
        let cross = -w * mean_i * mean_j;
        a[i][j] += cross;
        a[j][i] += cross;
    }

    // Gaussian elimination with partial pivoting
    let mut gains = vec![1.0f64; n_images];
    for col in 0..n_images {
        let mut pivot = col;
        let mut max_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > max_val {
                max_val = row[col].abs();
                pivot = r;
            }
        }
        if max_val < 1e-12 {
            return vec![1.0f32; n_images];
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let diag = a[col][col];
        for r in (col + 1)..n_images {
            let factor = a[r][col] / diag;
            a[r][col] = 0.0;
            for c in (col + 1)..n_images {
                let val = a[col][c];
                a[r][c] -= factor * val;
            }
            b[r] -= factor * b[col];
        }
    }

    for i in (0..n_images).rev() {
        let mut s = b[i];
        for c in (i + 1)..n_images {
            s -= a[i][c] * gains[c];
        }
        gains[i] = s / a[i][i];
    }

    let ref_gain = gains.get(ref_idx).copied().unwrap_or(1.0);
    let norm = if ref_gain.is_finite() && ref_gain > 0.1 {
        1.0 / ref_gain
    } else {
        1.0
    };

    gains
        .into_iter()
        .map(|g| {
            let scaled = if g.is_finite() { g * norm } else { 1.0 };
            (scaled as f32).clamp(0.4, 2.5)
        })
        .collect()
}

fn downsample_half_rgba_weighted(
    src: &[[f32; 3]],
    mask: &[f32],
    w: usize,
    h: usize,
) -> (Vec<[f32; 3]>, Vec<f32>, usize, usize) {
    let nw = w.div_ceil(2).max(1);
    let nh = h.div_ceil(2).max(1);
    let mut dst_rgb = vec![[0.0f32; 3]; nw * nh];
    let mut dst_mask = vec![0.0f32; nw * nh];

    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

    let mut tmp_rgb = vec![[0.0f32; 3]; h * nw];
    let mut tmp_mask = vec![0.0f32; h * nw];

    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mrow = &mask[y * w..(y + 1) * w];
        let out_row = &mut tmp_rgb[y * nw..(y + 1) * nw];
        let out_mrow = &mut tmp_mask[y * nw..(y + 1) * nw];
        for nx in 0..nw {
            let cx = (nx * 2) as isize;
            let mut rgb_acc = [0.0f32; 3];
            let mut m_acc = 0.0f32;
            let mut valid_w = 0.0f32;
            for (ki, &kw) in K.iter().enumerate() {
                let sx = (cx + ki as isize - 2).clamp(0, (w - 1) as isize) as usize;
                let mw = mrow[sx];
                m_acc += kw * mw;
                if mw > 0.0 {
                    let vw = kw * mw;
                    valid_w += vw;
                    rgb_acc[0] += vw * row[sx][0];
                    rgb_acc[1] += vw * row[sx][1];
                    rgb_acc[2] += vw * row[sx][2];
                }
            }
            out_mrow[nx] = m_acc;
            if valid_w > 1e-6 {
                let inv = 1.0 / valid_w;
                out_row[nx] = [rgb_acc[0] * inv, rgb_acc[1] * inv, rgb_acc[2] * inv];
            }
        }
    }

    for ny in 0..nh {
        let cy = (ny * 2) as isize;
        for nx in 0..nw {
            let mut rgb_acc = [0.0f32; 3];
            let mut m_acc = 0.0f32;
            let mut valid_w = 0.0f32;
            for (ki, &kw) in K.iter().enumerate() {
                let sy = (cy + ki as isize - 2).clamp(0, (h - 1) as isize) as usize;
                let idx = sy * nw + nx;
                let mw = tmp_mask[idx];
                m_acc += kw * mw;
                if mw > 0.0 {
                    let vw = kw * mw;
                    valid_w += vw;
                    rgb_acc[0] += vw * tmp_rgb[idx][0];
                    rgb_acc[1] += vw * tmp_rgb[idx][1];
                    rgb_acc[2] += vw * tmp_rgb[idx][2];
                }
            }
            let out_idx = ny * nw + nx;
            dst_mask[out_idx] = m_acc;
            if valid_w > 1e-6 {
                let inv = 1.0 / valid_w;
                dst_rgb[out_idx] = [rgb_acc[0] * inv, rgb_acc[1] * inv, rgb_acc[2] * inv];
            }
        }
    }

    (dst_rgb, dst_mask, nw, nh)
}

fn upsample_to_size_rgb(
    src: &[[f32; 3]],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; dw * dh];
    for y in 0..dh {
        let fy = (y as f32 * 0.5).clamp(0.0, (sh.saturating_sub(1)) as f32);
        let y0 = (fy.floor() as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy1 = fy - (y0 as f32);
        let wy0 = 1.0 - wy1;
        for x in 0..dw {
            let fx = (x as f32 * 0.5).clamp(0.0, (sw.saturating_sub(1)) as f32);
            let x0 = (fx.floor() as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx1 = fx - (x0 as f32);
            let wx0 = 1.0 - wx1;

            let p00 = src[y0 * sw + x0];
            let p10 = src[y0 * sw + x1];
            let p01 = src[y1 * sw + x0];
            let p11 = src[y1 * sw + x1];

            let w00 = wx0 * wy0;
            let w10 = wx1 * wy0;
            let w01 = wx0 * wy1;
            let w11 = wx1 * wy1;

            out[y * dw + x] = [
                p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
                p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
                p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
            ];
        }
    }
    out
}

fn downsample_half_scalar(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let nw = w.div_ceil(2).max(1);
    let nh = h.div_ceil(2).max(1);
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut tmp = vec![0.0f32; h * nw];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out_row = &mut tmp[y * nw..(y + 1) * nw];
        for (nx, out_slot) in out_row.iter_mut().enumerate().take(nw) {
            let cx = (nx * 2) as isize;
            let mut acc = 0.0f32;
            for (ki, &kw) in K.iter().enumerate() {
                let sx = (cx + ki as isize - 2).clamp(0, (w - 1) as isize) as usize;
                acc += kw * row[sx];
            }
            *out_slot = acc;
        }
    }
    let mut dst = vec![0.0f32; nw * nh];
    for ny in 0..nh {
        let cy = (ny * 2) as isize;
        for nx in 0..nw {
            let mut acc = 0.0f32;
            for (ki, &kw) in K.iter().enumerate() {
                let sy = (cy + ki as isize - 2).clamp(0, (h - 1) as isize) as usize;
                acc += kw * tmp[sy * nw + nx];
            }
            dst[ny * nw + nx] = acc;
        }
    }
    (dst, nw, nh)
}

/// Stitches two or more overlapping images into a unified panorama using SIFT feature matching,
/// Maximum-Spanning-Tree homography chaining, optional cylindrical projection, regularized exposure
/// gain compensation, and feather or multi-band Laplacian pyramid blending.
pub fn stitch_panorama(
    images: &[DynamicImage],
    options: &PanoramaOptions,
) -> Result<PanoramaResult, String> {
    if images.len() < 2 {
        return Err("Panorama stitching requires at least 2 input images".to_string());
    }

    let prepared_rgba: Vec<RgbaImage> = images
        .iter()
        .map(|img| match options.projection {
            ProjectionMode::Planar => img.to_rgba8(),
            ProjectionMode::Cylindrical { focal_px } => warp_image_cylindrical(img, focal_px),
        })
        .collect();

    let gray_arrays: Vec<ndarray::Array2<f32>> = prepared_rgba
        .iter()
        .map(|rgba| luma_to_array2(&DynamicImage::ImageRgba8(rgba.clone()).to_luma8()))
        .collect();

    #[cfg(feature = "rayon")]
    let gray_iter = gray_arrays.par_iter();
    #[cfg(not(feature = "rayon"))]
    let gray_iter = gray_arrays.iter();
    let features: Vec<Vec<SiftFeature>> = gray_iter
        .map(|arr| extract_sift_features(arr, options.octaves, options.scales, options.threshold))
        .collect();

    let n = prepared_rgba.len();
    let mut pair_indices = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pair_indices.push((i, j));
        }
    }

    let match_pair = |(i, j): (usize, usize)| -> Vec<PairEdge> {
        let raw_matches = match_descriptors(&features[i], &features[j], options.ratio);
        if raw_matches.len() < 6 {
            return Vec::new();
        }
        let (sim_inliers, sim_tf) = filter_matches_ransac(&raw_matches, options.ransac_threshold);
        let (hom_inliers, hom_tf) =
            estimate_homography_ransac(&raw_matches, options.ransac_threshold);
        let best_inliers = if hom_inliers.len() >= sim_inliers.len() {
            &hom_inliers
        } else {
            &sim_inliers
        };
        if best_inliers.len() < 6 {
            return Vec::new();
        }

        let (sw, sh) = (
            prepared_rgba[i].width() as usize,
            prepared_rgba[i].height() as usize,
        );
        let Some(h_ij) =
            resolve_safe_alignment_homography(hom_tf, hom_inliers.len(), sim_tf, sw, sh)
        else {
            return Vec::new();
        };
        let Some(h_ji) = invert_matrix_3x3(&h_ij.matrix) else {
            return Vec::new();
        };

        let mut sum_i = 0.0f64;
        let mut sum_j = 0.0f64;
        for m in best_inliers {
            sum_i += sample_patch_intensity(&gray_arrays[i], m.query_point.0, m.query_point.1);
            sum_j += sample_patch_intensity(&gray_arrays[j], m.target_point.0, m.target_point.1);
        }
        let inv_c = 1.0 / (best_inliers.len() as f64);
        let mean_i = (sum_i * inv_c).max(1.0);
        let mean_j = (sum_j * inv_c).max(1.0);
        let inlier_count = best_inliers.len();

        vec![
            PairEdge {
                from: i,
                to: j,
                h_from_to: h_to_f64(&h_ij),
                inliers: inlier_count,
                mean_i_from: mean_i,
                mean_i_to: mean_j,
            },
            PairEdge {
                from: j,
                to: i,
                h_from_to: h_ji,
                inliers: inlier_count,
                mean_i_from: mean_j,
                mean_i_to: mean_i,
            },
        ]
    };

    #[cfg(feature = "rayon")]
    let pair_edges: Vec<PairEdge> = pair_indices.into_par_iter().flat_map(match_pair).collect();
    #[cfg(not(feature = "rayon"))]
    let pair_edges: Vec<PairEdge> = pair_indices.into_iter().flat_map(match_pair).collect();

    if pair_edges.is_empty() {
        return Err("No valid overlapping image pairs could be matched".to_string());
    }

    let mut adj: Vec<Vec<PairEdge>> = vec![Vec::new(); n];
    for edge in pair_edges {
        adj[edge.from].push(edge);
    }

    // Find connected components and pick largest component
    let mut visited = vec![false; n];
    let mut best_comp: Vec<usize> = Vec::new();
    for start in 0..n {
        if visited[start] || adj[start].is_empty() {
            continue;
        }
        let mut comp = Vec::new();
        let mut q = VecDeque::new();
        visited[start] = true;
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for e in &adj[u] {
                if !visited[e.to] {
                    visited[e.to] = true;
                    q.push_back(e.to);
                }
            }
        }
        if comp.len() > best_comp.len() {
            best_comp = comp;
        }
    }

    if best_comp.len() < 2 {
        return Err("Could not form a connected panorama graph of >= 2 images".to_string());
    }

    // Reference image = node in largest component with highest total inlier connectivity
    let ref_idx = *best_comp
        .iter()
        .max_by_key(|&&u| {
            let weight: usize = adj[u].iter().map(|e| e.inliers).sum();
            (weight, u)
        })
        .unwrap();

    // Prim's Maximum Spanning Tree rooted at `ref_idx` to chain H_i_to_ref cleanly
    let mut h_to_ref: Vec<Option<[[f64; 3]; 3]>> = vec![None; n];
    h_to_ref[ref_idx] = Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let mut tree_edges = Vec::new();

    for _ in 1..best_comp.len() {
        let mut best_edge: Option<&PairEdge> = None;
        for &u in &best_comp {
            if h_to_ref[u].is_none() {
                continue;
            }
            for e in &adj[u] {
                // `e` goes from `u` (already registered) -> `e.to` (unregistered), so reverse edge gives `e.to -> u`
                if h_to_ref[e.to].is_none() && best_edge.is_none_or(|b| e.inliers > b.inliers) {
                    best_edge = Some(e);
                }
            }
        }
        let Some(fwd) = best_edge else {
            break;
        };
        let parent = fwd.from;
        let child = fwd.to;
        let rev = adj[child]
            .iter()
            .find(|e| e.to == parent)
            .expect("undirected pair edge");
        let h_parent_ref = h_to_ref[parent].unwrap();
        h_to_ref[child] = Some(compose_mat3(&h_parent_ref, &rev.h_from_to));
        tree_edges.push(rev.clone());
    }

    let registered_indices: Vec<usize> = (0..n).filter(|&i| h_to_ref[i].is_some()).collect();

    // Exposure gain compensation over graph edges in the registered component
    let gains = if options.exposure_compensation {
        let mut seen = BTreeMap::<(usize, usize), (f64, f64, f64)>::new();
        for &u in &registered_indices {
            for e in &adj[u] {
                if h_to_ref[e.to].is_some() {
                    let key = if e.from < e.to {
                        (e.from, e.to)
                    } else {
                        (e.to, e.from)
                    };
                    let (m1, m2) = if e.from < e.to {
                        (e.mean_i_from, e.mean_i_to)
                    } else {
                        (e.mean_i_to, e.mean_i_from)
                    };
                    seen.insert(key, (e.inliers as f64, m1, m2));
                }
            }
        }
        let pairs: Vec<(usize, usize, f64, f64, f64)> = seen
            .into_iter()
            .map(|((a, b), (cnt, ma, mb))| (a, b, cnt, ma, mb))
            .collect();
        solve_exposure_gains(&pairs, n, ref_idx, 10.0, 0.25)
    } else {
        vec![1.0f32; n]
    };

    // Union bounding box in reference coordinates
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut inv_mats: Vec<Option<[[f64; 3]; 3]>> = vec![None; n];
    let mut out_homographies: Vec<Option<Homography2D>> = vec![None; n];

    for &idx in &registered_indices {
        let h_mat = h_to_ref[idx].unwrap();
        let h2d = f64_to_h(&h_mat);
        let inv = invert_matrix_3x3(&h2d.matrix)
            .ok_or_else(|| format!("Registered image {idx} has singular homography"))?;
        out_homographies[idx] = Some(h2d);
        inv_mats[idx] = Some(inv);

        let w = (prepared_rgba[idx].width() as f64 - 1.0).max(0.0);
        let h = (prepared_rgba[idx].height() as f64 - 1.0).max(0.0);
        for &(cx, cy) in &[(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
            let (px, py) = project_f64(&h_mat, cx, cy)
                .ok_or_else(|| format!("Projected corner hit infinity for image {idx}"))?;
            if !px.is_finite() || !py.is_finite() {
                return Err(format!("Non-finite projected corner for image {idx}"));
            }
            min_x = min_x.min(px);
            min_y = min_y.min(py);
            max_x = max_x.max(px);
            max_y = max_y.max(py);
        }
    }

    let origin_x = min_x.floor();
    let origin_y = min_y.floor();
    let out_w_f = (max_x.ceil() - origin_x + 1.0).max(1.0);
    let out_h_f = (max_y.ceil() - origin_y + 1.0).max(1.0);

    if out_w_f > options.max_canvas_dim as f64
        || out_h_f > options.max_canvas_dim as f64
        || (out_w_f * out_h_f) > options.max_canvas_pixels as f64
    {
        return Err(format!(
            "Panorama canvas size {}x{} exceeds safety limits ({} dim, {} px)",
            out_w_f as usize, out_h_f as usize, options.max_canvas_dim, options.max_canvas_pixels
        ));
    }

    let out_w = out_w_f as usize;
    let out_h = out_h_f as usize;
    let mut canvas = RgbaImage::new(out_w as u32, out_h as u32);

    match options.blend {
        BlendMode::Feather => {
            let tile = options.tile_size.max(64);
            for ty0 in (0..out_h).step_by(tile) {
                let ty1 = (ty0 + tile).min(out_h);
                for tx0 in (0..out_w).step_by(tile) {
                    let tx1 = (tx0 + tile).min(out_w);
                    for oy in ty0..ty1 {
                        let ref_y = origin_y + oy as f64;
                        for ox in tx0..tx1 {
                            let ref_x = origin_x + ox as f64;
                            let mut acc_rgb = [0.0f64; 3];
                            let mut acc_w = 0.0f64;
                            for &idx in &registered_indices {
                                let inv = inv_mats[idx].as_ref().unwrap();
                                if let Some((sx, sy)) = project_f64(inv, ref_x, ref_y)
                                    && let Some((rgba_px, fw)) =
                                        sample_bilinear_rgba(&prepared_rgba[idx], sx, sy)
                                {
                                    let g = gains[idx] as f64;
                                    let w = fw.max(1e-4);
                                    acc_rgb[0] += w * (rgba_px[0] * g).clamp(0.0, 255.0);
                                    acc_rgb[1] += w * (rgba_px[1] * g).clamp(0.0, 255.0);
                                    acc_rgb[2] += w * (rgba_px[2] * g).clamp(0.0, 255.0);
                                    acc_w += w;
                                }
                            }
                            if acc_w > 0.0 {
                                let inv_w = 1.0 / acc_w;
                                canvas.put_pixel(
                                    ox as u32,
                                    oy as u32,
                                    Rgba([
                                        (acc_rgb[0] * inv_w).round().clamp(0.0, 255.0) as u8,
                                        (acc_rgb[1] * inv_w).round().clamp(0.0, 255.0) as u8,
                                        (acc_rgb[2] * inv_w).round().clamp(0.0, 255.0) as u8,
                                        255,
                                    ]),
                                );
                            }
                        }
                    }
                }
            }
        }
        BlendMode::MultiBand { levels } => {
            let npix = out_w * out_h;
            let k_reg = registered_indices.len();
            let mut warped_rgbs = vec![vec![[0.0f32; 3]; npix]; k_reg];
            let mut valid_masks = vec![vec![0.0f32; npix]; k_reg];
            let mut best_weight = vec![-1.0f32; npix];
            let mut best_owner = vec![usize::MAX; npix];

            for (pos, &img_idx) in registered_indices.iter().enumerate() {
                let inv = inv_mats[img_idx].as_ref().unwrap();
                let gain = gains[img_idx];
                let src = &prepared_rgba[img_idx];
                for oy in 0..out_h {
                    let ref_y = origin_y + oy as f64;
                    for ox in 0..out_w {
                        let ref_x = origin_x + ox as f64;
                        if let Some((sx, sy)) = project_f64(inv, ref_x, ref_y)
                            && let Some((rgba_px, fw)) = sample_bilinear_rgba(src, sx, sy)
                        {
                            let p_idx = oy * out_w + ox;
                            warped_rgbs[pos][p_idx] = [
                                (rgba_px[0] as f32 * gain).clamp(0.0, 255.0),
                                (rgba_px[1] as f32 * gain).clamp(0.0, 255.0),
                                (rgba_px[2] as f32 * gain).clamp(0.0, 255.0),
                            ];
                            valid_masks[pos][p_idx] = 1.0;
                            let fw_f32 = fw as f32;
                            if fw_f32 > best_weight[p_idx] {
                                best_weight[p_idx] = fw_f32;
                                best_owner[p_idx] = pos;
                            }
                        }
                    }
                }
            }

            let max_possible_levels = ((out_w.min(out_h) as f64).log2().floor() as usize).max(1);
            let num_levels = levels.clamp(1, 6).min(max_possible_levels);

            let mut blended_pyr: Vec<Vec<[f32; 3]>> = Vec::with_capacity(num_levels);
            let mut mask_sum_pyr: Vec<Vec<f32>> = Vec::with_capacity(num_levels);
            let mut level_dims: Vec<(usize, usize)> = Vec::with_capacity(num_levels);
            {
                let mut cw = out_w;
                let mut ch = out_h;
                for _ in 0..num_levels {
                    level_dims.push((cw, ch));
                    blended_pyr.push(vec![[0.0f32; 3]; cw * ch]);
                    mask_sum_pyr.push(vec![0.0f32; cw * ch]);
                    cw = cw.div_ceil(2).max(1);
                    ch = ch.div_ceil(2).max(1);
                }
            }

            for pos in 0..k_reg {
                // Hard Voronoi seam mask (1.0 if image `pos` owns pixel, 0.0 otherwise)
                let seam_mask: Vec<f32> = best_owner
                    .iter()
                    .map(|&owner| if owner == pos { 1.0 } else { 0.0 })
                    .collect();

                // Build Gaussian pyramids for warped image `pos` and its Voronoi seam mask
                let mut g_img = Vec::with_capacity(num_levels);
                let mut g_valid = Vec::with_capacity(num_levels);
                let mut g_mask = Vec::with_capacity(num_levels);

                g_img.push(std::mem::take(&mut warped_rgbs[pos]));
                g_valid.push(std::mem::take(&mut valid_masks[pos]));
                g_mask.push(seam_mask);

                for lvl in 1..num_levels {
                    let (pw, ph) = level_dims[lvl - 1];
                    let (next_img, next_val, _, _) =
                        downsample_half_rgba_weighted(&g_img[lvl - 1], &g_valid[lvl - 1], pw, ph);
                    let (next_mask, _, _) = downsample_half_scalar(&g_mask[lvl - 1], pw, ph);
                    g_img.push(next_img);
                    g_valid.push(next_val);
                    g_mask.push(next_mask);
                }

                // Accumulate mask-weighted Laplacian bands into blended_pyr (strictly inside valid footprint)
                let top = num_levels - 1;
                for (i, &w) in g_mask[top].iter().enumerate() {
                    if w > 0.0 && g_valid[top][i] > 1e-3 {
                        blended_pyr[top][i][0] += w * g_img[top][i][0];
                        blended_pyr[top][i][1] += w * g_img[top][i][1];
                        blended_pyr[top][i][2] += w * g_img[top][i][2];
                        mask_sum_pyr[top][i] += w;
                    }
                }

                for lvl in 0..top {
                    let (lw, lh) = level_dims[lvl];
                    let (nw, nh) = level_dims[lvl + 1];
                    let expanded = upsample_to_size_rgb(&g_img[lvl + 1], nw, nh, lw, lh);
                    for i in 0..(lw * lh) {
                        let w = g_mask[lvl][i];
                        if w > 0.0 && g_valid[lvl][i] > 1e-3 {
                            let lap = [
                                g_img[lvl][i][0] - expanded[i][0],
                                g_img[lvl][i][1] - expanded[i][1],
                                g_img[lvl][i][2] - expanded[i][2],
                            ];
                            blended_pyr[lvl][i][0] += w * lap[0];
                            blended_pyr[lvl][i][1] += w * lap[1];
                            blended_pyr[lvl][i][2] += w * lap[2];
                            mask_sum_pyr[lvl][i] += w;
                        }
                    }
                }
            }

            // Normalize accumulated bands and collapse Laplacian pyramid top-down
            for lvl in 0..num_levels {
                for (px, &msum) in blended_pyr[lvl].iter_mut().zip(mask_sum_pyr[lvl].iter()) {
                    if msum > 1e-6 && (msum - 1.0).abs() > 1e-5 {
                        let inv = 1.0 / msum;
                        px[0] *= inv;
                        px[1] *= inv;
                        px[2] *= inv;
                    }
                }
            }

            let mut recon = std::mem::take(&mut blended_pyr[num_levels - 1]);
            for lvl in (0..(num_levels - 1)).rev() {
                let (lw, lh) = level_dims[lvl];
                let (nw, nh) = level_dims[lvl + 1];
                let up = upsample_to_size_rgb(&recon, nw, nh, lw, lh);
                let mut next_recon = vec![[0.0f32; 3]; lw * lh];
                for i in 0..(lw * lh) {
                    next_recon[i] = [
                        up[i][0] + blended_pyr[lvl][i][0],
                        up[i][1] + blended_pyr[lvl][i][1],
                        up[i][2] + blended_pyr[lvl][i][2],
                    ];
                }
                recon = next_recon;
            }

            for oy in 0..out_h {
                for ox in 0..out_w {
                    let idx = oy * out_w + ox;
                    if best_owner[idx] != usize::MAX {
                        canvas.put_pixel(
                            ox as u32,
                            oy as u32,
                            Rgba([
                                recon[idx][0].round().clamp(0.0, 255.0) as u8,
                                recon[idx][1].round().clamp(0.0, 255.0) as u8,
                                recon[idx][2].round().clamp(0.0, 255.0) as u8,
                                255,
                            ]),
                        );
                    }
                }
            }
        }
    }

    Ok(PanoramaResult {
        image: canvas,
        reference_index: ref_idx,
        registered_indices,
        gains,
        homographies: out_homographies,
    })
}
