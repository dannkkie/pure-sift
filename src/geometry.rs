#![forbid(unsafe_code)]

use crate::pipeline::{
    FeatureMatch, Homography2D, build_ratio_weighted_cdf, sample_weighted_index,
};
use crate::visualize::invert_matrix_3x3;
use serde::{Deserialize, Serialize};

/// 3x3 Fundamental matrix `F` mapping homogeneous pixel coordinates `(x1, y1, 1)` to
/// `(x2, y2, 1)` via the epipolar constraint `x2^T F x1 = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FundamentalMatrix {
    pub matrix: [[f64; 3]; 3],
}

/// 3x3 Essential matrix `E` in normalized camera coordinates satisfying `x2_n^T E x1_n = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EssentialMatrix {
    pub matrix: [[f64; 3]; 3],
}

/// Pinhole camera intrinsics matrix `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Returns the 3x3 calibration matrix `K`.
    pub fn k_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Returns the inverse calibration matrix `K^-1`, or `None` if focal lengths are non-positive.
    pub fn k_inv_matrix(&self) -> Option<[[f64; 3]; 3]> {
        if !self.fx.is_finite()
            || !self.fy.is_finite()
            || !self.cx.is_finite()
            || !self.cy.is_finite()
            || self.fx.abs() < 1e-12
            || self.fy.abs() < 1e-12
        {
            return None;
        }
        Some([
            [1.0 / self.fx, 0.0, -self.cx / self.fx],
            [0.0, 1.0 / self.fy, -self.cy / self.fy],
            [0.0, 0.0, 1.0],
        ])
    }

    /// Maps pixel coordinates `(x, y)` into normalized image plane coordinates `(x_n, y_n)`.
    #[inline]
    pub fn normalize_point(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.cx) / self.fx, (y - self.cy) / self.fy)
    }
}

/// Recovered relative rigid transform `(R, t)` from camera 1 (`query`) to camera 2 (`target`)
/// up to positive translation scale (`||t|| = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelativePose {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub cheirality_inliers: usize,
}

/// Selected two-view geometric model under GRIC model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSelection {
    Homography,
    Fundamental,
}

/// Torr's Geometric Robust Information Criterion (GRIC) comparison between planar Homography and
/// general epipolar Fundamental matrix models.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GricComparison {
    pub selected: ModelSelection,
    pub homography_gric: f64,
    pub fundamental_gric: f64,
}

/// Options for Fundamental / Essential matrix RANSAC estimation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FundamentalRansacOptions {
    /// Maximum allowed Sampson distance in pixels (`sqrt(d_sampson^2) <= sampson_threshold`).
    pub sampson_threshold: f64,
    /// Hard upper bound on RANSAC iterations.
    pub max_iters: usize,
    /// Target probabilistic confidence in `(0.0, 1.0)`.
    pub confidence: f64,
    /// Deterministic RNG seed.
    pub seed: u64,
}

impl Default for FundamentalRansacOptions {
    fn default() -> Self {
        Self {
            sampson_threshold: 1.5,
            max_iters: 2000,
            confidence: 0.999,
            seed: 0xF00D_CAFE_1234_5678,
        }
    }
}

impl FundamentalRansacOptions {
    /// Sets the Sampson distance threshold in pixels.
    pub fn sampson_threshold(mut self, threshold: f64) -> Self {
        self.sampson_threshold = threshold;
        self
    }

    /// Sets the deterministic RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[inline]
fn mul_mat3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
fn transpose_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn det_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[inline]
fn normalize_frobenius(m: &mut [[f64; 3]; 3]) -> bool {
    let norm_sq: f64 = m.iter().flatten().map(|&v| v * v).sum();
    if !norm_sq.is_finite() || norm_sq < 1e-24 {
        return false;
    }
    let inv = 1.0 / norm_sq.sqrt();
    for row in m.iter_mut() {
        for val in row.iter_mut() {
            *val *= inv;
        }
    }
    true
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize_vec3(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot3(v, v).sqrt();
    if !n.is_finite() || n < 1e-12 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

/// Jacobi eigensolver for a 9x9 real symmetric matrix; returns `(eigenvalues, eigenvectors_cols)`
/// sorted in ascending eigenvalue order.
fn jacobi_symmetric_eigen_9x9(mut a: [[f64; 9]; 9]) -> Option<([f64; 9], [[f64; 9]; 9])> {
    let mut v = [[0.0f64; 9]; 9];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _ in 0..500 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = 0.0f64;
        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                let abs_v = val.abs();
                if abs_v > max_off {
                    max_off = abs_v;
                    p = i;
                    q = j;
                }
            }
        }
        if !max_off.is_finite() {
            return None;
        }
        if max_off < 1e-13 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..9 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                let new_arp = c * arp - s * arq;
                let new_arq = s * arp + c * arq;
                a[r][p] = new_arp;
                a[p][r] = new_arp;
                a[r][q] = new_arq;
                a[q][r] = new_arq;
            }
        }
        for row in &mut v {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = c * vrp - s * vrq;
            row[q] = s * vrp + c * vrq;
        }
    }

    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
    order.sort_by(|&i, &j| {
        a[i][i]
            .partial_cmp(&a[j][j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigvals = [0.0f64; 9];
    let mut eigvecs = [[0.0f64; 9]; 9];
    for (dst_c, &src_c) in order.iter().enumerate() {
        eigvals[dst_c] = a[src_c][src_c];
        for r in 0..9 {
            let val = v[r][src_c];
            if !val.is_finite() {
                return None;
            }
            eigvecs[r][dst_c] = val;
        }
    }
    Some((eigvals, eigvecs))
}

fn jacobi_symmetric_eigen_3x3(mut a: [[f64; 3]; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..100 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = 0.0f64;
        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                if val.abs() > max_off {
                    max_off = val.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if !max_off.is_finite() {
            return None;
        }
        if max_off < 1e-14 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                let new_arp = c * arp - s * arq;
                let new_arq = s * arp + c * arq;
                a[r][p] = new_arp;
                a[p][r] = new_arp;
                a[r][q] = new_arq;
                a[q][r] = new_arq;
            }
        }
        for row in &mut v {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = c * vrp - s * vrq;
            row[q] = s * vrp + c * vrq;
        }
    }

    // Sort descending by eigenvalue
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        a[j][j]
            .partial_cmp(&a[i][i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigvals = [0.0f64; 3];
    let mut eigvecs = [[0.0f64; 3]; 3];
    for (dst_c, &src_c) in order.iter().enumerate() {
        eigvals[dst_c] = a[src_c][src_c];
        for r in 0..3 {
            eigvecs[r][dst_c] = v[r][src_c];
        }
    }
    Some((eigvals, eigvecs))
}

fn jacobi_symmetric_eigen_4x4(mut a: [[f64; 4]; 4]) -> Option<[f64; 4]> {
    let mut v = [[0.0f64; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _ in 0..100 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = 0.0f64;
        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                if val.abs() > max_off {
                    max_off = val.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if !max_off.is_finite() {
            return None;
        }
        if max_off < 1e-13 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..4 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                let new_arp = c * arp - s * arq;
                let new_arq = s * arp + c * arq;
                a[r][p] = new_arp;
                a[p][r] = new_arp;
                a[r][q] = new_arq;
                a[q][r] = new_arq;
            }
        }
        for row in &mut v {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = c * vrp - s * vrq;
            row[q] = s * vrp + c * vrq;
        }
    }

    let mut min_idx = 0usize;
    let mut min_eig = a[0][0];
    for (i, row) in a.iter().enumerate().skip(1) {
        if row[i] < min_eig {
            min_eig = row[i];
            min_idx = i;
        }
    }

    let out = [v[0][min_idx], v[1][min_idx], v[2][min_idx], v[3][min_idx]];
    if out.iter().all(|x| x.is_finite()) {
        Some(out)
    } else {
        None
    }
}

struct Svd3 {
    u: [[f64; 3]; 3],
    s: [f64; 3],
    vt: [[f64; 3]; 3],
}

/// Computes thin/full 3x3 SVD `M = U diag(S) Vt` with `S[0] >= S[1] >= S[2] >= 0`
/// and orthonormal `U`, `Vt`.
fn svd_3x3(m: [[f64; 3]; 3]) -> Option<Svd3> {
    let mt = transpose_3x3(&m);
    let mtm = mul_mat3(&mt, &m);
    let (eigvals, v_mat) = jacobi_symmetric_eigen_3x3(mtm)?;
    let s = [
        eigvals[0].max(0.0).sqrt(),
        eigvals[1].max(0.0).sqrt(),
        eigvals[2].max(0.0).sqrt(),
    ];

    let v0 = [v_mat[0][0], v_mat[1][0], v_mat[2][0]];
    let v1 = [v_mat[0][1], v_mat[1][1], v_mat[2][1]];
    let v2 = [v_mat[0][2], v_mat[1][2], v_mat[2][2]];

    let mul_mv = |vec: [f64; 3]| -> [f64; 3] {
        [
            m[0][0] * vec[0] + m[0][1] * vec[1] + m[0][2] * vec[2],
            m[1][0] * vec[0] + m[1][1] * vec[1] + m[1][2] * vec[2],
            m[2][0] * vec[0] + m[2][1] * vec[1] + m[2][2] * vec[2],
        ]
    };

    let u0 = if s[0] > 1e-12 {
        normalize_vec3(mul_mv(v0)).unwrap_or([1.0, 0.0, 0.0])
    } else {
        [1.0, 0.0, 0.0]
    };

    let u1 = if s[1] > 1e-12 {
        let raw = mul_mv(v1);
        let proj = dot3(raw, u0);
        let orth = [
            raw[0] - proj * u0[0],
            raw[1] - proj * u0[1],
            raw[2] - proj * u0[2],
        ];
        normalize_vec3(orth).unwrap_or_else(|| {
            let basis = if u0[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            normalize_vec3(cross3(u0, basis)).unwrap_or([0.0, 1.0, 0.0])
        })
    } else {
        let basis = if u0[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        normalize_vec3(cross3(u0, basis)).unwrap_or([0.0, 1.0, 0.0])
    };

    let mut u2 = normalize_vec3(cross3(u0, u1))?;
    if s[2] > 1e-12 {
        let mv2 = mul_mv(v2);
        if dot3(u2, mv2) < 0.0 {
            u2 = [-u2[0], -u2[1], -u2[2]];
        }
    }

    let u = [
        [u0[0], u1[0], u2[0]],
        [u0[1], u1[1], u2[1]],
        [u0[2], u1[2], u2[2]],
    ];
    let vt = transpose_3x3(&v_mat);
    Some(Svd3 { u, s, vt })
}

fn rank2_project_f(mut f: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let Svd3 { u, s, vt } = svd_3x3(f)?;
    let diag = [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, 0.0]];
    f = mul_mat3(&u, &mul_mat3(&diag, &vt));
    if !normalize_frobenius(&mut f) {
        return None;
    }
    Some(f)
}

type NormalizedPoints2D = (Vec<(f64, f64)>, [[f64; 3]; 3]);

/// Hartley isotropic point normalization transforming points to mean `(0, 0)` and
/// mean distance `sqrt(2)` from origin.
fn normalize_points_2d(pts: &[(f64, f64)]) -> Option<NormalizedPoints2D> {
    if pts.is_empty() {
        return None;
    }
    let inv_n = 1.0 / (pts.len() as f64);
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for &(x, y) in pts {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        cx += x;
        cy += y;
    }
    cx *= inv_n;
    cy *= inv_n;

    let mut mean_dist = 0.0f64;
    for &(x, y) in pts {
        mean_dist += (x - cx).hypot(y - cy);
    }
    mean_dist *= inv_n;
    if !mean_dist.is_finite() || mean_dist < 1e-9 {
        return None;
    }

    let scale = std::f64::consts::SQRT_2 / mean_dist;
    let norm = pts
        .iter()
        .map(|&(x, y)| ((x - cx) * scale, (y - cy) * scale))
        .collect();
    let t = [
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ];
    Some((norm, t))
}

fn assemble_fundamental_ata(q_norm: &[(f64, f64)], t_norm: &[(f64, f64)]) -> [[f64; 9]; 9] {
    let mut ata = [[0.0f64; 9]; 9];
    for (&(x1, y1), &(x2, y2)) in q_norm.iter().zip(t_norm.iter()) {
        // x2^T F x1 = 0, row-major f = [f00, f01, f02, f10, f11, f12, f20, f21, f22]
        let r = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0];
        for i in 0..9 {
            for j in i..9 {
                ata[i][j] += r[i] * r[j];
            }
        }
    }
    for i in 0..9 {
        for j in 0..i {
            ata[i][j] = ata[j][i];
        }
    }
    ata
}

#[inline]
fn vec9_to_mat3(eigvecs: &[[f64; 9]; 9], col: usize) -> [[f64; 3]; 3] {
    [
        [eigvecs[0][col], eigvecs[1][col], eigvecs[2][col]],
        [eigvecs[3][col], eigvecs[4][col], eigvecs[5][col]],
        [eigvecs[6][col], eigvecs[7][col], eigvecs[8][col]],
    ]
}

/// Computes the squared Sampson approximation to geometric reprojection error
/// for correspondence `p1` (query) <-> `p2` (target) under `F` (`x2^T F x1 = 0`).
#[inline]
pub fn sampson_distance_sq(f: &FundamentalMatrix, p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let m = &f.matrix;
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    let fx1_0 = m[0][0] * x1 + m[0][1] * y1 + m[0][2];
    let fx1_1 = m[1][0] * x1 + m[1][1] * y1 + m[1][2];
    let fx1_2 = m[2][0] * x1 + m[2][1] * y1 + m[2][2];

    let ftx2_0 = m[0][0] * x2 + m[1][0] * y2 + m[2][0];
    let ftx2_1 = m[0][1] * x2 + m[1][1] * y2 + m[2][1];

    let x2_f_x1 = x2 * fx1_0 + y2 * fx1_1 + fx1_2;
    let denom = fx1_0 * fx1_0 + fx1_1 * fx1_1 + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1;
    if denom < 1e-18 || !denom.is_finite() {
        return f64::INFINITY;
    }
    (x2_f_x1 * x2_f_x1) / denom
}

/// Fits a rank-2 Fundamental matrix from `sample` (`len >= 8`) using the normalized 8-point algorithm.
pub fn fit_fundamental_8point(sample: &[FeatureMatch]) -> Option<FundamentalMatrix> {
    if sample.len() < 8 {
        return None;
    }
    let q_pts: Vec<(f64, f64)> = sample
        .iter()
        .map(|m| (m.query_point.0 as f64, m.query_point.1 as f64))
        .collect();
    let t_pts: Vec<(f64, f64)> = sample
        .iter()
        .map(|m| (m.target_point.0 as f64, m.target_point.1 as f64))
        .collect();

    let (q_norm, t_q) = normalize_points_2d(&q_pts)?;
    let (t_norm, t_t) = normalize_points_2d(&t_pts)?;

    let ata = assemble_fundamental_ata(&q_norm, &t_norm);
    let (_, eigvecs) = jacobi_symmetric_eigen_9x9(ata)?;
    let f_raw = vec9_to_mat3(&eigvecs, 0);
    let f_rank2 = rank2_project_f(f_raw)?;

    let mut f = mul_mat3(&transpose_3x3(&t_t), &mul_mat3(&f_rank2, &t_q));
    if !normalize_frobenius(&mut f) {
        return None;
    }
    Some(FundamentalMatrix { matrix: f })
}

/// Solves `c3 * t^3 + c2 * t^2 + c1 * t + c0 = 0` for real roots.
fn solve_cubic_real(c3: f64, c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    if c3.abs() < 1e-12 {
        if c2.abs() < 1e-12 {
            if c1.abs() < 1e-12 {
                return Vec::new();
            }
            return vec![-c0 / c1];
        }
        let disc = c1 * c1 - 4.0 * c2 * c0;
        if disc < 0.0 {
            return Vec::new();
        }
        let sq = disc.sqrt();
        return vec![(-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2)];
    }

    let a = c2 / c3;
    let b = c1 / c3;
    let c = c0 / c3;

    let a_third = a / 3.0;
    let p = b - a * a_third;
    let q = (2.0 * a * a * a) / 27.0 - (a * b) / 3.0 + c;
    let disc = (q * 0.5) * (q * 0.5) + (p / 3.0).powi(3);

    let mut roots = if disc > 1e-14 {
        let sq = disc.sqrt();
        let u = (-0.5 * q + sq).cbrt();
        let v = (-0.5 * q - sq).cbrt();
        vec![u + v - a_third]
    } else if disc >= -1e-14 {
        let u = (-0.5 * q).cbrt();
        vec![2.0 * u - a_third, -u - a_third]
    } else {
        let r = (-p / 3.0).sqrt();
        let cos_phi = ((-0.5 * q) / (r * r * r)).clamp(-1.0, 1.0);
        let phi = cos_phi.acos();
        let two_r = 2.0 * r;
        vec![
            two_r * (phi / 3.0).cos() - a_third,
            two_r * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos() - a_third,
            two_r * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos() - a_third,
        ]
    };

    // Polish each root with two Newton-Raphson steps
    for t in &mut roots {
        for _ in 0..3 {
            let f = ((c3 * *t + c2) * *t + c1) * *t + c0;
            let df = (3.0 * c3 * *t + 2.0 * c2) * *t + c1;
            if df.abs() > 1e-14 {
                *t -= f / df;
            }
        }
    }
    roots
}

/// Fits up to 3 candidate rank-2 Fundamental matrices from exact minimal `7` correspondences
/// by solving the cubic determinant constraint `det(t * F1 + (1 - t) * F2) = 0`.
pub fn fit_fundamental_7point(sample: &[FeatureMatch; 7]) -> Vec<FundamentalMatrix> {
    let q_pts: Vec<(f64, f64)> = sample
        .iter()
        .map(|m| (m.query_point.0 as f64, m.query_point.1 as f64))
        .collect();
    let t_pts: Vec<(f64, f64)> = sample
        .iter()
        .map(|m| (m.target_point.0 as f64, m.target_point.1 as f64))
        .collect();

    let Some((q_norm, t_q)) = normalize_points_2d(&q_pts) else {
        return Vec::new();
    };
    let Some((t_norm, t_t)) = normalize_points_2d(&t_pts) else {
        return Vec::new();
    };

    let ata = assemble_fundamental_ata(&q_norm, &t_norm);
    let Some((_, eigvecs)) = jacobi_symmetric_eigen_9x9(ata) else {
        return Vec::new();
    };

    let f1 = vec9_to_mat3(&eigvecs, 0);
    let f2 = vec9_to_mat3(&eigvecs, 1);

    // Parameterize M(t) = A + t * B with A = f2, B = f1 - f2 (so t=1 -> f1, t=0 -> f2)
    let mut b = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            b[r][c] = f1[r][c] - f2[r][c];
        }
    }

    let eval_mat = |t: f64| -> [[f64; 3]; 3] {
        let mut m = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] = f2[r][c] + t * b[r][c];
            }
        }
        m
    };

    let p0 = det_3x3(&f2);
    let p1 = det_3x3(&eval_mat(1.0));
    let pm1 = det_3x3(&eval_mat(-1.0));
    let p2 = det_3x3(&eval_mat(2.0));

    let c0 = p0;
    let c2 = 0.5 * (p1 + pm1) - c0;
    let s1 = 0.5 * (p1 - pm1);
    let s2 = p2 - c0 - 4.0 * c2;
    let c3 = (s2 - 2.0 * s1) / 6.0;
    let c1 = s1 - c3;

    let roots = solve_cubic_real(c3, c2, c1, c0);
    let t_t_t = transpose_3x3(&t_t);

    let mut out = Vec::with_capacity(3);
    for t in roots {
        if !t.is_finite() {
            continue;
        }
        let fn_cand = eval_mat(t);
        let mut f = mul_mat3(&t_t_t, &mul_mat3(&fn_cand, &t_q));
        if normalize_frobenius(&mut f) {
            out.push(FundamentalMatrix { matrix: f });
        }
    }

    // If `f1 - f2` itself is near-singular (leading cubic term vanished), include f1 - f2
    if c3.abs() < 1e-12 && det_3x3(&b).abs() < 1e-10 {
        let mut f = mul_mat3(&t_t_t, &mul_mat3(&b, &t_q));
        if normalize_frobenius(&mut f) {
            out.push(FundamentalMatrix { matrix: f });
        }
    }

    out
}

fn evaluate_fundamental_hypothesis(
    matches: &[FeatureMatch],
    f: &FundamentalMatrix,
    thresh_sq: f64,
) -> (usize, f64) {
    let mut count = 0usize;
    let mut truncated_err = 0.0f64;
    for m in matches {
        let d2 = sampson_distance_sq(
            f,
            (m.query_point.0 as f64, m.query_point.1 as f64),
            (m.target_point.0 as f64, m.target_point.1 as f64),
        );
        if d2 <= thresh_sq {
            count += 1;
            truncated_err += d2;
        } else {
            truncated_err += thresh_sq;
        }
    }
    (count, truncated_err)
}

fn adaptive_epipolar_iters(
    best_inliers: usize,
    total: usize,
    sample_size: usize,
    confidence: f64,
    min_iters: usize,
    max_iters: usize,
) -> usize {
    if total <= sample_size || best_inliers == 0 {
        return max_iters;
    }
    let w = (best_inliers as f64) / (total as f64);
    if w >= 0.9999 {
        return min_iters;
    }
    let prob_good = w.powi(sample_size as i32);
    if prob_good <= 1e-12 {
        return max_iters;
    }
    let denom = (1.0 - prob_good).max(1e-15).ln();
    let num = (1.0 - confidence.clamp(0.5, 0.999999)).ln();
    ((num / denom).ceil() as usize).clamp(min_iters, max_iters)
}

/// Estimates a Fundamental matrix and inlier set using normalized 8-point LO-RANSAC with
/// Sampson-distance scoring and adaptive termination.
pub fn estimate_fundamental_ransac(
    matches: &[FeatureMatch],
    sampson_threshold: f64,
) -> (Vec<FeatureMatch>, Option<FundamentalMatrix>) {
    estimate_fundamental_ransac_with_options(
        matches,
        FundamentalRansacOptions::default().sampson_threshold(sampson_threshold),
    )
}

/// Estimates a Fundamental matrix using explicit `FundamentalRansacOptions`.
pub fn estimate_fundamental_ransac_with_options(
    matches: &[FeatureMatch],
    options: FundamentalRansacOptions,
) -> (Vec<FeatureMatch>, Option<FundamentalMatrix>) {
    if matches.len() < 7 {
        return (Vec::new(), None);
    }

    let thresh = options.sampson_threshold.max(0.25);
    let thresh_sq = thresh * thresh;

    if matches.len() == 7 {
        let arr: [FeatureMatch; 7] = matches.try_into().unwrap();
        let mut best_f = None;
        let mut best_count = 0usize;
        let mut best_err = f64::INFINITY;
        for cand in fit_fundamental_7point(&arr) {
            let (count, err) = evaluate_fundamental_hypothesis(matches, &cand, thresh_sq);
            if count > best_count || (count == best_count && err < best_err) {
                best_count = count;
                best_err = err;
                best_f = Some(cand);
            }
        }
        let Some(f) = best_f else {
            return (Vec::new(), None);
        };
        let inliers = matches
            .iter()
            .copied()
            .filter(|m| {
                sampson_distance_sq(
                    &f,
                    (m.query_point.0 as f64, m.query_point.1 as f64),
                    (m.target_point.0 as f64, m.target_point.1 as f64),
                ) <= thresh_sq
            })
            .collect();
        return (inliers, Some(f));
    }

    let cdf = build_ratio_weighted_cdf(matches);
    let mut rng_state = options.seed;
    let hard_cap = options.max_iters.max(64);
    let min_iters = 64usize.min(hard_cap);
    let mut max_iters = hard_cap;

    let mut best_f: Option<FundamentalMatrix> = None;
    let mut best_count = 0usize;
    let mut best_err = f64::INFINITY;

    let mut iter = 0usize;
    while iter < max_iters {
        iter += 1;
        let mut idx = [0usize; 8];
        let mut chosen = 0usize;
        let mut attempts = 0usize;
        while chosen < 8 && attempts < 32 {
            attempts += 1;
            let cand = sample_weighted_index(&cdf, &mut rng_state);
            if !idx[..chosen].contains(&cand) {
                idx[chosen] = cand;
                chosen += 1;
            }
        }
        if chosen < 8 {
            continue;
        }

        let sample = [
            matches[idx[0]],
            matches[idx[1]],
            matches[idx[2]],
            matches[idx[3]],
            matches[idx[4]],
            matches[idx[5]],
            matches[idx[6]],
            matches[idx[7]],
        ];

        if let Some(f_cand) = fit_fundamental_8point(&sample) {
            let (count, err) = evaluate_fundamental_hypothesis(matches, &f_cand, thresh_sq);
            if count > best_count || (count == best_count && err < best_err) {
                best_count = count;
                best_err = err;
                best_f = Some(f_cand);
                max_iters = adaptive_epipolar_iters(
                    best_count,
                    matches.len(),
                    8,
                    options.confidence,
                    min_iters,
                    hard_cap,
                );
            }
        }
    }

    let Some(mut final_f) = best_f else {
        return (Vec::new(), None);
    };

    // LO-RANSAC refit on all consensus inliers via normalized 8-point solver
    for _ in 0..2 {
        let inliers: Vec<FeatureMatch> = matches
            .iter()
            .copied()
            .filter(|m| {
                sampson_distance_sq(
                    &final_f,
                    (m.query_point.0 as f64, m.query_point.1 as f64),
                    (m.target_point.0 as f64, m.target_point.1 as f64),
                ) <= thresh_sq
            })
            .collect();
        if inliers.len() < 8 {
            break;
        }
        if let Some(refined) = fit_fundamental_8point(&inliers) {
            let (ref_count, ref_err) =
                evaluate_fundamental_hypothesis(matches, &refined, thresh_sq);
            if ref_count > best_count || (ref_count == best_count && ref_err < best_err) {
                best_count = ref_count;
                best_err = ref_err;
                final_f = refined;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let inliers = matches
        .iter()
        .copied()
        .filter(|m| {
            sampson_distance_sq(
                &final_f,
                (m.query_point.0 as f64, m.query_point.1 as f64),
                (m.target_point.0 as f64, m.target_point.1 as f64),
            ) <= thresh_sq
        })
        .collect();

    (inliers, Some(final_f))
}

/// Computes the Essential matrix `E = K2^T F K1` and projects it onto the Essential manifold
/// (`U diag((s0 + s1)/2, (s0 + s1)/2, 0) Vt`).
pub fn essential_from_fundamental(
    f: &FundamentalMatrix,
    k1: &CameraIntrinsics,
    k2: &CameraIntrinsics,
) -> Option<EssentialMatrix> {
    let k1_m = k1.k_matrix();
    let k2_t = transpose_3x3(&k2.k_matrix());
    let e_raw = mul_mat3(&k2_t, &mul_mat3(&f.matrix, &k1_m));
    let Svd3 { u, s, vt } = svd_3x3(e_raw)?;
    let avg = 0.5 * (s[0] + s[1]);
    if !avg.is_finite() || avg < 1e-18 {
        return None;
    }
    let diag = [[avg, 0.0, 0.0], [0.0, avg, 0.0], [0.0, 0.0, 0.0]];
    let mut e = mul_mat3(&u, &mul_mat3(&diag, &vt));
    if !normalize_frobenius(&mut e) {
        return None;
    }
    Some(EssentialMatrix { matrix: e })
}

/// Estimates an Essential matrix `E` from pixel correspondences and known intrinsics
/// `k1`, `k2`, returning calibrated Fundamental inliers projected onto the Essential manifold.
pub fn estimate_essential_ransac(
    matches: &[FeatureMatch],
    k1: &CameraIntrinsics,
    k2: &CameraIntrinsics,
    options: FundamentalRansacOptions,
) -> (Vec<FeatureMatch>, Option<EssentialMatrix>) {
    let k1_inv = match k1.k_inv_matrix() {
        Some(m) => m,
        None => return (Vec::new(), None),
    };
    let k2_inv_t = match k2.k_inv_matrix() {
        Some(m) => transpose_3x3(&m),
        None => return (Vec::new(), None),
    };

    let (inliers, f_opt) = estimate_fundamental_ransac_with_options(matches, options);
    let Some(f) = f_opt else {
        return (inliers, None);
    };
    let Some(e) = essential_from_fundamental(&f, k1, k2) else {
        return (inliers, None);
    };

    // Re-score inliers using the projected Essential matrix lifted back to pixel space F_e = K2^-T E K1^-1
    let mut f_proj = mul_mat3(&k2_inv_t, &mul_mat3(&e.matrix, &k1_inv));
    if !normalize_frobenius(&mut f_proj) {
        return (inliers, Some(e));
    }
    let f_e = FundamentalMatrix { matrix: f_proj };
    let thresh = options.sampson_threshold.max(0.25);
    let thresh_sq = thresh * thresh;
    let filtered_inliers: Vec<FeatureMatch> = matches
        .iter()
        .copied()
        .filter(|m| {
            sampson_distance_sq(
                &f_e,
                (m.query_point.0 as f64, m.query_point.1 as f64),
                (m.target_point.0 as f64, m.target_point.1 as f64),
            ) <= thresh_sq
        })
        .collect();

    if filtered_inliers.len() >= 8 {
        (filtered_inliers, Some(e))
    } else {
        (inliers, Some(e))
    }
}

/// Linear DLT triangulation of a single 3D point from two 3x4 camera projection matrices
/// `p1`, `p2` and normalized/pixel 2D coordinates `x1`, `x2`.
pub fn triangulate_point_linear(
    p1: &[[f64; 4]; 3],
    p2: &[[f64; 4]; 3],
    x1: (f64, f64),
    x2: (f64, f64),
) -> Option<[f64; 3]> {
    let mut rows = [[0.0f64; 4]; 4];
    for j in 0..4 {
        rows[0][j] = x1.0 * p1[2][j] - p1[0][j];
        rows[1][j] = x1.1 * p1[2][j] - p1[1][j];
        rows[2][j] = x2.0 * p2[2][j] - p2[0][j];
        rows[3][j] = x2.1 * p2[2][j] - p2[1][j];
    }

    let mut ata = [[0.0f64; 4]; 4];
    for row in &rows {
        for i in 0..4 {
            for j in i..4 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..4 {
        for j in 0..i {
            ata[i][j] = ata[j][i];
        }
    }

    let x_hom = jacobi_symmetric_eigen_4x4(ata)?;
    if !x_hom[3].is_finite() || x_hom[3].abs() < 1e-12 {
        return None;
    }
    let inv_w = 1.0 / x_hom[3];
    let pt = [x_hom[0] * inv_w, x_hom[1] * inv_w, x_hom[2] * inv_w];
    if pt.iter().all(|v| v.is_finite()) {
        Some(pt)
    } else {
        None
    }
}

/// Decomposes an Essential matrix `E` into the physically valid relative camera pose `(R, t)`
/// by testing all four Hartley & Zisserman configurations with linear triangulation and cheirality counting.
pub fn recover_pose(
    e: &EssentialMatrix,
    inliers: &[FeatureMatch],
    k1: &CameraIntrinsics,
    k2: &CameraIntrinsics,
) -> Option<RelativePose> {
    if inliers.is_empty() {
        return None;
    }
    let Svd3 {
        mut u,
        s: _,
        mut vt,
    } = svd_3x3(e.matrix)?;

    if det_3x3(&u) < 0.0 {
        for row in &mut u {
            row[2] = -row[2];
        }
    }
    if det_3x3(&vt) < 0.0 {
        for v in &mut vt[2] {
            *v = -*v;
        }
    }

    let w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let wt = transpose_3x3(&w);

    let r1 = mul_mat3(&u, &mul_mat3(&w, &vt));
    let r2 = mul_mat3(&u, &mul_mat3(&wt, &vt));
    let t_pos = normalize_vec3([u[0][2], u[1][2], u[2][2]])?;
    let t_neg = [-t_pos[0], -t_pos[1], -t_pos[2]];

    let p1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    let norm_pairs: Vec<((f64, f64), (f64, f64))> = inliers
        .iter()
        .map(|m| {
            (
                k1.normalize_point(m.query_point.0 as f64, m.query_point.1 as f64),
                k2.normalize_point(m.target_point.0 as f64, m.target_point.1 as f64),
            )
        })
        .collect();

    let candidates = [(r1, t_pos), (r1, t_neg), (r2, t_pos), (r2, t_neg)];
    let mut best_pose: Option<RelativePose> = None;

    for &(r, t) in &candidates {
        let p2 = [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
        ];

        let mut positive_depth = 0usize;
        for &(xn1, xn2) in &norm_pairs {
            if let Some(pt) = triangulate_point_linear(&p1, &p2, xn1, xn2) {
                let z1 = pt[2];
                let z2 = r[2][0] * pt[0] + r[2][1] * pt[1] + r[2][2] * pt[2] + t[2];
                if z1 > 1e-6 && z2 > 1e-6 && z1.is_finite() && z2.is_finite() {
                    positive_depth += 1;
                }
            }
        }

        if best_pose
            .as_ref()
            .is_none_or(|b| positive_depth > b.cheirality_inliers)
        {
            best_pose = Some(RelativePose {
                rotation: r,
                translation: t,
                cheirality_inliers: positive_depth,
            });
        }
    }

    best_pose.filter(|p| p.cheirality_inliers > 0)
}

/// Computes the geodesic angle error in degrees between two 3x3 rotation matrices.
pub fn rotation_error_deg(r_est: &[[f64; 3]; 3], r_true: &[[f64; 3]; 3]) -> f64 {
    let diff = mul_mat3(&transpose_3x3(r_true), r_est);
    let trace = diff[0][0] + diff[1][1] + diff[2][2];
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

/// Compares planar `Homography2D` and general epipolar `FundamentalMatrix` using Torr's
/// Geometric Robust Information Criterion (GRIC), preferring `Homography` for planar scenes or
/// degenerate pure rotation and `Fundamental` for general 3D parallax scenes.
pub fn select_model_gric(
    matches: &[FeatureMatch],
    homography: Option<&Homography2D>,
    fundamental: Option<&FundamentalMatrix>,
    sigma: f64,
) -> Option<GricComparison> {
    if matches.is_empty() {
        return None;
    }

    let sigma_sq = sigma.max(0.25).powi(2);
    let r = 4.0f64;
    let lambda1 = 2.0f64;
    let lambda2 = 2.0f64;

    let h_gric = if let Some(h) = homography {
        let h_f64 = [
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
        ];
        let h_inv = invert_matrix_3x3(&h.matrix);
        let d_h = 2.0f64;
        let k_h = 8.0f64;
        let cap_h = lambda1 * (r - d_h);

        let data: f64 = matches
            .iter()
            .map(|m| {
                let (qx, qy) = (m.query_point.0 as f64, m.query_point.1 as f64);
                let (tx, ty) = (m.target_point.0 as f64, m.target_point.1 as f64);
                let fw = h_f64[2][0] * qx + h_f64[2][1] * qy + h_f64[2][2];
                let fwd_sq = if fw.abs() > 1e-12 {
                    let px = (h_f64[0][0] * qx + h_f64[0][1] * qy + h_f64[0][2]) / fw;
                    let py = (h_f64[1][0] * qx + h_f64[1][1] * qy + h_f64[1][2]) / fw;
                    (px - tx).powi(2) + (py - ty).powi(2)
                } else {
                    f64::INFINITY
                };
                let bwd_sq = if let Some(ref inv) = h_inv {
                    let bw = inv[2][0] * tx + inv[2][1] * ty + inv[2][2];
                    if bw.abs() > 1e-12 {
                        let px = (inv[0][0] * tx + inv[0][1] * ty + inv[0][2]) / bw;
                        let py = (inv[1][0] * tx + inv[1][1] * ty + inv[1][2]) / bw;
                        (px - qx).powi(2) + (py - qy).powi(2)
                    } else {
                        f64::INFINITY
                    }
                } else {
                    fwd_sq
                };
                let sym_sq = 0.5 * (fwd_sq + bwd_sq);
                (sym_sq / sigma_sq).min(cap_h)
            })
            .sum();
        data + lambda1 * d_h * (matches.len() as f64) + lambda2 * k_h
    } else {
        f64::INFINITY
    };

    let f_gric = if let Some(f) = fundamental {
        let d_f = 3.0f64;
        let k_f = 7.0f64;
        let cap_f = lambda1 * (r - d_f);

        let data: f64 = matches
            .iter()
            .map(|m| {
                let d2 = sampson_distance_sq(
                    f,
                    (m.query_point.0 as f64, m.query_point.1 as f64),
                    (m.target_point.0 as f64, m.target_point.1 as f64),
                );
                (d2 / sigma_sq).min(cap_f)
            })
            .sum();
        data + lambda1 * d_f * (matches.len() as f64) + lambda2 * k_f
    } else {
        f64::INFINITY
    };

    if !h_gric.is_finite() && !f_gric.is_finite() {
        return None;
    }

    let selected = if f_gric < h_gric {
        ModelSelection::Fundamental
    } else {
        ModelSelection::Homography
    };

    Some(GricComparison {
        selected,
        homography_gric: h_gric,
        fundamental_gric: f_gric,
    })
}
