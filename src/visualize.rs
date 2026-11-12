#![forbid(unsafe_code)]

use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};

use crate::pipeline::{FeatureMatch, Homography2D, SiftFeature, SimilarityTransform2D};

/// Maximum output canvas dimension (width or height) for homography warping to prevent OOM.
pub const MAX_WARP_DIM: usize = 8192;
/// Maximum output pixel count for homography warping.
pub const MAX_WARP_PIXELS: usize = 25_000_000;

/// Blends an RGBA foreground color onto an existing pixel in `img` if `(x, y)` is within bounds.
#[inline]
pub fn blend_pixel(img: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x < 0 || y < 0 {
        return;
    }
    let ux = x as u32;
    let uy = y as u32;
    if ux >= img.width() || uy >= img.height() {
        return;
    }

    let alpha = (color[3] as f32) / 255.0;
    if alpha >= 0.999 {
        img.put_pixel(ux, uy, Rgba([color[0], color[1], color[2], 255]));
        return;
    }

    let dst = img.get_pixel(ux, uy);
    let inv_alpha = 1.0 - alpha;
    let r = ((color[0] as f32) * alpha + (dst[0] as f32) * inv_alpha).round() as u8;
    let g = ((color[1] as f32) * alpha + (dst[1] as f32) * inv_alpha).round() as u8;
    let b = ((color[2] as f32) * alpha + (dst[2] as f32) * inv_alpha).round() as u8;
    let a = dst[3].max(color[3]);
    img.put_pixel(ux, uy, Rgba([r, g, b, a]));
}

/// Draws a 1px (or thick via 3x3 cross) Bresenham line between `(x0, y0)` and `(x1, y1)`.
fn draw_line(
    img: &mut RgbaImage,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Rgba<u8>,
    thick: bool,
) {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        blend_pixel(img, x, y, color);
        if thick {
            blend_pixel(img, x + 1, y, color);
            blend_pixel(img, x, y + 1, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}

/// Draws a midpoint circle outline centered at `(cx, cy)` with `radius`.
fn draw_circle(img: &mut RgbaImage, cx: i32, cy: i32, radius: i32, color: Rgba<u8>) {
    let r = radius.max(1);
    let mut x = r;
    let mut y = 0;
    let mut err = 1 - r;

    while x >= y {
        blend_pixel(img, cx + x, cy + y, color);
        blend_pixel(img, cx + y, cy + x, color);
        blend_pixel(img, cx - y, cy + x, color);
        blend_pixel(img, cx - x, cy + y, color);
        blend_pixel(img, cx - x, cy - y, color);
        blend_pixel(img, cx - y, cy - x, color);
        blend_pixel(img, cx + y, cy - x, color);
        blend_pixel(img, cx + x, cy - y, color);

        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Draws a small filled 3x3 marker centered at `(cx, cy)`.
fn draw_marker(img: &mut RgbaImage, cx: i32, cy: i32, color: Rgba<u8>) {
    for dy in -1..=1 {
        for dx in -1..=1 {
            blend_pixel(img, cx + dx, cy + dy, color);
        }
    }
}

/// Renders detected SIFT features onto a copy of `img` with scale-proportional circles
/// and radial orientation ticks.
///
/// Positive DoG responses (bright blobs) are drawn in cyan (`#00E5FF`) and negative responses
/// (dark blobs) in amber-orange (`#FF9100`).
pub fn draw_keypoints_overlay(img: &DynamicImage, features: &[SiftFeature]) -> RgbaImage {
    let mut canvas = img.to_rgba8();

    for feat in features {
        let cx = feat.x.round() as i32;
        let cy = feat.y.round() as i32;
        let radius = (feat.scale * 3.0).round().clamp(3.0, 28.0) as i32;

        let color = if feat.response >= 0.0 {
            Rgba([0, 229, 255, 220])
        } else {
            Rgba([255, 145, 0, 220])
        };

        draw_circle(&mut canvas, cx, cy, radius, color);
        draw_marker(&mut canvas, cx, cy, color);

        let tip_x = (feat.x + (radius as f32) * feat.orientation.cos()).round() as i32;
        let tip_y = (feat.y + (radius as f32) * feat.orientation.sin()).round() as i32;
        draw_line(&mut canvas, cx, cy, tip_x, tip_y, color, false);
    }

    canvas
}

/// Renders `query` (left) and `target` (right) side by side with green lines connecting
/// matched RANSAC inlier correspondences.
pub fn draw_matches_side_by_side(
    query: &DynamicImage,
    target: &DynamicImage,
    matches: &[FeatureMatch],
) -> RgbaImage {
    let q_rgba = query.to_rgba8();
    let t_rgba = target.to_rgba8();

    let q_w = q_rgba.width();
    let q_h = q_rgba.height();
    let t_w = t_rgba.width();
    let t_h = t_rgba.height();

    let total_w = q_w.saturating_add(t_w);
    let total_h = q_h.max(t_h);

    let mut canvas = RgbaImage::from_pixel(total_w, total_h, Rgba([18, 20, 26, 255]));

    for y in 0..q_h {
        for x in 0..q_w {
            canvas.put_pixel(x, y, *q_rgba.get_pixel(x, y));
        }
    }
    for y in 0..t_h {
        for x in 0..t_w {
            canvas.put_pixel(q_w + x, y, *t_rgba.get_pixel(x, y));
        }
    }

    let line_color = Rgba([34, 211, 102, 185]);
    let marker_color = Rgba([255, 235, 59, 235]);
    let x_offset = q_w as i32;

    for m in matches {
        let qx = m.query_point.0.round() as i32;
        let qy = m.query_point.1.round() as i32;
        let tx = x_offset + (m.target_point.0.round() as i32);
        let ty = m.target_point.1.round() as i32;

        draw_line(&mut canvas, qx, qy, tx, ty, line_color, false);
        draw_marker(&mut canvas, qx, qy, marker_color);
        draw_marker(&mut canvas, tx, ty, marker_color);
    }

    canvas
}

/// Computes the exact 3x3 inverse of a projective homography matrix in `f64`.
pub fn invert_matrix_3x3(m: &[[f32; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = [
        [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64],
        [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64],
        [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64],
    ];

    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]);
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];

    let c10 = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]);
    let c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let c12 = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]);

    let c20 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    let c21 = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]);
    let c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

#[inline]
pub fn project_f64(m: &[[f64; 3]; 3], x: f64, y: f64) -> Option<(f64, f64)> {
    let w = m[2][0] * x + m[2][1] * y + m[2][2];
    if w.abs() < 1e-7 {
        return None;
    }
    let inv_w = 1.0 / w;
    Some((
        (m[0][0] * x + m[0][1] * y + m[0][2]) * inv_w,
        (m[1][0] * x + m[1][1] * y + m[1][2]) * inv_w,
    ))
}

/// Bilinear pixel sampler on an `RgbaImage` returning `(r, g, b, feather_weight)`.
#[inline]
pub fn sample_bilinear_rgba(img: &RgbaImage, x: f64, y: f64) -> Option<([f64; 4], f64)> {
    let w = img.width() as f64;
    let h = img.height() as f64;
    if x < 0.0 || y < 0.0 || x > (w - 1.0) || y > (h - 1.0) {
        return None;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(img.width() - 1);
    let y1 = (y0 + 1).min(img.height() - 1);

    let fx = x - (x0 as f64);
    let fy = y - (y0 as f64);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let p00 = img.get_pixel(x0, y0).0;
    let p10 = img.get_pixel(x1, y0).0;
    let p01 = img.get_pixel(x0, y1).0;
    let p11 = img.get_pixel(x1, y1).0;

    let mut rgba = [0.0f64; 4];
    for c in 0..4 {
        rgba[c] = (p00[c] as f64) * w00
            + (p10[c] as f64) * w10
            + (p01[c] as f64) * w01
            + (p11[c] as f64) * w11;
    }

    // Smooth edge-distance feather weight (0 at border, 1 deep inside image)
    let dist_edge = x.min((w - 1.0) - x).min(y).min((h - 1.0) - y);
    let feather = ((dist_edge + 1.0) / 32.0).clamp(0.02, 1.0);
    Some((rgba, feather))
}

/// Promotes an optional `Homography2D` or `SimilarityTransform2D` into a 3x3 matrix.
pub fn resolve_alignment_homography(
    homography: Option<Homography2D>,
    similarity: Option<SimilarityTransform2D>,
) -> Option<Homography2D> {
    homography.or_else(|| {
        similarity.map(|s| Homography2D {
            matrix: [[s.a, -s.b, s.tx], [s.b, s.a, s.ty], [0.0, 0.0, 1.0]],
        })
    })
}

/// Checks whether a candidate 3x3 `Homography2D` maps a `src_w x src_h` rectangle into a
/// non-degenerate, horizon-free convex quadrilateral with reasonable area scaling.
pub fn is_sane_homography_for_dims(h: &Homography2D, src_w: usize, src_h: usize) -> bool {
    if src_w < 2 || src_h < 2 {
        return false;
    }
    let m = &h.matrix;
    let w = (src_w - 1) as f64;
    let ht = (src_h - 1) as f64;
    let corners = [(0.0, 0.0), (w, 0.0), (w, ht), (0.0, ht)];

    let wc = (m[2][0] as f64) * (w * 0.5) + (m[2][1] as f64) * (ht * 0.5) + (m[2][2] as f64);
    if !wc.is_finite() || wc.abs() < 1e-6 {
        return false;
    }

    let mut projected = [(0.0f64, 0.0f64); 4];
    for (idx, &(cx, cy)) in corners.iter().enumerate() {
        let wi = (m[2][0] as f64) * cx + (m[2][1] as f64) * cy + (m[2][2] as f64);
        // Horizon-crossing guard: all corners must have the same denominator sign and stay bounded
        if !wi.is_finite() || wi * wc <= 0.0 || wi.abs() < 0.1 * wc.abs() {
            return false;
        }
        let px = ((m[0][0] as f64) * cx + (m[0][1] as f64) * cy + (m[0][2] as f64)) / wi;
        let py = ((m[1][0] as f64) * cx + (m[1][1] as f64) * cy + (m[1][2] as f64)) / wi;
        if !px.is_finite() || !py.is_finite() {
            return false;
        }
        projected[idx] = (px, py);
    }

    // Verify strict convexity + orientation consistency via edge cross products
    let mut sign_pos = 0usize;
    let mut sign_neg = 0usize;
    let mut shoelace = 0.0f64;
    for i in 0..4 {
        let a = projected[i];
        let b = projected[(i + 1) % 4];
        let c = projected[(i + 2) % 4];
        shoelace += a.0 * b.1 - b.0 * a.1;
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        if cross > 1e-3 {
            sign_pos += 1;
        } else if cross < -1e-3 {
            sign_neg += 1;
        } else {
            return false;
        }
    }
    if !(sign_pos == 4 || sign_neg == 4) {
        return false;
    }

    let quad_area = 0.5 * shoelace.abs();
    let src_area = w * ht;
    quad_area >= 0.04 * src_area && quad_area <= 25.0 * src_area
}

/// Selects a well-conditioned alignment homography, falling back to `SimilarityTransform2D`
/// if `Homography2D` is geometrically degenerate or unsupported by sufficient inliers.
pub fn resolve_safe_alignment_homography(
    homography: Option<Homography2D>,
    hom_inliers: usize,
    similarity: Option<SimilarityTransform2D>,
    src_w: usize,
    src_h: usize,
) -> Option<Homography2D> {
    if let Some(h) = homography
        && hom_inliers >= 6
        && is_sane_homography_for_dims(&h, src_w, src_h)
    {
        return Some(h);
    }
    resolve_alignment_homography(None, similarity)
}

/// Warps `query` into `target`'s coordinate frame using `homography` via bilinear inverse mapping
/// and feather-blends overlapping regions into a composite `RgbaImage`.
pub fn warp_and_stitch(
    query: &DynamicImage,
    target: &DynamicImage,
    homography: &Homography2D,
) -> Result<RgbaImage, String> {
    let q_rgba = query.to_rgba8();
    let t_rgba = target.to_rgba8();

    let qw = q_rgba.width() as f64;
    let qh = q_rgba.height() as f64;
    let tw = t_rgba.width() as f64;
    let th = t_rgba.height() as f64;

    let h_inv = invert_matrix_3x3(&homography.matrix)
        .ok_or_else(|| "Homography matrix is singular and cannot be inverted".to_string())?;

    // Compute union bounding box in target coordinates
    let mut min_x = 0.0f64;
    let mut min_y = 0.0f64;
    let mut max_x = (tw - 1.0).max(0.0);
    let mut max_y = (th - 1.0).max(0.0);

    let corners = [
        (0.0, 0.0),
        (qw - 1.0, 0.0),
        (0.0, qh - 1.0),
        (qw - 1.0, qh - 1.0),
    ];
    for &(cx, cy) in &corners {
        let Some((px, py)) = homography.project(cx as f32, cy as f32) else {
            return Err("Projected query corner hit infinity under homography".to_string());
        };
        let pxf = px as f64;
        let pyf = py as f64;
        if !pxf.is_finite() || !pyf.is_finite() {
            return Err("Projected query corner produced non-finite coordinate".to_string());
        }
        min_x = min_x.min(pxf);
        min_y = min_y.min(pyf);
        max_x = max_x.max(pxf);
        max_y = max_y.max(pyf);
    }

    let origin_x = min_x.floor();
    let origin_y = min_y.floor();
    let out_w_f = (max_x.ceil() - origin_x + 1.0).max(1.0);
    let out_h_f = (max_y.ceil() - origin_y + 1.0).max(1.0);

    if out_w_f > (MAX_WARP_DIM as f64)
        || out_h_f > (MAX_WARP_DIM as f64)
        || (out_w_f * out_h_f) > (MAX_WARP_PIXELS as f64)
    {
        return Err(format!(
            "Stitched canvas dimensions {}x{} exceed safety limits ({}x{}, {} px)",
            out_w_f as usize, out_h_f as usize, MAX_WARP_DIM, MAX_WARP_DIM, MAX_WARP_PIXELS
        ));
    }

    let out_w = out_w_f as u32;
    let out_h = out_h_f as u32;
    let mut canvas = RgbaImage::from_pixel(out_w, out_h, Rgba([0, 0, 0, 0]));

    for oy in 0..out_h {
        let ty = (oy as f64) + origin_y;
        for ox in 0..out_w {
            let tx = (ox as f64) + origin_x;

            let target_sample = sample_bilinear_rgba(&t_rgba, tx, ty);
            let query_sample = project_f64(&h_inv, tx, ty)
                .and_then(|(qx, qy)| sample_bilinear_rgba(&q_rgba, qx, qy));

            let out_px = match (target_sample, query_sample) {
                (Some((t_c, t_w)), Some((q_c, q_w))) => {
                    let denom = t_w + q_w;
                    let wt = t_w / denom;
                    let wq = q_w / denom;
                    Rgba([
                        (t_c[0] * wt + q_c[0] * wq).round() as u8,
                        (t_c[1] * wt + q_c[1] * wq).round() as u8,
                        (t_c[2] * wt + q_c[2] * wq).round() as u8,
                        (t_c[3] * wt + q_c[3] * wq).round().clamp(0.0, 255.0) as u8,
                    ])
                }
                (Some((t_c, _)), None) => Rgba([
                    t_c[0].round() as u8,
                    t_c[1].round() as u8,
                    t_c[2].round() as u8,
                    t_c[3].round() as u8,
                ]),
                (None, Some((q_c, _))) => Rgba([
                    q_c[0].round() as u8,
                    q_c[1].round() as u8,
                    q_c[2].round() as u8,
                    q_c[3].round() as u8,
                ]),
                (None, None) => continue,
            };

            canvas.put_pixel(ox, oy, out_px);
        }
    }

    Ok(canvas)
}

/// Encodes an `RgbaImage` as PNG bytes in memory.
pub fn encode_rgba_png(img: &RgbaImage) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let dyn_img = DynamicImage::ImageRgba8(img.clone());
    dyn_img
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|e| format!("Failed to encode PNG output: {e}"))?;
    Ok(buf)
}
