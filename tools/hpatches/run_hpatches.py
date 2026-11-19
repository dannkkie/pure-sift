#!/usr/bin/env python3
"""
HPatches benchmark harness comparing `pure-sift` against OpenCV SIFT.

Reports per-split (viewpoint, illumination, overall) metrics:
  - Repeatability @ 3 px
  - Matching Score @ 3 px
  - Mean Matching Accuracy (MMA) @ 1 px / 3 px / 5 px
  - Homography AUC @ 3 px / 5 px / 10 px
  - Mean per-pair runtime (ms)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional when running --self-test without numpy
    np = None

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional when OpenCV is not installed
    cv2 = None

HPATCHES_URL = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"


def default_pure_sift_bin(repo_root: Path) -> Path:
    exe = "pure-sift.exe" if os.name == "nt" else "pure-sift"
    return repo_root / "target" / "release" / exe


def download_hpatches(cache_dir: Path) -> Path:
    seq_root = cache_dir / "hpatches-sequences-release"
    if seq_root.exists() and any(seq_root.iterdir()):
        print(f"Using cached HPatches sequences in {seq_root}")
        return seq_root

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "hpatches-sequences-release.tar.gz"
    if not archive.exists():
        print(f"Downloading HPatches (~1.5 GB) from {HPATCHES_URL} ...")
        urllib.request.urlretrieve(HPATCHES_URL, archive)

    print(f"Extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(cache_dir)
    return seq_root


def mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]
            for c in range(3)
        ]
        for r in range(3)
    ]


def mat3_inv(m: list[list[float]]) -> list[list[float]]:
    c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0])
    c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0]

    c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1])
    c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0]
    c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0])

    c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0])
    c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0]

    det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02
    if abs(det) < 1e-12:
        raise ValueError("Singular 3x3 homography matrix")
    inv_det = 1.0 / det
    return [
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ]


def project_pt(h: list[list[float]], x: float, y: float) -> tuple[float, float] | None:
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-9:
        return None
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )


def parse_hpatches_matrix(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append([float(tok) for tok in line.split()])
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"Expected 3x3 matrix in {path}, got {rows}")
    return rows


def write_hpatches_matrix(path: Path, m: list[list[float]]) -> None:
    lines = [" ".join(f"{v:.12g}" for v in row) for row in m]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_synthetic_hpatches(base_img_path: Path, out_dir: Path) -> Path:
    """Generates deterministic viewpoint (v_*) and illumination (i_*) sequences with exact H_1_k."""
    if not base_img_path.exists():
        raise FileNotFoundError(f"Sample image missing at {base_img_path}")

    with Image.open(base_img_path) as im:
        src = im.convert("RGB")
    w, h = src.size
    cx, cy = w * 0.5, h * 0.5

    # Viewpoint sequence: rotated + scaled + slight perspective warps
    v_dir = out_dir / "v_synthetic_aerial"
    v_dir.mkdir(parents=True, exist_ok=True)
    src.save(v_dir / "1.ppm")

    view_params = [
        (6.0, 0.94, 10.0, -6.0, 8e-5),
        (-9.0, 0.88, -12.0, 8.0, -7e-5),
        (14.0, 1.05, 6.0, 10.0, 9e-5),
    ]
    for idx, (deg, scale, tx, ty, px_tilt) in enumerate(view_params, start=2):
        rad = math.radians(deg)
        c, s = math.cos(rad) * scale, math.sin(rad) * scale
        t_to_center = [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]
        t_from_center = [[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]]
        core = [[c, -s, 0.0], [s, c, 0.0], [px_tilt, -0.5 * px_tilt, 1.0]]
        h_1_k = mat3_mul(t_from_center, mat3_mul(core, t_to_center))
        h_inv = mat3_inv(h_1_k)
        s_norm = 1.0 / h_inv[2][2]
        coeffs = (
            h_inv[0][0] * s_norm,
            h_inv[0][1] * s_norm,
            h_inv[0][2] * s_norm,
            h_inv[1][0] * s_norm,
            h_inv[1][1] * s_norm,
            h_inv[1][2] * s_norm,
            h_inv[2][0] * s_norm,
            h_inv[2][1] * s_norm,
        )
        warped = src.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BILINEAR)
        warped.save(v_dir / f"{idx}.ppm")
        write_hpatches_matrix(v_dir / f"H_1_{idx}", h_1_k)

    # Illumination sequence: exposure & contrast changes under identity geometry
    i_dir = out_dir / "i_synthetic_aerial"
    i_dir.mkdir(parents=True, exist_ok=True)
    src.save(i_dir / "1.ppm")
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for idx, (bright, contrast) in enumerate([(0.72, 1.05), (1.28, 0.95), (0.85, 1.22)], start=2):
        adj = ImageEnhance.Brightness(src).enhance(bright)
        adj = ImageEnhance.Contrast(adj).enhance(contrast)
        adj.save(i_dir / f"{idx}.ppm")
        write_hpatches_matrix(i_dir / f"H_1_{idx}", identity)

    return out_dir


def compute_repeatability_and_matching(
    kp1: list[tuple[float, float]],
    kp2: list[tuple[float, float]],
    matches: list[tuple[tuple[float, float], tuple[float, float]]],
    h_1_2: list[list[float]],
    img1_size: tuple[int, int],
    img2_size: tuple[int, int],
) -> tuple[float, float, float, float, float]:
    """Computes (repeatability@3px, matching_score@3px, mma@1, mma@3, mma@5)."""
    w1, h1 = img1_size
    w2, h2 = img2_size

    kp1_in_overlap: list[tuple[float, float]] = []
    for x1, y1 in kp1:
        p2 = project_pt(h_1_2, x1, y1)
        if p2 is not None and 0.0 <= p2[0] < w2 and 0.0 <= p2[1] < h2:
            kp1_in_overlap.append(p2)

    h_2_1 = mat3_inv(h_1_2)
    kp2_in_overlap: list[tuple[float, float]] = []
    for x2, y2 in kp2:
        p1 = project_pt(h_2_1, x2, y2)
        if p1 is not None and 0.0 <= p1[0] < w1 and 0.0 <= p1[1] < h1:
            kp2_in_overlap.append((x2, y2))

    denom = min(len(kp1_in_overlap), len(kp2_in_overlap))
    if denom == 0:
        repeatability = 0.0
    else:
        matched_pts = 0
        for px, py in kp1_in_overlap:
            if any((px - qx) ** 2 + (py - qy) ** 2 <= 9.0 for qx, qy in kp2_in_overlap):
                matched_pts += 1
        repeatability = matched_pts / denom

    if not matches:
        return repeatability, 0.0, 0.0, 0.0, 0.0

    errs: list[float] = []
    for (x1, y1), (x2, y2) in matches:
        proj = project_pt(h_1_2, x1, y1)
        if proj is None:
            errs.append(float("inf"))
        else:
            errs.append(math.hypot(proj[0] - x2, proj[1] - y2))

    n_m = len(errs)
    mma1 = sum(1 for e in errs if e <= 1.0) / n_m
    mma3 = sum(1 for e in errs if e <= 3.0) / n_m
    mma5 = sum(1 for e in errs if e <= 5.0) / n_m
    matching_score = (sum(1 for e in errs if e <= 3.0) / denom) if denom > 0 else 0.0
    return repeatability, matching_score, mma1, mma3, mma5


def homography_corner_error(
    h_est: list[list[float]] | None,
    h_gt: list[list[float]],
    w1: int,
    h1: int,
) -> float:
    if h_est is None:
        return float("inf")
    corners = [(0.0, 0.0), (float(w1 - 1), 0.0), (float(w1 - 1), float(h1 - 1)), (0.0, float(h1 - 1))]
    total = 0.0
    for cx, cy in corners:
        pe = project_pt(h_est, cx, cy)
        pg = project_pt(h_gt, cx, cy)
        if pe is None or pg is None:
            return float("inf")
        total += math.hypot(pe[0] - pg[0], pe[1] - pg[1])
    return total * 0.25


def compute_auc(errors: list[float], thresholds: tuple[int, ...] = (3, 5, 10)) -> dict[str, float]:
    """Computes Area Under the Cumulative Error Curve at integer thresholds (3, 5, 10 px)."""
    if not errors:
        return {f"auc_{t}": 0.0 for t in thresholds}
    sorted_errs = [e for e in sorted(errors) if math.isfinite(e)]
    n_total = len(errors)
    out: dict[str, float] = {}
    for thresh in thresholds:
        # Stepwise recall AUC over [0, thresh]
        prev_x = 0.0
        auc = 0.0
        for rank, err in enumerate(sorted_errs):
            if err >= thresh:
                break
            recall = rank / n_total
            auc += recall * (err - prev_x)
            prev_x = err
        n_under = sum(1 for e in sorted_errs if e < thresh)
        auc += (n_under / n_total) * (thresh - prev_x)
        out[f"auc_{thresh}"] = auc / float(thresh)
    return out


def ensure_png_or_compatible(src: Path, tmp_dir: Path) -> Path:
    """`pure-sift` decodes PNG/JPEG/BMP/TIFF/WebP; convert `.ppm` HPatches images to `.png`."""
    if src.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return src
    out = tmp_dir / f"{src.parent.name}_{src.stem}.png"
    if not out.exists():
        with Image.open(src) as im:
            im.save(out, format="PNG")
    return out


def run_pure_sift_pair(
    bin_path: Path,
    img1: Path,
    img2: Path,
    octaves: int = 4,
    scales: int = 3,
    threshold: float = 0.015,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ext1 = subprocess.run(
        [str(bin_path), "extract", str(img1), "--octaves", str(octaves), "--scales", str(scales), "--threshold", str(threshold)],
        capture_output=True,
        text=True,
        check=True,
    )
    ext2 = subprocess.run(
        [str(bin_path), "extract", str(img2), "--octaves", str(octaves), "--scales", str(scales), "--threshold", str(threshold)],
        capture_output=True,
        text=True,
        check=True,
    )
    raw_match = subprocess.run(
        [str(bin_path), "match", str(img1), str(img2), "--no-ransac", "--octaves", str(octaves), "--scales", str(scales), "--threshold", str(threshold)],
        capture_output=True,
        text=True,
        check=True,
    )
    hom_match = subprocess.run(
        [str(bin_path), "match", str(img1), str(img2), "--octaves", str(octaves), "--scales", str(scales), "--threshold", str(threshold)],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    d1 = json.loads(ext1.stdout)
    d2 = json.loads(ext2.stdout)
    rm = json.loads(raw_match.stdout)
    hm = json.loads(hom_match.stdout)

    kp1 = [(float(k["x"]), float(k["y"])) for k in d1["keypoints"]]
    kp2 = [(float(k["x"]), float(k["y"])) for k in d2["keypoints"]]
    matches = [
        (tuple(m["query_point"]), tuple(m["target_point"]))  # type: ignore[misc]
        for m in rm["matches"]
    ]
    h_est = hm["homography"]["matrix"] if hm.get("homography") else None
    return {
        "kp1": kp1,
        "kp2": kp2,
        "size1": (int(d1["width"]), int(d1["height"])),
        "size2": (int(d2["width"]), int(d2["height"])),
        "matches": matches,
        "h_est": h_est,
        "runtime_ms": elapsed_ms / 4.0,
    }


def run_opencv_pair(img1: Path, img2: Path) -> dict[str, Any] | None:
    if cv2 is None or np is None:
        return None
    im1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(str(img2), cv2.IMREAD_GRAYSCALE)
    if im1 is None or im2 is None:
        return None

    sift = cv2.SIFT_create()
    t0 = time.perf_counter()
    kps1, desc1 = sift.detectAndCompute(im1, None)
    kps2, desc2 = sift.detectAndCompute(im2, None)

    matches: list[tuple[tuple[float, float], tuple[float, float]]] = []
    h_est: list[list[float]] | None = None
    if desc1 is not None and desc2 is not None and len(desc1) >= 2 and len(desc2) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
                p1 = kps1[pair[0].queryIdx].pt
                p2 = kps2[pair[0].trainIdx].pt
                matches.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
        if len(good) >= 4:
            src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            h_np, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if h_np is not None:
                h_est = h_np.tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "kp1": [(float(k.pt[0]), float(k.pt[1])) for k in kps1],
        "kp2": [(float(k.pt[0]), float(k.pt[1])) for k in kps2],
        "size1": (int(im1.shape[1]), int(im1.shape[0])),
        "size2": (int(im2.shape[1]), int(im2.shape[0])),
        "matches": matches,
        "h_est": h_est,
        "runtime_ms": elapsed_ms,
    }


def aggregate_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    n = float(len(records))
    keys = ["repeatability_3", "matching_score_3", "mma_1", "mma_3", "mma_5", "runtime_ms"]
    agg = {k: sum(r[k] for r in records) / n for k in keys}
    aucs = compute_auc([r["corner_err"] for r in records])
    agg.update(aucs)
    agg["pairs"] = float(len(records))
    return agg


def evaluate_dataset(dataset_dir: Path, bin_path: Path, max_sequences: int | None) -> dict[str, Any]:
    seq_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name[0] in ("i", "v")])
    if max_sequences is not None:
        seq_dirs = seq_dirs[:max_sequences]

    results: dict[str, dict[str, list[dict[str, float]]]] = {
        "pure_sift": {"viewpoint": [], "illumination": [], "overall": []},
        "opencv_sift": {"viewpoint": [], "illumination": [], "overall": []},
    }

    with tempfile.TemporaryDirectory(prefix="pure_sift_hp_") as tmp:
        tmp_dir = Path(tmp)
        for seq_dir in seq_dirs:
            split = "viewpoint" if seq_dir.name.startswith("v") else "illumination"
            ref_raw = next((seq_dir / f"1{ext}" for ext in (".ppm", ".png", ".jpg") if (seq_dir / f"1{ext}").exists()), None)
            if ref_raw is None:
                continue
            ref_img = ensure_png_or_compatible(ref_raw, tmp_dir)

            for k in range(2, 7):
                tgt_raw = next((seq_dir / f"{k}{ext}" for ext in (".ppm", ".png", ".jpg") if (seq_dir / f"{k}{ext}").exists()), None)
                h_path = seq_dir / f"H_1_{k}"
                if tgt_raw is None or not h_path.exists():
                    continue
                tgt_img = ensure_png_or_compatible(tgt_raw, tmp_dir)
                h_gt = parse_hpatches_matrix(h_path)

                for engine_name, fn in (
                    ("pure_sift", lambda a, b: run_pure_sift_pair(bin_path, a, b)),
                    ("opencv_sift", lambda a, b: run_opencv_pair(a, b)),
                ):
                    out = fn(ref_img, tgt_img)
                    if out is None:
                        continue
                    rep3, mscore3, mma1, mma3, mma5 = compute_repeatability_and_matching(
                        out["kp1"],
                        out["kp2"],
                        out["matches"],
                        h_gt,
                        out["size1"],
                        out["size2"],
                    )
                    cerr = homography_corner_error(out["h_est"], h_gt, out["size1"][0], out["size1"][1])
                    rec = {
                        "repeatability_3": rep3,
                        "matching_score_3": mscore3,
                        "mma_1": mma1,
                        "mma_3": mma3,
                        "mma_5": mma5,
                        "corner_err": cerr,
                        "runtime_ms": out["runtime_ms"],
                    }
                    results[engine_name][split].append(rec)
                    results[engine_name]["overall"].append(rec)

    summary = {
        "pure_sift": {split: aggregate_metrics(recs) for split, recs in results["pure_sift"].items()},
        "opencv_sift": {split: aggregate_metrics(recs) for split, recs in results["opencv_sift"].items()},
    }
    return summary


def format_markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Engine | Split | Pairs | Repeat@3px | MatchScore@3px | MMA@1px | MMA@3px | MMA@5px | HomAUC@3/5/10px | Mean ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for engine, label in (("pure_sift", "`pure-sift`"), ("opencv_sift", "OpenCV SIFT")):
        for split in ("viewpoint", "illumination", "overall"):
            m = summary[engine].get(split) or {}
            if not m:
                lines.append(f"| {label} | {split} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| {label} | {split} | {int(m['pairs'])} | "
                f"{m['repeatability_3']*100:.1f}% | {m['matching_score_3']*100:.1f}% | "
                f"{m['mma_1']*100:.1f}% | {m['mma_3']*100:.1f}% | {m['mma_5']*100:.1f}% | "
                f"{m['auc_3']*100:.1f}% / {m['auc_5']*100:.1f}% / {m['auc_10']*100:.1f}% | "
                f"{m['runtime_ms']:.1f} ms |"
            )
    return "\n".join(lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Benchmark pure-sift vs OpenCV SIFT on HPatches.")
    parser.add_argument("--download", action="store_true", help="Download HPatches sequences into .cache/hpatches/")
    parser.add_argument("--dataset-dir", type=Path, help="Path to extracted HPatches sequences directory")
    parser.add_argument("--pure-sift-bin", type=Path, default=default_pure_sift_bin(repo_root))
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--output", type=Path, default=repo_root / "docs" / "benchmarks" / "hpatches_results.json")
    parser.add_argument("--self-test", action="store_true", help="Run offline smoke benchmark on synthetic aerial viewpoint + illumination sequences")
    args = parser.parse_args()

    if not args.pure_sift_bin.exists():
        print(f"Error: pure-sift binary not found at {args.pure_sift_bin}. Run `cargo build --release` first.", file=sys.stderr)
        return 1

    temp_synth: Path | None = None
    if args.self_test:
        temp_synth = Path(tempfile.mkdtemp(prefix="pure_sift_synth_hp_"))
        dataset_dir = build_synthetic_hpatches(repo_root / "docs" / "samples" / "aerial_west.jpg", temp_synth)
    elif args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
    elif args.download:
        dataset_dir = download_hpatches(repo_root / ".cache" / "hpatches")
    else:
        print("Error: specify --dataset-dir, --download, or --self-test", file=sys.stderr)
        return 1

    try:
        summary = evaluate_dataset(dataset_dir, args.pure_sift_bin, args.max_sequences)
    finally:
        if temp_synth is not None:
            shutil.rmtree(temp_synth, ignore_errors=True)

    payload = {
        "mode": "synthetic_self_test" if args.self_test else "hpatches",
        "opencv_available": cv2 is not None,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(format_markdown_table(summary))
    print(f"\nSaved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
