#![forbid(unsafe_code)]

use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::Path,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pipeline::{
    DescriptorKind, Homography2D, SIFT_DESCRIPTOR_LEN, SiftFeature, SimilarityTransform2D,
    TrackingSummary, dequantize_descriptor_u8, estimate_homography_ransac, filter_matches_ransac,
    match_descriptors, quantize_descriptor_u8, summarize_tracking,
};

/// Default maximum shortlist size of catalog entries verified with full 128-D SIFT + RANSAC.
pub const DEFAULT_CATALOG_SHORTLIST: usize = 16;

/// Default number of top catalog search results returned.
pub const DEFAULT_CATALOG_TOP_K: usize = 5;

/// Binary quantization threshold for 128-bit SIFT descriptor signatures.
/// Unit-L2 128-D descriptors have root-mean-square element `1 / sqrt(128) ~= 0.0884`
/// and typical median/mean active threshold around `0.065`.
const QUANTIZE_THRESHOLD: f32 = 0.065;

/// Maximum Hamming distance (out of 128 bits) counted as a coarse descriptor vote.
const COARSE_HAMMING_MAX_BITS: u32 = 24;

/// Packs a 128-D SIFT descriptor into a single 128-bit binary signature for 1-instruction
/// hardware `XOR + POPCNT` similarity prefiltering.
#[inline]
pub fn quantize_descriptor_u128(desc: &[f32]) -> u128 {
    let mut sig = 0u128;
    for (i, &val) in desc.iter().take(SIFT_DESCRIPTOR_LEN).enumerate() {
        if val > QUANTIZE_THRESHOLD {
            sig |= 1u128 << i;
        }
    }
    sig
}

/// Summary metadata for a single indexed image inside `FeatureCatalog`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntrySummary {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub feature_count: usize,
}

/// Compact SIFT feature storing quantized `u8` descriptors (128 bytes instead of 512 bytes for `Vec<f32>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactSiftFeature {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub response: f32,
    pub orientation: f32,
    pub descriptor_u8: Vec<u8>,
}

impl From<&SiftFeature> for CompactSiftFeature {
    fn from(feat: &SiftFeature) -> Self {
        Self {
            x: feat.x,
            y: feat.y,
            scale: feat.scale,
            response: feat.response,
            orientation: feat.orientation,
            descriptor_u8: quantize_descriptor_u8(&feat.descriptor).to_vec(),
        }
    }
}

impl CompactSiftFeature {
    /// Expands the compact `u8` descriptor back into a unit-L2 `SiftFeature`.
    pub fn to_sift_feature(&self) -> SiftFeature {
        SiftFeature {
            x: self.x,
            y: self.y,
            scale: self.scale,
            response: self.response,
            orientation: self.orientation,
            descriptor: dequantize_descriptor_u8(&self.descriptor_u8),
        }
    }
}

/// Stored catalog entry holding an image's extracted SIFT features and transient 128-bit signatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub features: Vec<SiftFeature>,
    #[serde(skip)]
    pub signatures: Vec<u128>,
}

impl CatalogEntry {
    pub fn new(
        id: impl Into<String>,
        width: usize,
        height: usize,
        features: Vec<SiftFeature>,
    ) -> Self {
        let signatures = features
            .iter()
            .map(|f| quantize_descriptor_u128(&f.descriptor))
            .collect();
        Self {
            id: id.into(),
            width,
            height,
            features,
            signatures,
        }
    }

    pub fn rebuild_signatures(&mut self) {
        self.signatures = self
            .features
            .iter()
            .map(|f| quantize_descriptor_u128(&f.descriptor))
            .collect();
    }

    /// Exports this entry's features with 4x-smaller `u8` quantized descriptors.
    pub fn compact_features(&self) -> Vec<CompactSiftFeature> {
        self.features.iter().map(CompactSiftFeature::from).collect()
    }
}

/// Ranked visual-search hit produced by `FeatureCatalog::search_features`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSearchHit {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub feature_count: usize,
    pub coarse_votes: usize,
    pub tracking: TrackingSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<SimilarityTransform2D>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homography: Option<Homography2D>,
}

/// Serializable, multi-core SIFT visual search catalog with 128-bit binary prefiltering
/// and exact geometric RANSAC verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureCatalog {
    pub octaves: usize,
    pub scales: usize,
    pub threshold: f32,
    #[serde(default)]
    pub descriptor_kind: DescriptorKind,
    pub entries: Vec<CatalogEntry>,
}

const BINARY_CATALOG_MAGIC: &[u8; 8] = b"PSIFTCAT";
const BINARY_CATALOG_VERSION: u32 = 1;

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create catalog directory `{}`: {e}",
                parent.display()
            )
        })?;
    }

    let tmp_name = format!(
        "{}.tmp.{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("catalog"),
        std::process::id()
    );
    let tmp_path = path.with_file_name(tmp_name);

    {
        let file = File::create(&tmp_path).map_err(|e| {
            format!(
                "Failed to create temporary catalog `{}`: {e}",
                tmp_path.display()
            )
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(bytes)
            .and_then(|_| writer.flush())
            .map_err(|e| format!("Failed to write catalog `{}`: {e}", tmp_path.display()))?;
    }

    fs::rename(&tmp_path, path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "Failed to atomically replace catalog `{}`: {e}",
            path.display()
        )
    })
}

fn read_exact_arr<const N: usize>(reader: &mut &[u8]) -> Result<[u8; N], String> {
    if reader.len() < N {
        return Err("Unexpected EOF while reading binary catalog".to_string());
    }
    let mut buf = [0u8; N];
    buf.copy_from_slice(&reader[..N]);
    *reader = &reader[N..];
    Ok(buf)
}

#[inline]
fn read_u32_le(r: &mut &[u8]) -> Result<u32, String> {
    Ok(u32::from_le_bytes(read_exact_arr::<4>(r)?))
}

#[inline]
fn read_u64_le(r: &mut &[u8]) -> Result<u64, String> {
    Ok(u64::from_le_bytes(read_exact_arr::<8>(r)?))
}

#[inline]
fn read_f32_le(r: &mut &[u8]) -> Result<f32, String> {
    Ok(f32::from_le_bytes(read_exact_arr::<4>(r)?))
}

#[inline]
fn read_u128_le(r: &mut &[u8]) -> Result<u128, String> {
    Ok(u128::from_le_bytes(read_exact_arr::<16>(r)?))
}

impl FeatureCatalog {
    /// Creates an empty `FeatureCatalog` configured with extraction parameters.
    pub fn new(octaves: usize, scales: usize, threshold: f32) -> Self {
        Self {
            octaves,
            scales,
            threshold,
            descriptor_kind: DescriptorKind::Sift,
            entries: Vec::new(),
        }
    }

    /// Inserts or replaces an image entry in the catalog.
    pub fn upsert_entry(
        &mut self,
        id: impl Into<String>,
        width: usize,
        height: usize,
        features: Vec<SiftFeature>,
    ) {
        let id_str = id.into();
        let entry = CatalogEntry::new(id_str.clone(), width, height, features);
        if let Some(existing) = self.entries.iter_mut().find(|e| e.id == id_str) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Returns lightweight summaries for all indexed entries.
    pub fn list_summaries(&self) -> Vec<CatalogEntrySummary> {
        self.entries
            .iter()
            .map(|e| CatalogEntrySummary {
                id: e.id.clone(),
                width: e.width,
                height: e.height,
                feature_count: e.features.len(),
            })
            .collect()
    }

    /// Recomputes transient 128-bit signatures (called after deserializing from disk).
    pub fn rebuild_all_signatures(&mut self) {
        self.entries
            .par_iter_mut()
            .for_each(CatalogEntry::rebuild_signatures);
    }

    /// Persists the catalog to a JSON file atomically (write to temp file + rename).
    pub fn save_json(&self, path: &Path, pretty: bool) -> Result<(), String> {
        let mut json = if pretty {
            serde_json::to_string_pretty(self)
        } else {
            serde_json::to_string(self)
        }
        .map_err(|e| format!("Failed to serialize catalog JSON: {e}"))?;
        json.push('\n');
        atomic_write_bytes(path, json.as_bytes())
    }

    /// Loads a `FeatureCatalog` from a JSON file and rebuilds in-memory 128-bit signatures.
    pub fn load_json(path: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read catalog `{}`: {e}", path.display()))?;
        let mut catalog: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("Failed to parse catalog JSON `{}`: {e}", path.display()))?;
        catalog.rebuild_all_signatures();
        Ok(catalog)
    }

    /// Persists the catalog in the compact, versioned little-endian binary format (`PSIFTCAT`)
    /// using quantized `u8` descriptors and precomputed 128-bit signatures, written atomically.
    pub fn save_binary(&self, path: &Path) -> Result<(), String> {
        let total_feats: usize = self.entries.iter().map(|e| e.features.len()).sum();
        let mut buf = Vec::with_capacity(64 + self.entries.len() * 64 + total_feats * 164);

        buf.extend_from_slice(BINARY_CATALOG_MAGIC);
        buf.extend_from_slice(&BINARY_CATALOG_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.octaves as u32).to_le_bytes());
        buf.extend_from_slice(&(self.scales as u32).to_le_bytes());
        buf.extend_from_slice(&self.threshold.to_le_bytes());
        let kind_byte: u8 = match self.descriptor_kind {
            DescriptorKind::Sift => 0,
            DescriptorKind::RootSift => 1,
        };
        buf.push(kind_byte);
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());

        for entry in &self.entries {
            let id_bytes = entry.id.as_bytes();
            buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(id_bytes);
            buf.extend_from_slice(&(entry.width as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.height as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.features.len() as u64).to_le_bytes());

            for (idx, feat) in entry.features.iter().enumerate() {
                buf.extend_from_slice(&feat.x.to_le_bytes());
                buf.extend_from_slice(&feat.y.to_le_bytes());
                buf.extend_from_slice(&feat.scale.to_le_bytes());
                buf.extend_from_slice(&feat.response.to_le_bytes());
                buf.extend_from_slice(&feat.orientation.to_le_bytes());
                let sig = entry
                    .signatures
                    .get(idx)
                    .copied()
                    .unwrap_or_else(|| quantize_descriptor_u128(&feat.descriptor));
                buf.extend_from_slice(&sig.to_le_bytes());
                let q = quantize_descriptor_u8(&feat.descriptor);
                buf.extend_from_slice(&q);
            }
        }

        atomic_write_bytes(path, &buf)
    }

    /// Loads a `FeatureCatalog` from the compact binary format (`PSIFTCAT`), restoring stored
    /// 128-bit signatures directly without recomputing them.
    pub fn load_binary(path: &Path) -> Result<Self, String> {
        let raw = fs::read(path)
            .map_err(|e| format!("Failed to read binary catalog `{}`: {e}", path.display()))?;
        let mut r: &[u8] = &raw;

        let magic = read_exact_arr::<8>(&mut r)?;
        if &magic != BINARY_CATALOG_MAGIC {
            return Err(format!(
                "Invalid binary catalog magic header in `{}`",
                path.display()
            ));
        }
        let version = read_u32_le(&mut r)?;
        if version != BINARY_CATALOG_VERSION {
            return Err(format!(
                "Unsupported binary catalog version {version} in `{}` (expected {BINARY_CATALOG_VERSION})",
                path.display()
            ));
        }

        let octaves = read_u32_le(&mut r)? as usize;
        let scales = read_u32_le(&mut r)? as usize;
        let threshold = read_f32_le(&mut r)?;
        let kind_byte = read_exact_arr::<1>(&mut r)?[0];
        let descriptor_kind = match kind_byte {
            0 => DescriptorKind::Sift,
            1 => DescriptorKind::RootSift,
            other => {
                return Err(format!(
                    "Unknown descriptor kind {other} in binary catalog `{}`",
                    path.display()
                ));
            }
        };
        let entry_count = read_u64_le(&mut r)? as usize;
        let mut entries = Vec::with_capacity(entry_count.min(100_000));

        for _ in 0..entry_count {
            let id_len = read_u32_le(&mut r)? as usize;
            if r.len() < id_len {
                return Err("Truncated entry id in binary catalog".to_string());
            }
            let id = std::str::from_utf8(&r[..id_len])
                .map_err(|e| format!("Invalid UTF-8 entry id in binary catalog: {e}"))?
                .to_string();
            r = &r[id_len..];

            let width = read_u32_le(&mut r)? as usize;
            let height = read_u32_le(&mut r)? as usize;
            let feat_count = read_u64_le(&mut r)? as usize;

            let mut features = Vec::with_capacity(feat_count.min(100_000));
            let mut signatures = Vec::with_capacity(feat_count.min(100_000));

            for _ in 0..feat_count {
                let x = read_f32_le(&mut r)?;
                let y = read_f32_le(&mut r)?;
                let scale = read_f32_le(&mut r)?;
                let response = read_f32_le(&mut r)?;
                let orientation = read_f32_le(&mut r)?;
                let sig = read_u128_le(&mut r)?;
                let q = read_exact_arr::<128>(&mut r)?;
                features.push(SiftFeature {
                    x,
                    y,
                    scale,
                    response,
                    orientation,
                    descriptor: dequantize_descriptor_u8(&q),
                });
                signatures.push(sig);
            }

            entries.push(CatalogEntry {
                id,
                width,
                height,
                features,
                signatures,
            });
        }

        Ok(Self {
            octaves,
            scales,
            threshold,
            descriptor_kind,
            entries,
        })
    }

    /// Saves as JSON when `path` ends with `.json` (case-insensitive), or compact binary otherwise.
    pub fn save_auto(&self, path: &Path, pretty: bool) -> Result<(), String> {
        let is_json = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        if is_json {
            self.save_json(path, pretty)
        } else {
            self.save_binary(path)
        }
    }

    /// Loads either a binary (`PSIFTCAT`) or legacy JSON catalog automatically by inspecting magic bytes.
    pub fn load_auto(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path)
            .map_err(|e| format!("Failed to open catalog `{}`: {e}", path.display()))?;
        let mut header = [0u8; 8];
        let n = file
            .read(&mut header)
            .map_err(|e| format!("Failed to read catalog header `{}`: {e}", path.display()))?;
        if n == 8 && &header == BINARY_CATALOG_MAGIC {
            Self::load_binary(path)
        } else {
            Self::load_json(path)
        }
    }

    /// Searches the catalog for the best geometric matches to `query_features`.
    ///
    /// 1. When `entries.len() > shortlist_k`, computes parallel 128-bit Hamming votes (`XOR + POPCNT`)
    ///    across all catalog entries to shortlist the top `shortlist_k` candidates.
    /// 2. Verifies shortlisted entries in parallel via exact 128-D Lowe ratio matching and
    ///    Similarity + Homography RANSAC, returning the top `top_k` hits ranked by inlier count.
    pub fn search_features(
        &self,
        query_features: &[SiftFeature],
        top_k: usize,
        shortlist_k: usize,
        ratio_threshold: f32,
        ransac_threshold: f32,
    ) -> Vec<CatalogSearchHit> {
        if self.entries.is_empty() || query_features.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let query_sigs: Vec<u128> = query_features
            .iter()
            .map(|f| quantize_descriptor_u128(&f.descriptor))
            .collect();

        // Step 1: Parallel coarse 128-bit Hamming voting across all catalog entries
        let mut candidates: Vec<(usize, usize)> = self
            .entries
            .par_iter()
            .enumerate()
            .map(|(idx, entry)| {
                let votes = count_coarse_hamming_votes(&query_sigs, &entry.signatures);
                (idx, votes)
            })
            .collect();

        if candidates.len() > shortlist_k.max(1) {
            candidates.sort_by(|a, b| {
                b.1.cmp(&a.1)
                    .then_with(|| self.entries[a.0].id.cmp(&self.entries[b.0].id))
            });
            candidates.truncate(shortlist_k.max(1));
        }

        // Step 2: Parallel exact 128-D Lowe ratio + RANSAC geometric verification on shortlist
        let mut hits: Vec<CatalogSearchHit> = candidates
            .par_iter()
            .filter_map(|&(idx, coarse_votes)| {
                let entry = &self.entries[idx];
                if entry.features.len() < 2 {
                    return None;
                }

                let raw_matches =
                    match_descriptors(query_features, &entry.features, ratio_threshold);
                if raw_matches.is_empty() {
                    return None;
                }

                let mut tracking =
                    summarize_tracking(query_features, &entry.features, &raw_matches);
                let (sim_inliers, sim_tf) = filter_matches_ransac(&raw_matches, ransac_threshold);
                let (hom_inliers, hom_tf) =
                    estimate_homography_ransac(&raw_matches, ransac_threshold);
                let best_inliers = hom_inliers.len().max(sim_inliers.len());
                tracking.match_count = best_inliers;
                if tracking.raw_match_count > 0 {
                    tracking.inlier_ratio =
                        (best_inliers as f32) / (tracking.raw_match_count as f32);
                }

                Some(CatalogSearchHit {
                    id: entry.id.clone(),
                    width: entry.width,
                    height: entry.height,
                    feature_count: entry.features.len(),
                    coarse_votes,
                    tracking,
                    transform: sim_tf,
                    homography: hom_tf,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.tracking
                .match_count
                .cmp(&a.tracking.match_count)
                .then_with(|| {
                    b.tracking
                        .inlier_ratio
                        .partial_cmp(&a.tracking.inlier_ratio)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b.coarse_votes.cmp(&a.coarse_votes))
                .then_with(|| a.id.cmp(&b.id))
        });

        hits.truncate(top_k);
        hits
    }
}

/// Counts how many query 128-bit signatures find a catalog signature within `COARSE_HAMMING_MAX_BITS`.
#[inline]
fn count_coarse_hamming_votes(query_sigs: &[u128], entry_sigs: &[u128]) -> usize {
    if entry_sigs.is_empty() {
        return 0;
    }
    let mut votes = 0usize;
    for &qs in query_sigs {
        let mut min_dist = u32::MAX;
        for &es in entry_sigs {
            let dist = (qs ^ es).count_ones();
            if dist < min_dist {
                min_dist = dist;
                if min_dist <= 8 {
                    break;
                }
            }
        }
        if min_dist <= COARSE_HAMMING_MAX_BITS {
            votes += 1;
        }
    }
    votes
}
