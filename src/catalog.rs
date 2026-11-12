#![forbid(unsafe_code)]

use std::{fs, path::Path};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pipeline::{
    Homography2D, SIFT_DESCRIPTOR_LEN, SiftFeature, SimilarityTransform2D, TrackingSummary,
    estimate_homography_ransac, filter_matches_ransac, match_descriptors, summarize_tracking,
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
    pub entries: Vec<CatalogEntry>,
}

impl FeatureCatalog {
    /// Creates an empty `FeatureCatalog` configured with extraction parameters.
    pub fn new(octaves: usize, scales: usize, threshold: f32) -> Self {
        Self {
            octaves,
            scales,
            threshold,
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

    /// Persists the catalog to a JSON file.
    pub fn save_json(&self, path: &Path, pretty: bool) -> Result<(), String> {
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
        let json = if pretty {
            serde_json::to_string_pretty(self)
        } else {
            serde_json::to_string(self)
        }
        .map_err(|e| format!("Failed to serialize catalog JSON: {e}"))?;
        fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("Failed to write catalog file `{}`: {e}", path.display()))
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
