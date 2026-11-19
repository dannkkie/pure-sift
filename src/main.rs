#![forbid(unsafe_code)]

use std::process::ExitCode;

use pure_sift::cli::{CliCommand, execute_cli_to_stdout, parse_cli_args};

#[cfg(feature = "server")]
mod server {
    use std::{
        path::PathBuf,
        sync::{
            Arc, RwLock,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
        time::{Duration, Instant},
    };

    use axum::{
        Json, Router,
        body::Body,
        extract::{DefaultBodyLimit, Multipart, Path, Query, State},
        http::{HeaderName, HeaderValue, Request, StatusCode, header},
        middleware::{self, Next},
        response::{Html, IntoResponse, Response},
        routing::{delete, get, post},
    };
    use pure_sift::{
        catalog::{
            CatalogEntrySummary, DEFAULT_CATALOG_SHORTLIST, DEFAULT_CATALOG_TOP_K, FeatureCatalog,
        },
        cli::{
            CatalogSearchOptions, DEFAULT_OCTAVES, DEFAULT_SCALES, PairPipelineOptions,
            VisionError, run_align_images_rgba, run_catalog_search_from_bytes,
            run_extract_from_bytes, run_match_from_bytes, run_visualize_features_rgba,
            run_visualize_matches_rgba, validate_match_params, validate_pipeline_params,
        },
        pipeline::{
            DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD,
            MatchOptions, RansacOptions, SiftConfig,
        },
        visualize::encode_rgba_png,
    };
    use serde::{Deserialize, Serialize};
    use tokio::{
        net::TcpListener,
        sync::{Notify, Semaphore},
    };

    /// Shared server state holding the live in-memory SIFT `FeatureCatalog`, concurrency limiter,
    /// optional bearer `api_token`, and debounced persistence trigger.
    #[derive(Clone)]
    pub struct AppState {
        pub catalog: Arc<RwLock<FeatureCatalog>>,
        pub catalog_path: Option<PathBuf>,
        pub catalog_save_notify: Arc<Notify>,
        pub job_semaphore: Arc<Semaphore>,
        pub api_token: Option<Arc<String>>,
        pub request_timeout: Duration,
    }

    /// Default bind address (loopback only; require explicit `--bind` or `BIND_ADDR` for public exposure).
    const DEFAULT_BIND_ADDR: &str = "127.0.0.1:3000";

    /// Default multipart upload body limit (16 MiB, overridable via `MAX_BODY_MB`).
    const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

    /// Default per-request processing timeout in seconds (overridable via `REQUEST_TIMEOUT_SECS`).
    const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

    /// Default maximum number of concurrent CPU-bound vision jobs (overridable via `MAX_CONCURRENT_JOBS`).
    const DEFAULT_MAX_CONCURRENT_JOBS: usize = 4;

    static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

    /// Optional query parameters for `POST /api/v1/process-features` and `/api/v1/visualize-features`.
    #[derive(Debug, Clone, Copy, Default, Deserialize)]
    pub struct ProcessFeaturesParams {
        pub octaves: Option<usize>,
        pub scales: Option<usize>,
        pub threshold: Option<f32>,
        pub include_descriptors: Option<bool>,
    }

    /// Optional query parameters for `POST /api/v1/match-features`.
    #[derive(Debug, Clone, Copy, Default, Deserialize)]
    pub struct MatchFeaturesParams {
        pub octaves: Option<usize>,
        pub scales: Option<usize>,
        pub threshold: Option<f32>,
        pub ratio: Option<f32>,
        pub ransac: Option<bool>,
        pub ransac_threshold: Option<f32>,
    }

    /// Optional query parameters for `POST /api/v1/align-images`.
    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct AlignImagesParams {
        pub octaves: Option<usize>,
        pub scales: Option<usize>,
        pub threshold: Option<f32>,
        pub ratio: Option<f32>,
        pub ransac: Option<bool>,
        pub ransac_threshold: Option<f32>,
        /// Visual output mode: `"stitch"` (default warped/blended panorama) or `"matches"` (side-by-side inliers).
        pub mode: Option<String>,
    }

    /// Optional query parameters for `POST /api/v1/catalog/index`.
    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct CatalogIndexParams {
        pub id: String,
    }

    /// Optional query parameters for `POST /api/v1/catalog/search`.
    #[derive(Debug, Clone, Copy, Default, Deserialize)]
    pub struct CatalogSearchParams {
        pub top_k: Option<usize>,
        pub shortlist: Option<usize>,
        pub ratio: Option<f32>,
        pub ransac_threshold: Option<f32>,
    }

    /// Response returned by `GET /api/v1/catalog`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CatalogListResponse {
        pub catalog_size: usize,
        pub octaves: usize,
        pub scales: usize,
        pub threshold: f32,
        pub entries: Vec<CatalogEntrySummary>,
    }

    /// Response returned by `POST /api/v1/catalog/index`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CatalogIndexResponse {
        pub id: String,
        pub width: usize,
        pub height: usize,
        pub feature_count: usize,
        pub catalog_size: usize,
    }

    /// Structured JSON response returned by `GET /health`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HealthResponse {
        pub status: &'static str,
        pub service: &'static str,
        pub version: &'static str,
    }

    /// Structured JSON error payload for invalid requests.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ErrorResponse {
        pub error: String,
    }

    fn json_error_response(status: StatusCode, msg: impl Into<String>) -> Response {
        (
            status,
            Json(serde_json::json!(ErrorResponse { error: msg.into() })),
        )
            .into_response()
    }

    fn map_vision_error(err: VisionError) -> Response {
        let status = match &err {
            VisionError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            VisionError::UnprocessableImage(_) | VisionError::ImageDecode { .. } => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            VisionError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            VisionError::Io { .. } | VisionError::Json { .. } | VisionError::Runtime(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        json_error_response(status, err.to_string())
    }

    fn png_response(png_bytes: Vec<u8>) -> Response {
        let mut resp = Response::new(Body::from(png_bytes));
        *resp.status_mut() = StatusCode::OK;
        resp.headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
        resp
    }

    #[allow(clippy::result_large_err)]
    async fn read_multipart_images(
        mut multipart: Multipart,
        expected_count: usize,
    ) -> Result<Vec<Vec<u8>>, Response> {
        let mut images: Vec<Vec<u8>> = Vec::with_capacity(expected_count);

        loop {
            match multipart.next_field().await {
                Ok(Some(field)) => match field.bytes().await {
                    Ok(bytes) if !bytes.is_empty() => {
                        images.push(bytes.to_vec());
                        if images.len() == expected_count {
                            break;
                        }
                    }
                    Ok(_) => continue,
                    Err(err) => {
                        let status = if err.status() == StatusCode::PAYLOAD_TOO_LARGE {
                            StatusCode::PAYLOAD_TOO_LARGE
                        } else {
                            StatusCode::BAD_REQUEST
                        };
                        return Err(json_error_response(
                            status,
                            format!("Failed to read multipart field bytes: {err}"),
                        ));
                    }
                },
                Ok(None) => break,
                Err(err) => {
                    let status = if err.status() == StatusCode::PAYLOAD_TOO_LARGE {
                        StatusCode::PAYLOAD_TOO_LARGE
                    } else {
                        StatusCode::BAD_REQUEST
                    };
                    return Err(json_error_response(
                        status,
                        format!("Invalid multipart payload: {err}"),
                    ));
                }
            }
        }

        if images.len() < expected_count {
            let msg = if expected_count == 1 {
                "No image payload found in multipart form-data request".to_string()
            } else {
                format!(
                    "Expected {expected_count} image fields (query and target) in multipart request"
                )
            };
            return Err(json_error_response(StatusCode::BAD_REQUEST, msg));
        }

        Ok(images)
    }

    #[allow(clippy::result_large_err)]
    pub(crate) async fn run_blocking_with_timeout<T, F>(
        state: &AppState,
        timeout_msg: &'static str,
        job: F,
    ) -> Result<T, Response>
    where
        T: Send + 'static,
        F: FnOnce(Arc<AtomicBool>) -> Result<T, VisionError> + Send + 'static,
    {
        let timeout = state.request_timeout;
        let permit = match tokio::time::timeout(
            timeout,
            state.job_semaphore.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                return Err(json_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Server vision worker pool closed",
                ));
            }
            Err(_) => {
                return Err(json_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Server busy: concurrent vision job limit reached",
                ));
            }
        };

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_flag = Arc::clone(&cancelled);
        let task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if cancel_flag.load(Ordering::Relaxed) {
                return Err(VisionError::InvalidInput("Request timed out".to_string()));
            }
            job(cancel_flag)
        });

        match tokio::time::timeout(timeout, task).await {
            Ok(Ok(Ok(val))) => Ok(val),
            Ok(Ok(Err(err))) => Err(map_vision_error(err)),
            Ok(Err(join_err)) => Err(json_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Blocking vision worker panicked: {join_err}"),
            )),
            Err(_) => {
                cancelled.store(true, Ordering::Relaxed);
                Err(json_error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    timeout_msg,
                ))
            }
        }
    }

    fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut diff = 0u8;
        for (&x, &y) in a.iter().zip(b.iter()) {
            diff |= x ^ y;
        }
        diff == 0
    }

    /// Optional bearer token authentication middleware protecting `/api/*` routes when `API_TOKEN` is set.
    async fn auth_middleware(
        State(state): State<AppState>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        if let Some(expected_token) = &state.api_token
            && req.uri().path().starts_with("/api/")
        {
            let auth_header = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok());
            let valid = match auth_header.and_then(|h| h.strip_prefix("Bearer ")) {
                Some(provided) => constant_time_eq(provided.as_bytes(), expected_token.as_bytes()),
                None => false,
            };
            if !valid {
                return json_error_response(
                    StatusCode::UNAUTHORIZED,
                    "Missing or invalid Bearer token",
                );
            }
        }
        next.run(req).await
    }

    /// Structured request logger middleware using `tracing` with a per-request `x-request-id`.
    /// Logs only the HTTP path (omitting query strings) to avoid leaking tokens or parameters.
    async fn request_logger_middleware(req: Request<Body>, next: Next) -> Response {
        let req_id = format!("req-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed));
        let method = req.method().clone();
        let path = req.uri().path().to_string();
        let start = Instant::now();
        let mut response = next.run(req).await;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let status = response.status().as_u16();
        if let Ok(hdr_val) = HeaderValue::from_str(&req_id) {
            response
                .headers_mut()
                .insert(HeaderName::from_static("x-request-id"), hdr_val);
        }
        tracing::info!(
            request_id = %req_id,
            method = %method,
            path = %path,
            status = status,
            elapsed_ms = %format!("{elapsed_ms:.2}"),
            "http request"
        );
        response
    }

    /// Embedded single-file HTML/CSS/JS workbench served at `GET /` with zero external dependencies.
    const WEB_UI_HTML: &str = include_str!("web_ui.html");

    /// Server runtime options constructed from environment variables or test overrides.
    #[derive(Clone)]
    pub struct ServerOptions {
        pub max_body_bytes: usize,
        pub max_concurrent_jobs: usize,
        pub request_timeout: Duration,
        pub api_token: Option<String>,
        pub catalog_path: Option<PathBuf>,
    }

    impl Default for ServerOptions {
        fn default() -> Self {
            let max_body_bytes = std::env::var("MAX_BODY_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|mb| mb.saturating_mul(1024 * 1024))
                .unwrap_or(DEFAULT_MAX_BODY_BYTES);

            let max_concurrent_jobs = std::env::var("MAX_CONCURRENT_JOBS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(DEFAULT_MAX_CONCURRENT_JOBS);

            let timeout_secs = std::env::var("REQUEST_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS);

            let api_token = std::env::var("API_TOKEN")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());

            let catalog_path = std::env::var("CATALOG_PATH")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from);

            Self {
                max_body_bytes,
                max_concurrent_jobs,
                request_timeout: Duration::from_secs(timeout_secs.max(1)),
                api_token,
                catalog_path,
            }
        }
    }

    /// Builds the hardened Axum application router with explicit `ServerOptions`.
    pub fn build_router_with_options(opts: ServerOptions) -> (Router, AppState) {
        let catalog = if let Some(ref path) = opts.catalog_path
            && path.exists()
        {
            FeatureCatalog::load_auto(path).unwrap_or_else(|err| {
                tracing::warn!(path = %path.display(), error = %err, "failed to load CATALOG_PATH");
                FeatureCatalog::new(DEFAULT_OCTAVES, DEFAULT_SCALES, DEFAULT_CONTRAST_THRESHOLD)
            })
        } else {
            FeatureCatalog::new(DEFAULT_OCTAVES, DEFAULT_SCALES, DEFAULT_CONTRAST_THRESHOLD)
        };

        let catalog_arc = Arc::new(RwLock::new(catalog));
        let catalog_save_notify = Arc::new(Notify::new());

        // Spawn debounced background catalog persistence task when CATALOG_PATH is configured
        if let Some(save_path) = opts.catalog_path.clone() {
            let catalog_ref = Arc::clone(&catalog_arc);
            let notify_ref = Arc::clone(&catalog_save_notify);
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    loop {
                        notify_ref.notified().await;
                        tokio::time::sleep(Duration::from_millis(250)).await;
                        let cat_clone = {
                            catalog_ref.read().ok().map(|g| g.clone())
                        };
                        if let Some(cat) = cat_clone {
                            let path_clone = save_path.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                if let Err(err) = cat.save_auto(&path_clone, false) {
                                    tracing::error!(path = %path_clone.display(), error = %err, "background catalog save failed");
                                }
                            })
                            .await;
                        }
                    }
                });
            }
        }

        let state = AppState {
            catalog: catalog_arc,
            catalog_path: opts.catalog_path,
            catalog_save_notify,
            job_semaphore: Arc::new(Semaphore::new(opts.max_concurrent_jobs)),
            api_token: opts.api_token.map(Arc::new),
            request_timeout: opts.request_timeout,
        };

        let router = Router::new()
            .route("/", get(web_ui_handler))
            .route("/health", get(health_handler))
            .route("/api/v1/process-features", post(process_features_handler))
            .route("/api/v1/match-features", post(match_features_handler))
            .route(
                "/api/v1/visualize-features",
                post(visualize_features_handler),
            )
            .route("/api/v1/align-images", post(align_images_handler))
            .route("/api/v1/catalog", get(catalog_list_handler))
            .route("/api/v1/catalog/index", post(catalog_index_handler))
            .route("/api/v1/catalog/search", post(catalog_search_handler))
            .route("/api/v1/catalog/{id}", delete(catalog_delete_handler))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state.clone())
            .layer(DefaultBodyLimit::max(opts.max_body_bytes))
            .layer(middleware::from_fn(request_logger_middleware));

        (router, state)
    }

    /// Builds the hardened Axum application router from environment configuration.
    #[cfg(test)]
    pub fn build_router() -> Router {
        build_router_with_options(ServerOptions::default()).0
    }

    /// `GET /` — serves the interactive single-file SIFT & RANSAC Web UI workbench.
    async fn web_ui_handler() -> Html<&'static str> {
        Html(WEB_UI_HTML)
    }

    /// Health check endpoint for load balancers, container orchestrators, and readiness probes.
    async fn health_handler() -> impl IntoResponse {
        (
            StatusCode::OK,
            Json(HealthResponse {
                status: "ok",
                service: "pure-sift",
                version: env!("CARGO_PKG_VERSION"),
            }),
        )
    }

    /// `POST /api/v1/process-features` — returns detected keypoints (and optional 128-D descriptors) as JSON.
    async fn process_features_handler(
        State(state): State<AppState>,
        Query(params): Query<ProcessFeaturesParams>,
        multipart: Multipart,
    ) -> Response {
        let octaves = params.octaves.unwrap_or(DEFAULT_OCTAVES);
        let scales = params.scales.unwrap_or(DEFAULT_SCALES);
        let threshold = params.threshold.unwrap_or(DEFAULT_CONTRAST_THRESHOLD);
        let include_descriptors = params.include_descriptors.unwrap_or(false);

        if let Err(err) = validate_pipeline_params(octaves, scales, threshold) {
            return map_vision_error(err);
        }

        let images = match read_multipart_images(multipart, 1).await {
            Ok(imgs) => imgs,
            Err(resp) => return resp,
        };
        let raw_bytes = images.into_iter().next().expect("length 1 verified");

        match run_blocking_with_timeout(&state, "Feature extraction timed out", move |_cancel| {
            run_extract_from_bytes(
                &raw_bytes,
                octaves,
                scales,
                threshold,
                include_descriptors,
                true,
            )
        })
        .await
        {
            Ok(response) => (StatusCode::OK, Json(serde_json::json!(response))).into_response(),
            Err(resp) => resp,
        }
    }

    /// `POST /api/v1/match-features` — returns SIFT matches, Similarity & Homography transforms, and tracking summary.
    async fn match_features_handler(
        State(state): State<AppState>,
        Query(params): Query<MatchFeaturesParams>,
        multipart: Multipart,
    ) -> Response {
        let octaves = params.octaves.unwrap_or(DEFAULT_OCTAVES);
        let scales = params.scales.unwrap_or(DEFAULT_SCALES);
        let threshold = params.threshold.unwrap_or(DEFAULT_CONTRAST_THRESHOLD);
        let ratio_threshold = params.ratio.unwrap_or(DEFAULT_MATCH_RATIO);
        let ransac_enabled = params.ransac.unwrap_or(true);
        let ransac_threshold = params
            .ransac_threshold
            .unwrap_or(DEFAULT_RANSAC_REPROJ_THRESHOLD);

        if let Err(err) = validate_pipeline_params(octaves, scales, threshold) {
            return map_vision_error(err);
        }
        if let Err(err) = validate_match_params(ratio_threshold, ransac_threshold) {
            return map_vision_error(err);
        }

        let mut images = match read_multipart_images(multipart, 2).await {
            Ok(imgs) => imgs,
            Err(resp) => return resp,
        };
        let target_bytes = images.pop().expect("length 2 verified");
        let query_bytes = images.pop().expect("length 2 verified");

        match run_blocking_with_timeout(&state, "Feature matching timed out", move |_cancel| {
            run_match_from_bytes(
                &query_bytes,
                &target_bytes,
                PairPipelineOptions {
                    sift: SiftConfig::default()
                        .octaves(octaves)
                        .scales(scales)
                        .contrast_threshold(threshold),
                    match_options: MatchOptions {
                        ratio: ratio_threshold,
                        mutual: false,
                    },
                    ransac_enabled,
                    ransac: RansacOptions::default().reproj_threshold(ransac_threshold),
                    enforce_pixel_cap: true,
                },
            )
        })
        .await
        {
            Ok(response) => (StatusCode::OK, Json(serde_json::json!(response))).into_response(),
            Err(resp) => resp,
        }
    }

    /// `POST /api/v1/visualize-features` — returns a PNG with detected keypoints (scale circles & orientation ticks).
    async fn visualize_features_handler(
        State(state): State<AppState>,
        Query(params): Query<ProcessFeaturesParams>,
        multipart: Multipart,
    ) -> Response {
        let octaves = params.octaves.unwrap_or(DEFAULT_OCTAVES);
        let scales = params.scales.unwrap_or(DEFAULT_SCALES);
        let threshold = params.threshold.unwrap_or(DEFAULT_CONTRAST_THRESHOLD);

        if let Err(err) = validate_pipeline_params(octaves, scales, threshold) {
            return map_vision_error(err);
        }

        let images = match read_multipart_images(multipart, 1).await {
            Ok(imgs) => imgs,
            Err(resp) => return resp,
        };
        let raw_bytes = images.into_iter().next().expect("length 1 verified");

        match run_blocking_with_timeout(&state, "Feature visualization timed out", move |_cancel| {
            let (rendered, _) =
                run_visualize_features_rgba(&raw_bytes, octaves, scales, threshold, true)?;
            encode_rgba_png(&rendered).map_err(VisionError::Runtime)
        })
        .await
        {
            Ok(png_bytes) => png_response(png_bytes),
            Err(resp) => resp,
        }
    }

    /// `POST /api/v1/align-images` — returns either a warped/feather-blended PNG (`mode=stitch`, default)
    /// or a side-by-side correspondence PNG (`mode=matches`).
    async fn align_images_handler(
        State(state): State<AppState>,
        Query(params): Query<AlignImagesParams>,
        multipart: Multipart,
    ) -> Response {
        let octaves = params.octaves.unwrap_or(DEFAULT_OCTAVES);
        let scales = params.scales.unwrap_or(DEFAULT_SCALES);
        let threshold = params.threshold.unwrap_or(DEFAULT_CONTRAST_THRESHOLD);
        let ratio_threshold = params.ratio.unwrap_or(DEFAULT_MATCH_RATIO);
        let ransac_enabled = params.ransac.unwrap_or(true);
        let ransac_threshold = params
            .ransac_threshold
            .unwrap_or(DEFAULT_RANSAC_REPROJ_THRESHOLD);
        let mode = params.mode.unwrap_or_else(|| "stitch".to_string());

        if mode != "stitch" && mode != "matches" {
            return json_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid `mode`=`{mode}`: expected `stitch` or `matches`"),
            );
        }
        if let Err(err) = validate_pipeline_params(octaves, scales, threshold) {
            return map_vision_error(err);
        }
        if let Err(err) = validate_match_params(ratio_threshold, ransac_threshold) {
            return map_vision_error(err);
        }

        let mut images = match read_multipart_images(multipart, 2).await {
            Ok(imgs) => imgs,
            Err(resp) => return resp,
        };
        let target_bytes = images.pop().expect("length 2 verified");
        let query_bytes = images.pop().expect("length 2 verified");

        match run_blocking_with_timeout(&state, "Image alignment timed out", move |_cancel| {
            let pair_opts = PairPipelineOptions {
                sift: SiftConfig::default()
                    .octaves(octaves)
                    .scales(scales)
                    .contrast_threshold(threshold),
                match_options: MatchOptions {
                    ratio: ratio_threshold,
                    mutual: false,
                },
                ransac_enabled,
                ransac: RansacOptions::default().reproj_threshold(ransac_threshold),
                enforce_pixel_cap: true,
            };
            let rgba = if mode == "matches" {
                let (canvas, _, _) =
                    run_visualize_matches_rgba(&query_bytes, &target_bytes, pair_opts)?;
                canvas
            } else {
                let (stitched, _) = run_align_images_rgba(&query_bytes, &target_bytes, pair_opts)?;
                stitched
            };
            encode_rgba_png(&rgba).map_err(VisionError::Runtime)
        })
        .await
        {
            Ok(png_bytes) => png_response(png_bytes),
            Err(resp) => resp,
        }
    }

    /// `GET /api/v1/catalog` — lists all indexed entries in the live SIFT search catalog.
    async fn catalog_list_handler(State(state): State<AppState>) -> Response {
        let state_clone = state.clone();
        match run_blocking_with_timeout(&state, "Catalog read timed out", move |_cancel| {
            let guard = state_clone
                .catalog
                .read()
                .map_err(|_| VisionError::Runtime("Catalog lock poisoned".to_string()))?;
            Ok(CatalogListResponse {
                catalog_size: guard.entries.len(),
                octaves: guard.octaves,
                scales: guard.scales,
                threshold: guard.threshold,
                entries: guard.list_summaries(),
            })
        })
        .await
        {
            Ok(resp) => (StatusCode::OK, Json(serde_json::json!(resp))).into_response(),
            Err(resp) => resp,
        }
    }

    /// `POST /api/v1/catalog/index?id=<NAME>` — indexes an uploaded image into the live SIFT catalog.
    async fn catalog_index_handler(
        State(state): State<AppState>,
        Query(params): Query<CatalogIndexParams>,
        multipart: Multipart,
    ) -> Response {
        let id = params.id.trim().to_string();
        if id.is_empty() {
            return json_error_response(
                StatusCode::BAD_REQUEST,
                "Query parameter `id` must not be empty",
            );
        }

        let images = match read_multipart_images(multipart, 1).await {
            Ok(imgs) => imgs,
            Err(resp) => return resp,
        };
        let raw_bytes = images.into_iter().next().expect("length 1 verified");
        let state_clone = state.clone();

        match run_blocking_with_timeout(&state, "Catalog indexing timed out", move |cancel| {
            let (octaves, scales, threshold) = {
                let guard = state_clone
                    .catalog
                    .read()
                    .map_err(|_| VisionError::Runtime("Catalog lock poisoned".to_string()))?;
                (guard.octaves, guard.scales, guard.threshold)
            };

            let extracted =
                run_extract_from_bytes(&raw_bytes, octaves, scales, threshold, true, true)?;
            if cancel.load(Ordering::Relaxed) {
                return Err(VisionError::InvalidInput("Request timed out".to_string()));
            }
            let features = extracted.features.unwrap_or_default();
            let feature_count = features.len();

            let catalog_size = {
                let mut guard = state_clone
                    .catalog
                    .write()
                    .map_err(|_| VisionError::Runtime("Catalog lock poisoned".to_string()))?;
                guard.upsert_entry(id.clone(), extracted.width, extracted.height, features);
                guard.entries.len()
            };

            if state_clone.catalog_path.is_some() {
                state_clone.catalog_save_notify.notify_one();
            }

            Ok(CatalogIndexResponse {
                id,
                width: extracted.width,
                height: extracted.height,
                feature_count,
                catalog_size,
            })
        })
        .await
        {
            Ok(resp) => (StatusCode::OK, Json(serde_json::json!(resp))).into_response(),
            Err(resp) => resp,
        }
    }

    /// `POST /api/v1/catalog/search` — searches the live SIFT catalog for geometric matches to an uploaded query image.
    async fn catalog_search_handler(
        State(state): State<AppState>,
        Query(params): Query<CatalogSearchParams>,
        multipart: Multipart,
    ) -> Response {
        let top_k = params.top_k.unwrap_or(DEFAULT_CATALOG_TOP_K);
        let shortlist_k = params.shortlist.unwrap_or(DEFAULT_CATALOG_SHORTLIST);
        let ratio = params.ratio.unwrap_or(DEFAULT_MATCH_RATIO);
        let ransac_threshold = params
            .ransac_threshold
            .unwrap_or(DEFAULT_RANSAC_REPROJ_THRESHOLD);

        if top_k == 0 || shortlist_k == 0 {
            return json_error_response(
                StatusCode::BAD_REQUEST,
                "`top_k` and `shortlist` must be at least 1",
            );
        }
        if let Err(err) = validate_match_params(ratio, ransac_threshold) {
            return map_vision_error(err);
        }

        let images = match read_multipart_images(multipart, 1).await {
            Ok(imgs) => imgs,
            Err(resp) => return resp,
        };
        let query_bytes = images.into_iter().next().expect("length 1 verified");
        let state_clone = state.clone();

        match run_blocking_with_timeout(&state, "Catalog search timed out", move |_cancel| {
            let guard = state_clone
                .catalog
                .read()
                .map_err(|_| VisionError::Runtime("Catalog lock poisoned".to_string()))?;
            run_catalog_search_from_bytes(
                &guard,
                &query_bytes,
                CatalogSearchOptions {
                    top_k,
                    shortlist_k,
                    ratio_threshold: ratio,
                    ransac_threshold,
                    enforce_pixel_cap: true,
                },
            )
        })
        .await
        {
            Ok(resp) => (StatusCode::OK, Json(serde_json::json!(resp))).into_response(),
            Err(resp) => resp,
        }
    }

    /// `DELETE /api/v1/catalog/{id}` — removes an indexed entry by ID from the live SIFT catalog.
    async fn catalog_delete_handler(
        State(state): State<AppState>,
        Path(id): Path<String>,
    ) -> Response {
        let trimmed = id.trim().to_string();
        if trimmed.is_empty() {
            return json_error_response(StatusCode::BAD_REQUEST, "Entry `id` must not be empty");
        }

        let (removed, catalog_size) = {
            let Ok(mut guard) = state.catalog.write() else {
                return json_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Catalog lock poisoned",
                );
            };
            let removed = guard.remove(&trimmed);
            let catalog_size = guard.entries.len();
            (removed, catalog_size)
        };

        if !removed {
            return json_error_response(
                StatusCode::NOT_FOUND,
                format!("Catalog entry `{trimmed}` not found"),
            );
        }

        if state.catalog_path.is_some() {
            state.catalog_save_notify.notify_one();
        }

        (
            StatusCode::OK,
            Json(serde_json::json!({
                "deleted": true,
                "id": trimmed,
                "catalog_size": catalog_size,
            })),
        )
            .into_response()
    }

    /// Waits for `Ctrl+C` termination signal to initiate graceful HTTP server drain.
    async fn shutdown_signal() {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received; draining in-flight requests");
    }

    pub async fn run_server(bind_addr_override: Option<String>) -> Result<(), String> {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .try_init();

        let opts = ServerOptions::default();
        let auth_enabled = opts.api_token.is_some();
        let (app, _state) = build_router_with_options(opts);
        let bind_addr = bind_addr_override
            .or_else(|| std::env::var("BIND_ADDR").ok())
            .unwrap_or_else(|| DEFAULT_BIND_ADDR.to_string());

        let listener = TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| format!("Failed to bind TCP listener on `{bind_addr}`: {e}"))?;

        tracing::info!(
            bind_addr = %bind_addr,
            auth_enabled = auth_enabled,
            "pure-sift HTTP server listening"
        );
        println!("pure-sift HTTP server listening on http://{bind_addr}");
        if let Some(port) = bind_addr.strip_prefix("0.0.0.0:") {
            println!("Interactive Web UI available at http://127.0.0.1:{port}/");
        } else {
            println!("Interactive Web UI available at http://{bind_addr}/");
        }

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| format!("Axum server terminated unexpectedly: {e}"))
    }
} // mod server

fn main() -> ExitCode {
    let cmd = match parse_cli_args(std::env::args().skip(1)) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("Error: {err}");
            return ExitCode::from(2);
        }
    };

    match cmd {
        CliCommand::Serve { bind_addr } => {
            #[cfg(feature = "server")]
            {
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(err) => {
                        eprintln!("Error initializing Tokio runtime: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                if let Err(err) = rt.block_on(server::run_server(bind_addr)) {
                    eprintln!("Error: {err}");
                    return ExitCode::FAILURE;
                }
                ExitCode::SUCCESS
            }
            #[cfg(not(feature = "server"))]
            {
                let _ = bind_addr;
                eprintln!(
                    "Error: `serve` requires the `server` Cargo feature. Rebuild with: `cargo run --release --features server -- serve`"
                );
                ExitCode::from(2)
            }
        }
        other => match execute_cli_to_stdout(&other) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err}");
                ExitCode::FAILURE
            }
        },
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::server::{
        ServerOptions, build_router, build_router_with_options, run_blocking_with_timeout,
    };

    async fn send_raw_http_request(addr: std::net::SocketAddr, req: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("Failed to connect to test HTTP server");
        stream
            .write_all(req.as_bytes())
            .await
            .expect("Failed to write HTTP request");
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .await
            .expect("Failed to read HTTP response");
        String::from_utf8_lossy(&buf).into_owned()
    }

    async fn send_raw_http_get(addr: std::net::SocketAddr, path: &str) -> String {
        let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        send_raw_http_request(addr, &req).await
    }

    #[tokio::test]
    async fn test_get_root_serves_embedded_web_ui_and_health() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");

        let server_task = tokio::spawn(async move {
            let _ = axum::serve(listener, build_router()).await;
        });

        let root_resp = send_raw_http_get(addr, "/").await;
        assert!(
            root_resp.starts_with("HTTP/1.1 200 OK"),
            "Expected HTTP 200 OK for GET /, got:\n{root_resp}"
        );
        assert!(
            root_resp
                .to_ascii_lowercase()
                .contains("content-type: text/html"),
            "Expected text/html content-type header"
        );
        assert!(
            root_resp
                .to_ascii_lowercase()
                .contains("x-request-id: req-"),
            "Expected x-request-id header"
        );
        assert!(root_resp.contains("pure-sift"));
        assert!(root_resp.contains("data-preset=\"medical\""));
        assert!(root_resp.contains("data-preset=\"tracking\""));
        assert!(root_resp.contains("data-preset=\"stitch\""));
        assert!(root_resp.contains("data-preset=\"catalog\""));
        assert!(root_resp.contains("/api/v1/visualize-features"));
        assert!(root_resp.contains("/api/v1/match-features"));
        assert!(root_resp.contains("/api/v1/align-images"));
        assert!(root_resp.contains("/api/v1/catalog/search"));

        let health_resp = send_raw_http_get(addr, "/health").await;
        assert!(health_resp.starts_with("HTTP/1.1 200 OK"));
        assert!(health_resp.contains("\"status\":\"ok\""));

        server_task.abort();
    }

    #[tokio::test]
    async fn test_api_token_bearer_auth_and_error_paths() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let opts = ServerOptions {
            max_body_bytes: 128,
            max_concurrent_jobs: 2,
            request_timeout: Duration::from_millis(50),
            api_token: Some("secret-token-123".to_string()),
            catalog_path: None,
        };
        let (router, state) = build_router_with_options(opts);
        let server_task = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        // 1. Missing bearer token on /api/v1/catalog -> 401
        let unauth = send_raw_http_get(addr, "/api/v1/catalog").await;
        assert!(
            unauth.starts_with("HTTP/1.1 401 Unauthorized"),
            "Expected 401 without token, got:\n{unauth}"
        );

        // 2. Valid bearer token on /api/v1/catalog -> 200
        let auth_ok = send_raw_http_request(
            addr,
            "GET /api/v1/catalog HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-token-123\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(
            auth_ok.starts_with("HTTP/1.1 200 OK"),
            "Expected 200 with valid token, got:\n{auth_ok}"
        );

        // 3. Oversized body (> 128 bytes) -> rejected before processing (413 Payload Too Large)
        let payload = format!(
            "--BOUND\r\nContent-Disposition: form-data; name=\"image\"; filename=\"big.png\"\r\nContent-Type: image/png\r\n\r\n{}\r\n--BOUND--\r\n",
            "x".repeat(256)
        );
        let oversized_req = format!(
            "POST /api/v1/process-features HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-token-123\r\nContent-Type: multipart/form-data; boundary=BOUND\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        let oversized_resp = send_raw_http_request(addr, &oversized_req).await;
        assert!(
            oversized_resp.starts_with("HTTP/1.1 413")
                || oversized_resp.contains("length limit exceeded"),
            "Expected 413 Payload Too Large or length limit rejection, got:\n{oversized_resp}"
        );

        // 4. Bad multipart payload -> 400 Bad Request
        let bad_mp_req = "POST /api/v1/process-features HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-token-123\r\nContent-Type: text/plain\r\nContent-Length: 4\r\nConnection: close\r\n\r\ntest";
        let bad_mp_resp = send_raw_http_request(addr, bad_mp_req).await;
        assert!(
            bad_mp_resp.starts_with("HTTP/1.1 400"),
            "Expected 400 Bad Request for non-multipart request, got:\n{bad_mp_resp}"
        );

        // 5. Timeout -> 504 Gateway Timeout and cooperative cancellation flag set
        let res = run_blocking_with_timeout(&state, "Timed out", |cancel| {
            std::thread::sleep(Duration::from_millis(120));
            assert!(cancel.load(std::sync::atomic::Ordering::Relaxed));
            Ok(())
        })
        .await;
        assert_eq!(
            res.unwrap_err().status(),
            axum::http::StatusCode::GATEWAY_TIMEOUT
        );

        // 6. DELETE /api/v1/catalog/{id} -> 404 when absent, 200 after inserting
        let del_missing = send_raw_http_request(
            addr,
            "DELETE /api/v1/catalog/nonexistent HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-token-123\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(
            del_missing.starts_with("HTTP/1.1 404"),
            "Expected 404 for missing catalog entry, got:\n{del_missing}"
        );

        {
            let mut guard = state.catalog.write().unwrap();
            guard.upsert_entry("sample.png", 100, 100, Vec::new());
        }
        let del_ok = send_raw_http_request(
            addr,
            "DELETE /api/v1/catalog/sample.png HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-token-123\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(
            del_ok.starts_with("HTTP/1.1 200 OK"),
            "Expected 200 OK deleting sample.png, got:\n{del_ok}"
        );
        assert!(del_ok.contains("\"deleted\":true"));
        assert_eq!(state.catalog.read().unwrap().entries.len(), 0);

        server_task.abort();
    }
}
