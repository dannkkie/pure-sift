#![forbid(unsafe_code)]

use std::process::ExitCode;

use pure_sift::cli::{CliCommand, execute_cli_to_stdout, parse_cli_args};

#[cfg(feature = "server")]
mod server {
    use std::{
        path::PathBuf,
        sync::{Arc, RwLock},
        time::{Duration, Instant},
    };

    use axum::{
        Json, Router,
        body::Body,
        extract::{DefaultBodyLimit, Multipart, Query, State},
        http::{HeaderValue, Request, StatusCode, header},
        middleware::{self, Next},
        response::{Html, IntoResponse, Response},
        routing::{get, post},
    };
    use pure_sift::{
        catalog::{
            CatalogEntrySummary, DEFAULT_CATALOG_SHORTLIST, DEFAULT_CATALOG_TOP_K, FeatureCatalog,
        },
        cli::{
            DEFAULT_OCTAVES, DEFAULT_SCALES, VisionError, run_align_images_rgba,
            run_catalog_search_from_bytes, run_extract_from_bytes, run_match_from_bytes,
            run_visualize_features_rgba, run_visualize_matches_rgba, validate_match_params,
            validate_pipeline_params,
        },
        pipeline::{
            DEFAULT_CONTRAST_THRESHOLD, DEFAULT_MATCH_RATIO, DEFAULT_RANSAC_REPROJ_THRESHOLD,
        },
        visualize::encode_rgba_png,
    };
    use serde::{Deserialize, Serialize};
    use tokio::net::TcpListener;

    /// Shared server state holding the live in-memory SIFT `FeatureCatalog` (optionally backed by `CATALOG_PATH`).
    #[derive(Clone)]
    pub struct AppState {
        pub catalog: Arc<RwLock<FeatureCatalog>>,
        pub catalog_path: Option<PathBuf>,
    }

    /// Default multipart upload body limit (16 MiB, overridable via `MAX_BODY_MB`).
    const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

    /// Default per-request processing timeout in seconds (overridable via `REQUEST_TIMEOUT_SECS`).
    const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

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
            VisionError::UnprocessableImage(_) => StatusCode::UNPROCESSABLE_ENTITY,
            VisionError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            VisionError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
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

        while let Ok(Some(field)) = multipart.next_field().await {
            match field.bytes().await {
                Ok(bytes) if !bytes.is_empty() => {
                    images.push(bytes.to_vec());
                    if images.len() == expected_count {
                        break;
                    }
                }
                Ok(_) => continue,
                Err(err) => {
                    return Err(json_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read multipart field bytes: {err}"),
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
    async fn run_blocking_with_timeout<T, F>(
        timeout_msg: &'static str,
        job: F,
    ) -> Result<T, Response>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, VisionError> + Send + 'static,
    {
        let task = tokio::task::spawn_blocking(job);
        match tokio::time::timeout(request_timeout(), task).await {
            Ok(Ok(Ok(val))) => Ok(val),
            Ok(Ok(Err(err))) => Err(map_vision_error(err)),
            Ok(Err(join_err)) => Err(json_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Blocking vision worker panicked: {join_err}"),
            )),
            Err(_) => Err(json_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                timeout_msg,
            )),
        }
    }

    /// Lightweight structured request logger middleware recording method, path, status, and duration.
    async fn request_logger_middleware(req: Request<Body>, next: Next) -> Response {
        let method = req.method().clone();
        let uri = req.uri().clone();
        let start = Instant::now();
        let response = next.run(req).await;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[HTTP] {} {} -> {} ({:.2} ms)",
            method,
            uri,
            response.status().as_u16(),
            elapsed_ms
        );
        response
    }

    /// Embedded single-file HTML/CSS/JS workbench served at `GET /` with zero external dependencies.
    const WEB_UI_HTML: &str = include_str!("web_ui.html");

    /// Builds the hardened Axum application router exposing the embedded Web UI, JSON APIs, visual PNGs,
    /// and catalog search endpoints.
    pub fn build_router() -> Router {
        let max_body_bytes = std::env::var("MAX_BODY_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(DEFAULT_MAX_BODY_BYTES);

        let catalog_path = std::env::var("CATALOG_PATH")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from);

        let catalog = if let Some(ref path) = catalog_path
            && path.exists()
        {
            FeatureCatalog::load_json(path).unwrap_or_else(|err| {
                eprintln!(
                    "Warning: failed to load CATALOG_PATH `{}`: {err}",
                    path.display()
                );
                FeatureCatalog::new(DEFAULT_OCTAVES, DEFAULT_SCALES, DEFAULT_CONTRAST_THRESHOLD)
            })
        } else {
            FeatureCatalog::new(DEFAULT_OCTAVES, DEFAULT_SCALES, DEFAULT_CONTRAST_THRESHOLD)
        };

        let state = AppState {
            catalog: Arc::new(RwLock::new(catalog)),
            catalog_path,
        };

        Router::new()
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
            .with_state(state)
            .layer(DefaultBodyLimit::max(max_body_bytes))
            .layer(middleware::from_fn(request_logger_middleware))
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

    fn request_timeout() -> Duration {
        let secs = std::env::var("REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS);
        Duration::from_secs(secs.max(1))
    }

    /// `POST /api/v1/process-features` — returns detected keypoints (and optional 128-D descriptors) as JSON.
    async fn process_features_handler(
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

        match run_blocking_with_timeout("Feature extraction timed out", move || {
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

        match run_blocking_with_timeout("Feature matching timed out", move || {
            run_match_from_bytes(
                &query_bytes,
                &target_bytes,
                octaves,
                scales,
                threshold,
                ratio_threshold,
                ransac_enabled,
                ransac_threshold,
                true,
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

        match run_blocking_with_timeout("Feature visualization timed out", move || {
            let (rendered, _) =
                run_visualize_features_rgba(&raw_bytes, octaves, scales, threshold, true)?;
            encode_rgba_png(&rendered).map_err(VisionError::Io)
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

        match run_blocking_with_timeout("Image alignment timed out", move || {
            let rgba = if mode == "matches" {
                let (canvas, _, _) = run_visualize_matches_rgba(
                    &query_bytes,
                    &target_bytes,
                    octaves,
                    scales,
                    threshold,
                    ratio_threshold,
                    ransac_enabled,
                    ransac_threshold,
                    true,
                )?;
                canvas
            } else {
                let (stitched, _) = run_align_images_rgba(
                    &query_bytes,
                    &target_bytes,
                    octaves,
                    scales,
                    threshold,
                    ratio_threshold,
                    ransac_threshold,
                    true,
                )?;
                stitched
            };
            encode_rgba_png(&rgba).map_err(VisionError::Io)
        })
        .await
        {
            Ok(png_bytes) => png_response(png_bytes),
            Err(resp) => resp,
        }
    }

    /// `GET /api/v1/catalog` — lists all indexed entries in the live SIFT search catalog.
    async fn catalog_list_handler(State(state): State<AppState>) -> Response {
        match run_blocking_with_timeout("Catalog read timed out", move || {
            let guard = state
                .catalog
                .read()
                .map_err(|_| VisionError::Io("Catalog lock poisoned".to_string()))?;
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

        match run_blocking_with_timeout("Catalog indexing timed out", move || {
            let (octaves, scales, threshold) = {
                let guard = state
                    .catalog
                    .read()
                    .map_err(|_| VisionError::Io("Catalog lock poisoned".to_string()))?;
                (guard.octaves, guard.scales, guard.threshold)
            };

            let extracted =
                run_extract_from_bytes(&raw_bytes, octaves, scales, threshold, true, true)?;
            let features = extracted.features.unwrap_or_default();
            let feature_count = features.len();

            let mut guard = state
                .catalog
                .write()
                .map_err(|_| VisionError::Io("Catalog lock poisoned".to_string()))?;
            guard.upsert_entry(id.clone(), extracted.width, extracted.height, features);
            let catalog_size = guard.entries.len();

            if let Some(ref path) = state.catalog_path {
                guard.save_json(path, false).map_err(VisionError::Io)?;
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

        match run_blocking_with_timeout("Catalog search timed out", move || {
            let guard = state
                .catalog
                .read()
                .map_err(|_| VisionError::Io("Catalog lock poisoned".to_string()))?;
            run_catalog_search_from_bytes(
                &guard,
                &query_bytes,
                top_k,
                shortlist_k,
                ratio,
                ransac_threshold,
                true,
            )
        })
        .await
        {
            Ok(resp) => (StatusCode::OK, Json(serde_json::json!(resp))).into_response(),
            Err(resp) => resp,
        }
    }

    /// Waits for `Ctrl+C` termination signal to initiate graceful HTTP server drain.
    async fn shutdown_signal() {
        let _ = tokio::signal::ctrl_c().await;
        println!("Shutdown signal received; draining in-flight requests...");
    }

    pub async fn run_server(bind_addr_override: Option<String>) -> Result<(), String> {
        let app = build_router();
        let bind_addr = bind_addr_override
            .or_else(|| std::env::var("BIND_ADDR").ok())
            .unwrap_or_else(|| "0.0.0.0:3000".to_string());

        let listener = TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| format!("Failed to bind TCP listener on `{bind_addr}`: {e}"))?;

        println!("Native Vision Feature Server listening on http://{bind_addr}");
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
                    "Error: `serve` requires the `server` Cargo feature (enabled by default)."
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::server::build_router;

    async fn send_raw_http_get(addr: std::net::SocketAddr, path: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("Failed to connect to test HTTP server");
        let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
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
        assert!(root_resp.contains("Native Vision Engine"));
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
}
