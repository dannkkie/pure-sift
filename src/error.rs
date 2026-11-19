#![forbid(unsafe_code)]

use thiserror::Error;

/// Structured error type shared across the library, CLI, and optional HTTP server.
#[derive(Debug, Error)]
pub enum VisionError {
    /// Invalid user parameters or CLI flags.
    #[error("{0}")]
    InvalidInput(String),

    /// Input image could not be decoded or geometric estimation failed.
    #[error("{0}")]
    UnprocessableImage(String),

    /// Image decoding or encoding failure preserving the underlying [`image::ImageError`].
    #[error("{context}: {source}")]
    ImageDecode {
        /// Human-readable operation context.
        context: String,
        /// Underlying image decoder/encoder error.
        #[source]
        source: image::ImageError,
    },

    /// Input image dimensions or upload payload exceeded configured limits.
    #[error("{0}")]
    PayloadTooLarge(String),

    /// Filesystem or stream I/O failure preserving the underlying [`std::io::Error`].
    #[error("{context}: {source}")]
    Io {
        /// Human-readable operation context.
        context: String,
        /// Underlying OS I/O error.
        #[source]
        source: std::io::Error,
    },

    /// JSON serialization failure preserving the underlying [`serde_json::Error`].
    #[error("{context}: {source}")]
    Json {
        /// Human-readable operation context.
        context: String,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// General runtime or catalog error without a structured OS source error.
    #[error("{0}")]
    Runtime(String),
}

impl VisionError {
    /// Constructs a [`VisionError::Io`] with contextual description and structured cause.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Constructs a [`VisionError::ImageDecode`] with contextual description and structured cause.
    pub fn image_decode(context: impl Into<String>, source: image::ImageError) -> Self {
        Self::ImageDecode {
            context: context.into(),
            source,
        }
    }

    /// Constructs a [`VisionError::Json`] with contextual description and structured cause.
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    /// Prepends `prefix` to the error context while preserving any structured `#[source]` cause.
    pub fn with_prefix(self, prefix: &str) -> Self {
        match self {
            Self::InvalidInput(m) => Self::InvalidInput(format!("{prefix}: {m}")),
            Self::UnprocessableImage(m) => Self::UnprocessableImage(format!("{prefix}: {m}")),
            Self::ImageDecode { context, source } => Self::ImageDecode {
                context: format!("{prefix}: {context}"),
                source,
            },
            Self::PayloadTooLarge(m) => Self::PayloadTooLarge(format!("{prefix}: {m}")),
            Self::Io { context, source } => Self::Io {
                context: format!("{prefix}: {context}"),
                source,
            },
            Self::Json { context, source } => Self::Json {
                context: format!("{prefix}: {context}"),
                source,
            },
            Self::Runtime(m) => Self::Runtime(format!("{prefix}: {m}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io;

    use super::*;

    #[test]
    fn test_vision_error_preserves_structured_sources_and_prefix() {
        let raw_io = io::Error::new(io::ErrorKind::NotFound, "missing file");
        let err = VisionError::io("Failed to read image", raw_io).with_prefix("Query image");
        assert!(
            err.source().is_some(),
            "Io variant must preserve Error::source"
        );
        assert!(
            err.to_string()
                .contains("Query image: Failed to read image")
        );

        let json_err = serde_json::from_str::<serde_json::Value>("{bad").unwrap_err();
        let err = VisionError::json("Failed to serialize JSON", json_err);
        assert!(
            err.source().is_some(),
            "Json variant must preserve Error::source"
        );
    }
}
