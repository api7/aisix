//! Embed the Vite-built admin UI into the binary at compile time.
//!
//! Vite emits to `crates/aisix-admin/ui-dist/`. `rust-embed` walks
//! that directory at compile time and pulls every file in. Missing
//! dist (no UI build yet) is tolerated — `Asset::iter()` returns an
//! empty iterator and the handler returns 404 for every UI request.
//!
//! The router mounts assets at `/ui/*` and a thin redirect at `/ui`
//! that points at `/ui/index.html` so a bare visit lands on the
//! single-page app.

use axum::body::Body;
use axum::extract::Path;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "ui-dist/"]
struct Asset;

/// Handler for `GET /ui` — redirect to the index so SPA routing works
/// against a clean URL bar.
pub async fn ui_root() -> Redirect {
    Redirect::permanent("/ui/index.html")
}

/// Handler for `GET /ui/*path` — serve the embedded asset, falling back
/// to `index.html` for unknown paths so the SPA's client-side router
/// can take over (typical `try_files` pattern).
pub async fn ui_asset(Path(path): Path<String>) -> Response {
    serve(&path).unwrap_or_else(|| serve("index.html").unwrap_or_else(not_found))
}

fn serve(path: &str) -> Option<Response> {
    let asset = Asset::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(asset.data.into_owned()))
        .ok()?;
    if let Ok(value) = HeaderValue::from_str(mime.as_ref()) {
        response.headers_mut().insert(header::CONTENT_TYPE, value);
    }
    Some(response)
}

fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        "admin UI is not bundled in this binary build",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// rust-embed compiles the `ui-dist/` folder into the binary at
    /// build time. CI runs the UI build before the Rust build, so the
    /// embed is non-empty in CI artefacts; on a developer machine that
    /// hasn't run `pnpm build` yet, it's empty — both are valid states
    /// and the handler degrades gracefully.
    #[tokio::test]
    async fn ui_root_redirects_to_index() {
        let resp = ui_root().await.into_response();
        assert_eq!(resp.status(), StatusCode::PERMANENT_REDIRECT);
        let location = resp
            .headers()
            .get(header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(location, "/ui/index.html");
    }

    #[tokio::test]
    async fn ui_asset_returns_404_for_missing_when_no_index() {
        // We can't assume index.html exists in a dev build with no
        // ui-dist; only assert the handler doesn't panic on missing
        // paths and returns *some* HTTP response.
        let resp = ui_asset(Path("definitely-not-here.css".into())).await;
        let status = resp.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::NOT_FOUND,
            "unexpected status: {status}",
        );
    }
}
