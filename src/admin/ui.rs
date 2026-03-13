use axum::response::{Html, IntoResponse, Response};
use http::Uri;
use rust_embed::Embed;

const UI_INDEX_HTML: &str = "index.html";

#[derive(Embed)]
#[folder = "ui/dist/"]
struct Assets;

pub async fn handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches("/ui/");

    if path.is_empty() || path == UI_INDEX_HTML {
        return index_html();
    }

    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            ([(http::header::CONTENT_TYPE, mime.as_ref())], content.data).into_response()
        }
        None => index_html(),
    }
}

fn index_html() -> Response {
    match Assets::get(UI_INDEX_HTML) {
        Some(content) => Html(content.data).into_response(),
        None => not_found(),
    }
}

fn not_found() -> Response {
    (http::StatusCode::NOT_FOUND, "404 not found").into_response()
}
