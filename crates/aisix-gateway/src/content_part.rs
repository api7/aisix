//! The OpenAI chat content parts a native (non-OpenAI-shape) bridge has to
//! translate: `text`, `image_url`, `input_audio` and `file`.
//!
//! Every native bridge reads parts through [`parse_part`], so they agree on
//! what a part means and differ only in what their upstream can take. A
//! part an upstream cannot take is refused with a 400 naming the part and
//! what to send instead — never dropped, because a dropped image is a
//! request the model answers blind while the caller sees a 200.
//!
//! The gateway never fetches a caller-supplied URL. A URL either reaches
//! the upstream as a reference its own API resolves, or the request is
//! refused.

use serde_json::Value;

use crate::chat::ChatMessage;

/// One content part, as far as a translation needs to know.
#[derive(Debug, PartialEq, Eq)]
pub enum ContentPart<'a> {
    Text(&'a str),
    Media(Media<'a>),
    /// A part type this gateway does not translate. Carries the type name
    /// for the error message.
    Unsupported(&'a str),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Media<'a> {
    pub kind: MediaKind,
    pub source: MediaSource<'a>,
}

/// Which OpenAI part type the media came from. Only matters where the
/// media type alone cannot place it (an Anthropic URL source with no
/// recognisable extension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Audio,
    File,
}

#[derive(Debug, PartialEq, Eq)]
pub enum MediaSource<'a> {
    /// Base64 bytes carried in the request (`data:` URL, `input_audio`,
    /// `file.file_data`).
    Inline { mime_type: String, data: &'a str },
    /// A reference the upstream would have to resolve. `mime_type` is the
    /// caller's explicit `format` / `mime_type` / `content_type`, else the
    /// one the path's extension names, else `None`.
    Uri {
        uri: &'a str,
        mime_type: Option<String>,
    },
}

impl MediaSource<'_> {
    /// The URI's scheme, lower-cased (`https`, `gs`), for an error message
    /// or a scheme check. `None` for inline data.
    pub fn scheme(&self) -> Option<String> {
        match self {
            Self::Inline { .. } => None,
            Self::Uri { uri, .. } => uri
                .split_once("://")
                .map(|(scheme, _)| scheme.to_ascii_lowercase()),
        }
    }
}

/// Read one entry of an OpenAI `content` array. `Err` is a part this
/// gateway recognises but that is malformed (an `image_url` with no URL,
/// a `data:` URL that is not base64): the sentence to put in the 400.
pub fn parse_part(block: &Value) -> Result<ContentPart<'_>, String> {
    let ty = block.get("type").and_then(Value::as_str).unwrap_or("");
    match ty {
        "text" => Ok(ContentPart::Text(
            block.get("text").and_then(Value::as_str).unwrap_or(""),
        )),
        "image_url" => parse_image_url(block).map(ContentPart::Media),
        "input_audio" => parse_input_audio(block).map(ContentPart::Media),
        "file" => parse_file(block).map(ContentPart::Media),
        "" => Ok(ContentPart::Unsupported("(missing type)")),
        other => Ok(ContentPart::Unsupported(other)),
    }
}

/// The caller's explicit media type: `format`, then `mime_type`, then
/// `content_type`.
fn explicit_mime_type(obj: &Value) -> Option<String> {
    ["format", "mime_type", "content_type"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn parse_image_url(block: &Value) -> Result<Media<'_>, String> {
    let image_url = block.get("image_url");
    // Some clients send `image_url` as the bare URL string.
    let (url, explicit) = match image_url {
        Some(Value::String(url)) => (Some(url.as_str()), None),
        Some(obj @ Value::Object(_)) => (
            obj.get("url").and_then(Value::as_str),
            explicit_mime_type(obj),
        ),
        _ => (None, None),
    };
    let url = url
        .filter(|u| !u.is_empty())
        .ok_or("an image_url part must carry image_url.url")?;
    Ok(Media {
        kind: MediaKind::Image,
        source: source_from_url(url, explicit)?,
    })
}

fn parse_input_audio(block: &Value) -> Result<Media<'_>, String> {
    let audio = block.get("input_audio");
    let data = audio.and_then(|a| a.get("data")).and_then(Value::as_str);
    let format = audio.and_then(|a| a.get("format")).and_then(Value::as_str);
    let (Some(data), Some(format)) = (data, format) else {
        return Err(
            "an input_audio part must carry input_audio.data and input_audio.format".into(),
        );
    };
    let mime_type = if format.contains('/') {
        format.to_string()
    } else {
        format!("audio/{format}")
    };
    Ok(Media {
        kind: MediaKind::Audio,
        source: MediaSource::Inline { mime_type, data },
    })
}

fn parse_file(block: &Value) -> Result<Media<'_>, String> {
    let file = block
        .get("file")
        .ok_or("a file part must carry a file object")?;
    let explicit = explicit_mime_type(file);
    let filename = file.get("filename").and_then(Value::as_str);
    let source = if let Some(data) = file
        .get("file_data")
        .and_then(Value::as_str)
        .filter(|d| !d.is_empty())
    {
        if data.starts_with("data:") {
            source_from_url(data, explicit)?
        } else {
            // Bare base64: the media type has to come from somewhere else.
            let mime_type = explicit
                .or_else(|| filename.and_then(mime_type_from_path).map(str::to_string))
                .ok_or(
                    "a file part whose file_data is bare base64 needs file.format \
                     or a filename with a recognisable extension",
                )?;
            MediaSource::Inline { mime_type, data }
        }
    } else if let Some(id) = file
        .get("file_id")
        .and_then(Value::as_str)
        .filter(|d| !d.is_empty())
    {
        source_from_url(id, explicit)?
    } else {
        return Err("a file part must carry file.file_data or file.file_id".into());
    };
    Ok(Media {
        kind: MediaKind::File,
        source,
    })
}

/// A `data:` URL becomes inline bytes; anything else is a reference.
fn source_from_url(url: &str, explicit: Option<String>) -> Result<MediaSource<'_>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        let mime_type = explicit.or_else(|| mime_type_from_path(url).map(str::to_string));
        return Ok(MediaSource::Uri {
            uri: url,
            mime_type,
        });
    };
    let (header, data) = rest
        .split_once(',')
        .ok_or("a data: URL must have the form data:<media type>;base64,<data>")?;
    let mut params = header.split(';');
    let declared = params.next().unwrap_or("");
    if !params.any(|p| p.eq_ignore_ascii_case("base64")) {
        return Err("a data: URL must be base64-encoded (data:<media type>;base64,<data>)".into());
    }
    let mime_type = explicit
        .or_else(|| (!declared.is_empty()).then(|| declared.to_string()))
        .ok_or("a data: URL must name its media type (data:<media type>;base64,<data>)")?;
    Ok(MediaSource::Inline { mime_type, data })
}

/// The media type a URL path's extension names, for the formats the
/// multimodal upstreams accept. Query and fragment are ignored.
pub fn mime_type_from_path(url: &str) -> Option<&'static str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let name = path.rsplit('/').next().unwrap_or(path);
    let (_, ext) = name.rsplit_once('.')?;
    Some(match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "heic" => "image/heic",
        "heif" => "image/heif",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "html" | "htm" => "text/html",
        "md" => "text/markdown",
        "wav" => "audio/wav",
        "mp3" => "audio/mp3",
        "aac" => "audio/aac",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "m4a" => "audio/m4a",
        "mp4" => "video/mp4",
        "mov" => "video/mov",
        "webm" => "video/webm",
        "mpeg" | "mpg" => "video/mpeg",
        "avi" => "video/avi",
        "wmv" => "video/wmv",
        "flv" => "video/x-flv",
        "3gp" => "video/3gpp",
        _ => return None,
    })
}

/// The 400 sentence for a part an upstream cannot take: which message and
/// part, and why. `reason` is the bridge's own explanation, ending in what
/// the caller can send instead.
pub fn part_error(message_index: usize, part_index: usize, reason: &str) -> String {
    format!("messages[{message_index}].content[{part_index}]: {reason}")
}

/// A tool result reaches every native upstream as text, so a non-text
/// part in one is refused rather than dropped (an OpenAI-shape upstream
/// refuses it too).
pub fn check_tool_result_parts(message: &ChatMessage, message_index: usize) -> Result<(), String> {
    for (part_index, block) in message.content_blocks.iter().flatten().enumerate() {
        if let ContentPart::Text(_) =
            parse_part(block).map_err(|r| part_error(message_index, part_index, &r))?
        {
            continue;
        }
        let ty = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("(missing type)");
        return Err(part_error(
            message_index,
            part_index,
            &format!(
                "a tool result can carry only text parts, not {ty:?}; \
                 send the media in a user message instead"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn data_url_becomes_inline_bytes_with_its_media_type() {
        let block = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBOR"}});
        assert_eq!(
            parse_part(&block).unwrap(),
            ContentPart::Media(Media {
                kind: MediaKind::Image,
                source: MediaSource::Inline {
                    mime_type: "image/png".into(),
                    data: "iVBOR"
                },
            })
        );
    }

    #[test]
    fn remote_url_is_a_reference_typed_by_format_then_extension() {
        let typed =
            json!({"type":"image_url","image_url":{"url":"gs://b/cat","format":"image/webp"}});
        let by_ext = json!({"type":"image_url","image_url":"https://x.example/a/cat.JPG?sig=1"});
        let untyped = json!({"type":"image_url","image_url":{"url":"https://x.example/cat"}});
        fn uri(b: &Value) -> MediaSource<'_> {
            match parse_part(b).unwrap() {
                ContentPart::Media(Media { source, .. }) => source,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(
            uri(&typed),
            MediaSource::Uri {
                uri: "gs://b/cat",
                mime_type: Some("image/webp".into())
            }
        );
        assert_eq!(
            uri(&by_ext),
            MediaSource::Uri {
                uri: "https://x.example/a/cat.JPG?sig=1",
                mime_type: Some("image/jpeg".into())
            }
        );
        assert_eq!(
            uri(&untyped),
            MediaSource::Uri {
                uri: "https://x.example/cat",
                mime_type: None
            }
        );
    }

    #[test]
    fn input_audio_and_file_data_are_inline() {
        let audio = json!({"type":"input_audio","input_audio":{"data":"UklG","format":"wav"}});
        let file = json!({"type":"file","file":{"file_data":"data:application/pdf;base64,JVBE","filename":"a.pdf"}});
        let bare = json!({"type":"file","file":{"file_data":"JVBE","filename":"a.pdf"}});
        for (block, mime) in [
            (audio, "audio/wav"),
            (file, "application/pdf"),
            (bare, "application/pdf"),
        ] {
            match parse_part(&block).unwrap() {
                ContentPart::Media(Media {
                    source: MediaSource::Inline { mime_type, .. },
                    ..
                }) => assert_eq!(mime_type, mime),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn malformed_parts_are_errors_and_unknown_types_are_reported() {
        for bad in [
            json!({"type":"image_url","image_url":{}}),
            json!({"type":"image_url","image_url":{"url":"data:image/png,raw"}}),
            json!({"type":"image_url","image_url":{"url":"data:;base64,xx"}}),
            json!({"type":"input_audio","input_audio":{"data":"x"}}),
            json!({"type":"file","file":{}}),
            json!({"type":"file","file":{"file_data":"JVBE"}}),
        ] {
            assert!(parse_part(&bad).is_err(), "{bad}");
        }
        assert_eq!(
            parse_part(&json!({"type":"video_url","video_url":{"url":"x"}})).unwrap(),
            ContentPart::Unsupported("video_url")
        );
    }
}
