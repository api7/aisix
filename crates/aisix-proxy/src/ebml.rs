//! Audio length from a WebM / Matroska container.
//!
//! `lofty` covers every audio container the transcription endpoints
//! accept except this one, and WebM is not a corner: `MediaRecorder`
//! produces `audio/webm;codecs=opus` by default, so it is what a browser
//! uploads. Without it a WebM transcription asked for as `text` / `srt` /
//! `vtt` would report no length and bill nothing (#1138).
//!
//! Only the header path is walked — EBML root → `Segment` → `Info` — to
//! read `Duration` (a float in `TimecodeScale` units) and `TimecodeScale`
//! (nanoseconds, default 1,000,000). Clusters are never entered and no
//! audio is decoded.
//!
//! The input is a caller upload, so the walk is total: every read is
//! bounds-checked, element sizes are clamped to the enclosing element,
//! and the number of elements visited is capped. Anything malformed,
//! truncated, or simply not Matroska yields `None`.
//!
//! <https://www.rfc-editor.org/rfc/rfc8794> (EBML),
//! <https://www.matroska.org/technical/elements.html>

/// EBML magic — the `EBML` root element every Matroska/WebM file opens
/// with. Checked by the caller to pick this reader over `lofty`.
pub(crate) const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];

const ID_SEGMENT: u64 = 0x1853_8067;
const ID_INFO: u64 = 0x1549_A966;
const ID_DURATION: u64 = 0x4489;
const ID_TIMECODE_SCALE: u64 = 0x002A_D7B1;

/// Matroska's default when `Info` omits `TimecodeScale`: one millisecond
/// expressed in nanoseconds.
const DEFAULT_TIMECODE_SCALE: u64 = 1_000_000;

/// Total elements the walk may visit before giving up. A well-formed
/// header reaches `Info` in a few dozen; the cap bounds the work a
/// crafted file can cause, since sizes come from the file itself.
const MAX_ELEMENTS: usize = 8_192;

/// One decoded EBML variable-length integer: its value and its width.
struct VInt {
    value: u64,
    width: usize,
}

/// Decode an EBML variable-length integer at `pos`.
///
/// `keep_marker` distinguishes the two uses: element IDs keep the length
/// marker (that is what makes `0x1A45DFA3` the literal id), while sizes
/// strip it to recover the magnitude.
fn read_vint(data: &[u8], pos: usize, keep_marker: bool) -> Option<VInt> {
    let first = *data.get(pos)?;
    if first == 0 {
        // A zero lead byte would mean a width above 8 bytes, which no
        // real element uses and which the loop below cannot describe.
        return None;
    }
    let width = first.leading_zeros() as usize + 1;
    if width > 8 || pos + width > data.len() {
        return None;
    }
    let mut value: u64 = if keep_marker {
        u64::from(first)
    } else {
        // Strip the width marker to recover the magnitude. At width 8 the
        // lead byte is all marker (0x01) and carries no value bits, and
        // `0xFF >> 8` would overflow the shift.
        let value_bits = if width >= 8 { 0 } else { 0xFFu8 >> width };
        u64::from(first & value_bits)
    };
    for byte in &data[pos + 1..pos + width] {
        value = (value << 8) | u64::from(*byte);
    }
    Some(VInt { value, width })
}

/// True when a size vint carries the all-ones "unknown size" pattern —
/// live-muxed WebM leaves `Segment` open this way, which is exactly what
/// a browser upload looks like.
fn is_unknown_size(size: &VInt) -> bool {
    let bits = 7 * size.width as u32;
    size.value == (1u64 << bits) - 1
}

/// Audio length in seconds, or `None` if this is not a readable
/// Matroska/WebM header.
pub(crate) fn duration_seconds(data: &[u8]) -> Option<f64> {
    if !data.starts_with(&EBML_MAGIC) {
        return None;
    }
    let mut budget = MAX_ELEMENTS;
    let segment = find_child(data, 0, data.len(), ID_SEGMENT, &mut budget)?;
    let info = find_child(data, segment.0, segment.1, ID_INFO, &mut budget)?;

    let mut duration: Option<f64> = None;
    let mut scale = DEFAULT_TIMECODE_SCALE;
    let mut pos = info.0;
    while pos < info.1 {
        if budget == 0 {
            return None;
        }
        budget -= 1;
        let (id, body_start, body_end, next) = read_element(data, pos, info.1)?;
        match id {
            ID_DURATION => duration = read_float(&data[body_start..body_end]),
            ID_TIMECODE_SCALE => scale = read_uint(&data[body_start..body_end]),
            _ => {}
        }
        pos = next?;
    }

    let ticks = duration?;
    if !ticks.is_finite() || ticks <= 0.0 || scale == 0 {
        return None;
    }
    let seconds = ticks * (scale as f64) / 1_000_000_000.0;
    (seconds.is_finite() && seconds > 0.0).then_some(seconds)
}

/// Scan the children of one master element for `wanted`, returning the
/// body range of the match.
fn find_child(
    data: &[u8],
    start: usize,
    end: usize,
    wanted: u64,
    budget: &mut usize,
) -> Option<(usize, usize)> {
    let mut pos = start;
    while pos < end {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let (id, body_start, body_end, next) = read_element(data, pos, end)?;
        if id == wanted {
            return Some((body_start, body_end));
        }
        // An unknown-size element that is not the one we want ends the
        // scan: its extent is undefined, so nothing after it can be
        // located without descending into it.
        pos = next?;
    }
    None
}

/// Read one element header at `pos`.
///
/// Returns its id, the bounds of its body, and where the next sibling
/// starts — `None` for the sibling when the element declared an unknown
/// size, since there is no defined end to skip to.
fn read_element(
    data: &[u8],
    pos: usize,
    parent_end: usize,
) -> Option<(u64, usize, usize, Option<usize>)> {
    let id = read_vint(data, pos, true)?;
    let size_pos = pos.checked_add(id.width)?;
    let size = read_vint(data, size_pos, false)?;
    let body_start = size_pos.checked_add(size.width)?;
    if body_start > parent_end {
        return None;
    }
    if is_unknown_size(&size) {
        // Body runs to the end of the enclosing element.
        return Some((id.value, body_start, parent_end, None));
    }
    // Clamp rather than reject: a truncated upload still has a readable
    // prefix, and the caller only needs `Info`, which precedes the media
    // data in every muxer's output.
    let body_end = body_start
        .checked_add(usize::try_from(size.value).ok()?)?
        .min(parent_end);
    Some((id.value, body_start, body_end, Some(body_end)))
}

/// EBML floats are 4 or 8 bytes, big-endian. Any other width is invalid.
fn read_float(body: &[u8]) -> Option<f64> {
    match body.len() {
        4 => Some(f64::from(f32::from_be_bytes(body.try_into().ok()?))),
        8 => Some(f64::from_be_bytes(body.try_into().ok()?)),
        _ => None,
    }
}

/// EBML unsigned ints are big-endian and up to 8 bytes wide.
fn read_uint(body: &[u8]) -> u64 {
    if body.is_empty() || body.len() > 8 {
        return DEFAULT_TIMECODE_SCALE;
    }
    body.iter().fold(0u64, |acc, b| (acc << 8) | u64::from(*b))
}

#[cfg(test)]
pub(crate) mod tests {
    /// The header of a real `ffmpeg -c:a libopus -f webm` file: an EBML
    /// header, then a `Segment` left at unknown size (how a live muxer —
    /// and a browser's MediaRecorder — writes it), a SeekHead whose
    /// SeekID payloads repeat the `Info` id, and finally the real `Info`
    /// carrying TimecodeScale 1,000,000 and Duration 7008.0.
    ///
    /// The SeekHead is the point of the fixture: a byte scan for the
    /// `Info` id finds those payloads first and reads garbage. Only a
    /// structural walk lands on the real element.
    fn webm_header(duration_ticks: f64) -> Vec<u8> {
        let mut out = Vec::new();
        // EBML header (id 1A45DFA3), one child: DocType "webm".
        let doctype = [0x42u8, 0x82, 0x84, b'w', b'e', b'b', b'm'];
        out.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);
        out.push(0x80 | doctype.len() as u8);
        out.extend_from_slice(&doctype);

        // Segment (id 18538067) with the unknown-size marker.
        out.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
        out.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

        // SeekHead (id 114D9B74) holding one Seek → SeekID = Info id.
        let seek_id_payload = [0x53u8, 0xAB, 0x84, 0x15, 0x49, 0xA9, 0x66];
        let mut seek = Vec::new();
        seek.extend_from_slice(&[0x4D, 0xBB]); // Seek
        seek.push(0x80 | seek_id_payload.len() as u8);
        seek.extend_from_slice(&seek_id_payload);
        out.extend_from_slice(&[0x11, 0x4D, 0x9B, 0x74]);
        out.push(0x80 | seek.len() as u8);
        out.extend_from_slice(&seek);

        // Info (id 1549A966): TimecodeScale + Duration.
        let mut info = Vec::new();
        info.extend_from_slice(&[0x2A, 0xD7, 0xB1, 0x83, 0x0F, 0x42, 0x40]); // 1_000_000
        info.extend_from_slice(&[0x44, 0x89, 0x88]); // Duration, 8-byte float
        info.extend_from_slice(&duration_ticks.to_be_bytes());
        out.extend_from_slice(&[0x15, 0x49, 0xA9, 0x66]);
        out.push(0x80 | info.len() as u8);
        out.extend_from_slice(&info);
        out
    }

    /// The fixture above is hand-built; this one is the first 264 bytes
    /// of a file `ffmpeg -c:a libopus -f webm` actually produced — EBML
    /// header, unknown-size Segment, SeekHead, and the real Info — cut
    /// immediately after Info so no audio data rides along. ffprobe
    /// reports 7.008s for the full file.
    #[rustfmt::skip]
    pub(crate) const REAL_FFMPEG_WEBM_HEADER: &[u8] = &[
        0x1A, 0x45, 0xDF, 0xA3, 0x9F, 0x42, 0x86, 0x81, 0x01, 0x42, 0xF7, 0x81, 0x01, 0x42, 0xF2, 0x81,
        0x04, 0x42, 0xF3, 0x81, 0x08, 0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6D, 0x42, 0x87, 0x81, 0x04,
        0x42, 0x85, 0x81, 0x02, 0x18, 0x53, 0x80, 0x67, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x14, 0x6A,
        0x11, 0x4D, 0x9B, 0x74, 0xBB, 0x4D, 0xBB, 0x8B, 0x53, 0xAB, 0x84, 0x15, 0x49, 0xA9, 0x66, 0x53,
        0xAC, 0x81, 0xA1, 0x4D, 0xBB, 0x8B, 0x53, 0xAB, 0x84, 0x16, 0x54, 0xAE, 0x6B, 0x53, 0xAC, 0x81,
        0xD8, 0x4D, 0xBB, 0x8C, 0x53, 0xAB, 0x84, 0x12, 0x54, 0xC3, 0x67, 0x53, 0xAC, 0x82, 0x01, 0x42,
        0x4D, 0xBB, 0x8D, 0x53, 0xAB, 0x84, 0x1C, 0x53, 0xBB, 0x6B, 0x53, 0xAC, 0x83, 0x01, 0x14, 0x42,
        0xEC, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x15, 0x49, 0xA9, 0x66, 0xB2, 0x2A, 0xD7, 0xB1, 0x83, 0x0F, 0x42, 0x40, 0x4D, 0x80, 0x8D,
        0x4C, 0x61, 0x76, 0x66, 0x36, 0x32, 0x2E, 0x31, 0x32, 0x2E, 0x31, 0x30, 0x32, 0x57, 0x41, 0x8D,
        0x4C, 0x61, 0x76, 0x66, 0x36, 0x32, 0x2E, 0x31, 0x32, 0x2E, 0x31, 0x30, 0x32, 0x44, 0x89, 0x88,
        0x40, 0xBB, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn reads_a_header_a_real_muxer_produced() {
        let got = super::duration_seconds(REAL_FFMPEG_WEBM_HEADER)
            .expect("a real webm header must yield a duration");
        assert!(
            (got - 7.008).abs() < 0.01,
            "ffprobe reports 7.008s for this file, reader said {got}"
        );
    }

    #[test]
    fn reads_duration_past_a_seekhead_and_unknown_size_segment() {
        let webm = webm_header(7008.0);
        let got = super::duration_seconds(&webm).expect("a webm header must yield a duration");
        assert!(
            (got - 7.008).abs() < 0.001,
            "7008 ticks × 1e6 ns should be 7.008s, got {got}"
        );
    }

    /// A 4-byte float is equally valid per the spec.
    #[test]
    fn reads_a_32_bit_duration() {
        let mut info = Vec::new();
        info.extend_from_slice(&[0x44, 0x89, 0x84]);
        info.extend_from_slice(&2500.0f32.to_be_bytes());
        let mut out = Vec::new();
        out.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3, 0x80]);
        out.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
        out.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        out.extend_from_slice(&[0x15, 0x49, 0xA9, 0x66]);
        out.push(0x80 | info.len() as u8);
        out.extend_from_slice(&info);
        assert_eq!(super::duration_seconds(&out), Some(2.5));
    }

    /// Not Matroska at all — the caller falls through to `lofty`.
    #[test]
    fn non_ebml_input_is_declined() {
        assert_eq!(super::duration_seconds(b"RIFF....WAVEfmt "), None);
        assert_eq!(super::duration_seconds(&[]), None);
    }

    /// Uploads are caller-controlled: truncation anywhere in the header
    /// must yield `None` rather than panic on a slice.
    #[test]
    fn every_truncation_of_a_valid_header_is_survivable() {
        let webm = webm_header(7008.0);
        for cut in 0..webm.len() {
            let _ = super::duration_seconds(&webm[..cut]);
        }
    }

    /// A declared size far past the end of the buffer must not read out
    /// of bounds — it is clamped to the parent, and the walk simply
    /// finds nothing.
    #[test]
    fn oversized_declarations_are_clamped_not_trusted() {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3, 0x80]);
        out.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
        // Segment claims ~4 GiB of body in a 20-byte file.
        out.extend_from_slice(&[0x10, 0xFF, 0xFF, 0xFF]);
        out.extend_from_slice(&[0x15, 0x49, 0xA9, 0x66, 0x80]);
        assert_eq!(super::duration_seconds(&out), None);
    }

    /// A zero or absent duration is not a cost basis.
    #[test]
    fn zero_and_missing_durations_are_declined() {
        assert_eq!(super::duration_seconds(&webm_header(0.0)), None);

        let mut out = Vec::new();
        out.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3, 0x80]);
        out.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
        out.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        out.extend_from_slice(&[0x15, 0x49, 0xA9, 0x66, 0x80]); // empty Info
        assert_eq!(super::duration_seconds(&out), None);
    }
}
