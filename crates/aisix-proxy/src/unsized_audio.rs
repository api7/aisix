//! Audio length from a WAV or FLAC file whose header leaves the length
//! unset.
//!
//! A writer that streams its output cannot seek back to fill in the
//! length, so it writes a placeholder: WAV's RIFF and `data` sizes become
//! `0xFFFFFFFF`, FLAC's `STREAMINFO` total-sample count becomes 0. That is
//! what OpenAI's `/v1/audio/speech` returns for `response_format=wav` and
//! `flac`, and what `ffmpeg` writes to a pipe. `lofty` rejects the WAV
//! ("no data chunk") and reports 0 s for the FLAC, so these readers take
//! over when it comes back empty.
//!
//! Headers only, as for the other probes: the WAV length is the bytes
//! after the `data` chunk header divided by the byte rate, the FLAC length
//! is read off the last frame's header. Nothing is decoded, every read is
//! bounds-checked, and anything that does not parse yields `None`.
//!
//! <http://soundfile.sapp.org/doc/WaveFormat/>,
//! <https://www.rfc-editor.org/rfc/rfc9639> (FLAC)

pub(crate) fn wav_duration_seconds(audio: &[u8]) -> Option<f64> {
    if audio.get(0..4)? != b"RIFF" || audio.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut byte_rate = None;
    let mut offset = 12usize;
    while let Some(header) = audio.get(offset..offset + 8) {
        let size = u32::from_le_bytes(header[4..8].try_into().ok()?) as usize;
        let body = offset + 8;
        match &header[0..4] {
            b"fmt " => {
                let rate = u32::from_le_bytes(audio.get(body + 8..body + 12)?.try_into().ok()?);
                byte_rate = Some(rate).filter(|r| *r > 0);
            }
            b"data" => {
                // An unset (or otherwise oversized) length means "to the end
                // of the file".
                let len = size.min(audio.len() - body);
                let seconds = len as f64 / f64::from(byte_rate?);
                return (seconds > 0.0).then_some(seconds);
            }
            _ => {}
        }
        offset = body.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

pub(crate) fn flac_duration_seconds(audio: &[u8]) -> Option<f64> {
    if audio.get(0..4)? != b"fLaC" {
        return None;
    }
    // The first metadata block is always STREAMINFO (34 bytes).
    let info = audio.get(8..8 + 34)?;
    if audio[4] & 0x7F != 0 {
        return None;
    }
    let min_block = u16::from_be_bytes([info[0], info[1]]);
    let max_block = u16::from_be_bytes([info[2], info[3]]);
    let sample_rate =
        (u32::from(info[10]) << 12) | (u32::from(info[11]) << 4) | (u32::from(info[12]) >> 4);
    if sample_rate == 0 {
        return None;
    }
    let declared = (u64::from(info[13] & 0x0F) << 32)
        | u64::from(u32::from_be_bytes([info[14], info[15], info[16], info[17]]));
    let total = if declared > 0 {
        declared
    } else {
        // The largest frame the block size, channel count and sample width
        // allow uncompressed (one extra bit per sample covers a stereo side
        // channel), tightened by STREAMINFO's maximum frame size when set.
        let channels = usize::from((info[12] >> 1) & 0x07) + 1;
        let bits = usize::from(((info[12] & 1) << 4) | (info[13] >> 4)) + 1;
        let bound = (usize::from(max_block) * channels * (bits + 1)).div_ceil(8) + 64;
        let declared_max =
            (usize::from(info[7]) << 16) | (usize::from(info[8]) << 8) | usize::from(info[9]);
        let max_frame = if declared_max == 0 {
            bound
        } else {
            declared_max.min(bound)
        };
        // A fixed-blocksize stream numbers frames, not samples, so its
        // position is only known when STREAMINFO names one valid block size.
        let fixed_block =
            (min_block == max_block && min_block >= 16).then_some(u64::from(min_block));
        last_frame_end_sample(audio, fixed_block, max_frame)?
    };
    Some(total as f64 / f64::from(sample_rate)).filter(|s| *s > 0.0)
}

/// The sample count the stream ends on, from the header of the frame that
/// ends exactly at EOF. A candidate counts only if its header CRC-8 and the
/// frame's CRC-16 (which runs to the end of the file) both match, so a sync
/// pattern inside compressed audio is never mistaken for a frame.
///
/// The search is bounded because the file is a caller upload: it starts no
/// further back than one frame can be long, and at most
/// `MAX_FRAME_CANDIDATES` headers get the CRC-16 check.
fn last_frame_end_sample(audio: &[u8], fixed_block: Option<u64>, max_frame: usize) -> Option<u64> {
    const MAX_FRAME_CANDIDATES: usize = 4;
    let body_end = audio.len().checked_sub(2)?;
    let tail_crc = u16::from_be_bytes([audio[body_end], audio[body_end + 1]]);
    let earliest = body_end.saturating_sub(max_frame);
    let mut checked = 0;
    for start in (earliest..body_end.saturating_sub(1)).rev() {
        if audio[start] != 0xFF || audio[start + 1] & 0xFE != 0xF8 {
            continue;
        }
        let Some(end_sample) = parse_frame_header(&audio[start..], fixed_block) else {
            continue;
        };
        if crc16_flac(&audio[start..body_end]) == tail_crc {
            return Some(end_sample);
        }
        checked += 1;
        if checked == MAX_FRAME_CANDIDATES {
            return None;
        }
    }
    None
}

/// The first sample after the frame whose valid header starts `frame`.
fn parse_frame_header(frame: &[u8], fixed_block: Option<u64>) -> Option<u64> {
    let variable = frame[1] & 1 == 1;
    let b2 = *frame.get(2)?;
    let b3 = *frame.get(3)?;
    let (block_code, rate_code) = (b2 >> 4, b2 & 0x0F);
    if block_code == 0 || rate_code == 0x0F || (b3 >> 1) & 0x07 == 3 || b3 & 1 != 0 {
        return None;
    }
    // The UTF-8-style coded frame (fixed blocking) or sample (variable)
    // number.
    let lead = *frame.get(4)?;
    let extra = match lead.leading_ones() {
        0 => 0,
        n @ 2..=7 => n as usize - 1,
        _ => return None,
    };
    let mut number = u64::from(if extra == 0 {
        lead
    } else {
        lead & (0x7F >> (extra + 1))
    });
    for i in 0..extra {
        let b = *frame.get(5 + i)?;
        if b & 0xC0 != 0x80 {
            return None;
        }
        number = (number << 6) | u64::from(b & 0x3F);
    }
    let mut pos = 5 + extra;
    let block = match block_code {
        1 => 192,
        2..=5 => 576 << (block_code - 2),
        6 => {
            pos += 1;
            u64::from(*frame.get(pos - 1)?) + 1
        }
        7 => {
            pos += 2;
            u64::from(u16::from_be_bytes([
                *frame.get(pos - 2)?,
                *frame.get(pos - 1)?,
            ])) + 1
        }
        _ => 256 << (block_code - 8),
    };
    pos += match rate_code {
        12 => 1,
        13 | 14 => 2,
        _ => 0,
    };
    if crc8_flac(frame.get(..pos)?) != *frame.get(pos)? {
        return None;
    }
    let first = if variable {
        number
    } else {
        number.checked_mul(fixed_block?)?
    };
    first.checked_add(block)
}

fn crc8_flac(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |mut crc, &b| {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
        crc
    })
}

fn crc16_flac(bytes: &[u8]) -> u16 {
    bytes.iter().fold(0u16, |crc, &b| {
        (crc << 8) ^ CRC16_TABLE[usize::from((crc >> 8) as u8 ^ b)]
    })
}

/// CRC-16 with FLAC's polynomial (0x8005), one entry per byte value.
const CRC16_TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    /// A streamed WAV as OpenAI's `/v1/audio/speech` and `ffmpeg` write it:
    /// RIFF and `data` sizes `0xFFFFFFFF`, with a `LIST` chunk ahead of the
    /// audio the way ffmpeg puts one.
    fn streamed_wav(seconds: usize) -> Vec<u8> {
        let (rate, data_len) = (24_000u32, 24_000 * 2 * seconds);
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&u32::MAX.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&3u32.to_le_bytes());
        wav.extend_from_slice(b"abc\0"); // odd size + pad byte
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&u32::MAX.to_le_bytes());
        wav.resize(wav.len() + data_len, 0);
        wav
    }

    #[test]
    fn a_streamed_wav_is_measured_to_the_end_of_the_file() {
        let seconds = super::wav_duration_seconds(&streamed_wav(5)).expect("measurable");
        assert!((seconds - 5.0).abs() < 1e-9, "got {seconds}");
        // And it is what the upload probe now reports, where `lofty`
        // alone found no `data` chunk.
        let probed = crate::audio::probe_audio_duration_seconds(&streamed_wav(5));
        assert_eq!(probed, Some(seconds));
    }

    #[test]
    fn a_wav_with_no_format_chunk_yields_nothing() {
        let mut wav = b"RIFF\xff\xff\xff\xffWAVEdata\xff\xff\xff\xff".to_vec();
        wav.resize(1000, 0);
        assert_eq!(super::wav_duration_seconds(&wav), None);
    }

    /// `ffmpeg -f flac pipe:1` — one second of 8 kHz mono — writes a
    /// STREAMINFO total-sample count of 0, as OpenAI's `response_format=flac`
    /// does, so the length comes off the last frame's header.
    const STREAMED_FLAC: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/streamed-1s.flac"
    ));

    #[test]
    fn a_streamed_flac_is_measured_off_its_last_frame() {
        assert_eq!(
            &STREAMED_FLAC[21..26],
            &[0xF0, 0, 0, 0, 0],
            "fixture declares no length"
        );
        assert_eq!(super::flac_duration_seconds(STREAMED_FLAC), Some(1.0));
        assert_eq!(
            crate::audio::probe_audio_duration_seconds(STREAMED_FLAC),
            Some(1.0)
        );
    }

    /// A fixed-blocksize stream numbers its frames, so without one valid
    /// block size in STREAMINFO the position cannot be computed — declined
    /// rather than billed as a single frame.
    #[test]
    fn a_streamed_flac_without_a_block_size_yields_nothing() {
        let mut flac = STREAMED_FLAC.to_vec();
        flac[8..10].copy_from_slice(&0u16.to_be_bytes());
        assert_eq!(super::flac_duration_seconds(&flac), None);
    }

    /// A FLAC cut short no longer ends on a frame whose CRC-16 matches, so
    /// it is not measured rather than measured wrong.
    #[test]
    fn a_truncated_streamed_flac_yields_nothing() {
        let cut = &STREAMED_FLAC[..STREAMED_FLAC.len() - 100];
        assert_eq!(super::flac_duration_seconds(cut), None);
    }
}
