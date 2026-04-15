//! Speech-aware WAV splitting for parallel Groq ASR.
//!
//! Fixed time cuts can bisect words; we refine each nominal boundary toward a
//! short window with minimum RMS energy (natural pause / breath).

use std::io::Cursor;

use hound::SampleFormat;

/// Split WAV into chunks of roughly `chunk_sec`, refining each cut to sit in a
/// low-energy gap within ±`search_sec` of the nominal boundary.
pub fn split_wav_for_parallel_groq(
    raw: &[u8],
    chunk_sec: f64,
    search_sec: f64,
    min_chunk_sec: f64,
) -> Result<Vec<Vec<u8>>, String> {
    if chunk_sec <= 0.0 {
        return Ok(vec![raw.to_vec()]);
    }
    let mut reader = hound::WavReader::new(Cursor::new(raw)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err("WAV has zero channels".to_string());
    }
    let sample_rate = spec.sample_rate;
    let total_frames = reader.duration();
    if total_frames == 0 {
        return Err("WAV has zero frames".to_string());
    }

    let chunk_frames = (chunk_sec * f64::from(sample_rate)).floor().max(1.0) as u32;
    if total_frames <= chunk_frames {
        return Ok(vec![raw.to_vec()]);
    }

    let min_chunk_frames = (min_chunk_sec * f64::from(sample_rate)).floor().max(1.0) as u32;
    let min_chunk_frames = min_chunk_frames.min(chunk_frames.saturating_sub(1).max(1));

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let samples: Result<Vec<i16>, _> = reader.samples().collect();
            let samples = samples.map_err(|e| e.to_string())?;
            split_pcm_i16_speech_aware(
                &spec,
                &samples,
                channels,
                total_frames,
                sample_rate,
                chunk_frames,
                search_sec,
                min_chunk_frames,
            )
        }
        (SampleFormat::Float, 32) => {
            let samples: Result<Vec<f32>, _> = reader.samples().collect();
            let samples = samples.map_err(|e| e.to_string())?;
            split_pcm_f32_speech_aware(
                &spec,
                &samples,
                channels,
                total_frames,
                sample_rate,
                chunk_frames,
                search_sec,
                min_chunk_frames,
            )
        }
        _ => Err(
            "WAV format not supported for parallel ASR (need PCM 16-bit int or 32-bit float)"
                .to_string(),
        ),
    }
}

fn mono_from_i16_interleaved(samples: &[i16], channels: usize, frames: usize) -> Vec<f32> {
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let base = f * channels;
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += samples[base + c] as f32;
        }
        mono.push(acc / (32768.0 * channels as f32));
    }
    mono
}

fn mono_from_f32_interleaved(samples: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let base = f * channels;
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += samples[base + c];
        }
        mono.push(acc / channels as f32);
    }
    mono
}

/// RMS per short time bin (typically 12–15 ms) for pause search.
fn compute_smoothed_rms_bins(mono: &[f32], sample_rate: u32) -> (Vec<f32>, usize) {
    let bin_len = ((sample_rate as usize) * 12 / 1000).max(48);
    let mut bins = Vec::new();
    for chunk in mono.chunks(bin_len) {
        if chunk.is_empty() {
            break;
        }
        let e = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
        bins.push(e.sqrt());
    }
    if bins.len() >= 3 {
        let mut s = Vec::with_capacity(bins.len());
        s.push(bins[0]);
        for i in 1..bins.len() - 1 {
            s.push((bins[i - 1] + bins[i] + bins[i + 1]) / 3.0);
        }
        s.push(*bins.last().expect("len >= 3"));
        (s, bin_len)
    } else {
        (bins, bin_len)
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_boundary_frame(
    ideal_end_frame: u32,
    prev_end_frame: u32,
    total_frames: u32,
    sample_rate: u32,
    search_sec: f64,
    min_chunk_frames: u32,
    rms_bins: &[f32],
    bin_len: usize,
) -> u32 {
    let sr = f64::from(sample_rate);
    let search_frames = (search_sec * sr).ceil().max(1.0) as u32;

    let lo = ideal_end_frame
        .saturating_sub(search_frames)
        .max(prev_end_frame.saturating_add(min_chunk_frames));

    let hi = (ideal_end_frame + search_frames).min(
        total_frames
            .saturating_sub(min_chunk_frames)
            .max(prev_end_frame.saturating_add(min_chunk_frames)),
    );

    if lo >= hi {
        return ideal_end_frame.clamp(
            prev_end_frame + min_chunk_frames,
            total_frames.saturating_sub(min_chunk_frames),
        );
    }

    let lo_bin = lo as usize / bin_len;
    let hi_bin = (hi as usize / bin_len).min(rms_bins.len().saturating_sub(1));
    if lo_bin >= hi_bin || rms_bins.is_empty() {
        return ideal_end_frame.clamp(lo, hi);
    }

    let mut best_bin = lo_bin;
    let mut best = f32::MAX;
    for (b, v) in rms_bins.iter().enumerate().take(hi_bin + 1).skip(lo_bin) {
        if *v < best {
            best = *v;
            best_bin = b;
        }
    }

    let mut split = ((best_bin + 1) * bin_len).min(total_frames as usize) as u32;
    split = split.max(prev_end_frame + min_chunk_frames).min(hi);
    split
}

fn build_speech_aware_boundaries(
    total_frames: u32,
    sample_rate: u32,
    chunk_frames: u32,
    search_sec: f64,
    min_chunk_frames: u32,
    mono: &[f32],
) -> Vec<u32> {
    let (rms_bins, bin_len) = compute_smoothed_rms_bins(mono, sample_rate);
    if rms_bins.is_empty() {
        return vec![0, total_frames];
    }

    let mut boundaries = vec![0u32];
    let mut prev = 0u32;

    loop {
        let remaining = total_frames.saturating_sub(prev);
        if remaining <= chunk_frames {
            break;
        }

        let ideal_end = prev.saturating_add(chunk_frames).min(total_frames);
        if ideal_end >= total_frames.saturating_sub(min_chunk_frames) {
            break;
        }

        let split = refine_boundary_frame(
            ideal_end,
            prev,
            total_frames,
            sample_rate,
            search_sec,
            min_chunk_frames,
            &rms_bins,
            bin_len,
        );

        if split <= prev {
            break;
        }
        boundaries.push(split);
        prev = split;
    }

    if boundaries.last().copied() != Some(total_frames) {
        boundaries.push(total_frames);
    }

    boundaries
}

#[allow(clippy::too_many_arguments)]
fn split_pcm_i16_speech_aware(
    spec: &hound::WavSpec,
    samples: &[i16],
    channels: usize,
    total_frames: u32,
    sample_rate: u32,
    chunk_frames: u32,
    search_sec: f64,
    min_chunk_frames: u32,
) -> Result<Vec<Vec<u8>>, String> {
    let frames = total_frames as usize;
    let expected = frames * channels;
    if samples.len() != expected {
        return Err("WAV sample count mismatch".to_string());
    }

    let mono = mono_from_i16_interleaved(samples, channels, frames);
    let boundaries = build_speech_aware_boundaries(
        total_frames,
        sample_rate,
        chunk_frames,
        search_sec,
        min_chunk_frames,
        &mono,
    );

    wav_chunks_from_i16(spec, samples, channels, &boundaries)
}

#[allow(clippy::too_many_arguments)]
fn split_pcm_f32_speech_aware(
    spec: &hound::WavSpec,
    samples: &[f32],
    channels: usize,
    total_frames: u32,
    sample_rate: u32,
    chunk_frames: u32,
    search_sec: f64,
    min_chunk_frames: u32,
) -> Result<Vec<Vec<u8>>, String> {
    let frames = total_frames as usize;
    let expected = frames * channels;
    if samples.len() != expected {
        return Err("WAV sample count mismatch".to_string());
    }

    let mono = mono_from_f32_interleaved(samples, channels, frames);
    let boundaries = build_speech_aware_boundaries(
        total_frames,
        sample_rate,
        chunk_frames,
        search_sec,
        min_chunk_frames,
        &mono,
    );

    wav_chunks_from_f32(spec, samples, channels, &boundaries)
}

fn wav_bytes_from_i16(spec: &hound::WavSpec, samples: &[i16]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, *spec).map_err(|e| e.to_string())?;
    for s in samples {
        writer.write_sample(*s).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

fn wav_bytes_from_f32(spec: &hound::WavSpec, samples: &[f32]) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, *spec).map_err(|e| e.to_string())?;
    for s in samples {
        writer.write_sample(*s).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

fn wav_chunks_from_i16(
    spec: &hound::WavSpec,
    samples: &[i16],
    channels: usize,
    boundaries: &[u32],
) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::with_capacity(boundaries.len().saturating_sub(1));
    for w in boundaries.windows(2) {
        let start = w[0] as usize * channels;
        let end = w[1] as usize * channels;
        out.push(wav_bytes_from_i16(spec, &samples[start..end])?);
    }
    Ok(out)
}

fn wav_chunks_from_f32(
    spec: &hound::WavSpec,
    samples: &[f32],
    channels: usize,
    boundaries: &[u32],
) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::with_capacity(boundaries.len().saturating_sub(1));
    for w in boundaries.windows(2) {
        let start = w[0] as usize * channels;
        let end = w[1] as usize * channels;
        out.push(wav_bytes_from_f32(spec, &samples[start..end])?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};
    use std::io::Cursor;

    fn sine_frame(freq_hz: f32, sr: u32, frame: usize) -> f32 {
        (2.0 * std::f32::consts::PI * freq_hz * frame as f32 / sr as f32).sin() * 0.2
    }

    /// High-energy "speech" then silence then speech; nominal cut in the silence.
    #[test]
    fn refinement_prefers_quiet_region() {
        let sr = 16_000u32;
        let spec = WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut samples: Vec<i16> = Vec::new();
        let speech_a = sr as usize * 3;
        let pause = sr as usize / 4;
        let speech_b = sr as usize * 3;
        for i in 0..speech_a {
            samples.push((sine_frame(220.0, sr, i) * 30000.0) as i16);
        }
        samples.extend(std::iter::repeat(0i16).take(pause));
        for i in 0..speech_b {
            samples.push((sine_frame(330.0, sr, i) * 30000.0) as i16);
        }

        let mut cursor = Cursor::new(Vec::new());
        let mut w = WavWriter::new(&mut cursor, spec).unwrap();
        for s in &samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
        let _wav = cursor.into_inner();

        let mono = mono_from_i16_interleaved(&samples, 1, samples.len());
        let (rms, _bin_len) = compute_smoothed_rms_bins(&mono, sr);
        assert!(!rms.is_empty());

        let total_frames = samples.len() as u32;
        let chunk_frames = sr * 3;
        let boundaries =
            build_speech_aware_boundaries(total_frames, sr, chunk_frames, 0.9, sr / 2, &mono);
        assert!(boundaries.len() >= 2);
        let split = boundaries[1];
        let pause_center = speech_a + pause / 2;
        let dist = (split as isize - pause_center as isize).unsigned_abs();
        assert!(
            dist < sr as usize / 3,
            "split {split} too far from pause near {pause_center}"
        );
    }
}
