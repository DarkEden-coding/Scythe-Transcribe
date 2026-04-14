//! Keyword / phrase corrections for transcripts.

use regex::Regex;
use std::sync::LazyLock;

static SPLIT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*(?:->|=>|→|⇒|\t)\s*").expect("split pattern"));

/// Parse multiline replacement rules into (from, to) pairs.
#[must_use]
pub fn parse_replacement_spec(spec: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for raw_line in spec.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = SPLIT_PATTERN.splitn(line, 2).collect();
        if parts.len() != 2 {
            continue;
        }
        let src = parts[0].trim();
        let dst = parts[1].trim();
        if src.is_empty() {
            continue;
        }
        pairs.push((src.to_string(), dst.to_string()));
    }
    pairs
}

/// Build a Groq Whisper `prompt` string from keyword dictionary rules.
#[must_use]
pub fn groq_asr_prompt_from_replacement_spec(spec: &str, max_chars: usize) -> Option<String> {
    let pairs = parse_replacement_spec(spec);
    if pairs.is_empty() {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    let mut ordered = Vec::new();
    for (src, dst) in pairs {
        for part in [src.trim(), dst.trim()] {
            if part.is_empty() {
                continue;
            }
            let key = part.to_lowercase();
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            ordered.push(part.to_string());
        }
    }
    if ordered.is_empty() {
        return None;
    }
    let body = ordered.join(", ");
    let prefix = "When transcribing, use these terms and spellings where appropriate: ";
    let text = format!("{prefix}{body}");
    if text.len() <= max_chars {
        return Some(text);
    }
    let budget = max_chars.saturating_sub(prefix.len()).saturating_sub(1);
    if budget < 8 {
        return Some(prefix.trim().to_string());
    }
    let truncated: String = body.chars().take(budget).collect();
    let cut = truncated.rfind(", ").unwrap_or(0);
    let truncated = if cut > 0 {
        &truncated[..cut]
    } else {
        truncated.as_str()
    };
    Some(format!("{prefix}{truncated}…"))
}

/// Apply replacements longest-first.
#[must_use]
pub fn apply_replacements(text: &str, pairs: &[(String, String)]) -> String {
    if text.is_empty() || pairs.is_empty() {
        return text.to_string();
    }
    let mut indexed: Vec<(usize, &(String, String))> = pairs.iter().enumerate().collect();
    indexed.sort_by_key(|(i, (src, _))| (std::cmp::Reverse(src.len()), *i));
    let mut out = text.to_string();
    for (_, (src, dst)) in indexed {
        if src.is_empty() {
            continue;
        }
        out = out.replace(src, dst);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_comments_and_empty() {
        let spec = "# c\nfoo -> bar\n\nbaz => qux";
        let p = parse_replacement_spec(spec);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].0, "foo");
        assert_eq!(p[0].1, "bar");
    }

    #[test]
    fn apply_longest_first() {
        let pairs = parse_replacement_spec("ab -> X\nabc -> Y");
        let out = apply_replacements("abc", &pairs);
        assert_eq!(out, "Y");
    }
}
