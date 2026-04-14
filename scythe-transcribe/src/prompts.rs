//! Shared prompt text and output sentinels.

pub const OPENROUTER_TRANSCRIPTION_INSTRUCTION: &str = concat!(
    "Transcribe this audio accurately. If no words are in the audio, reply with exactly None. ",
    "Otherwise reply with only the transcript."
);

pub const OPENROUTER_TRANSCRIPTION_NONE_OUTPUT: &str = "None";
