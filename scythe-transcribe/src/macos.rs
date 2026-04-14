//! macOS privacy helpers (Accessibility, Input Monitoring, Microphone).

#[cfg(target_os = "macos")]
#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
    fn AXIsProcessTrustedWithOptions(options: core_foundation::dictionary::CFDictionaryRef) -> u8;
    static kAXTrustedCheckOptionPrompt: core_foundation::string::CFStringRef;
}

#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGPreflightListenEventAccess() -> bool;
    fn CGRequestListenEventAccess() -> bool;
}

#[cfg(target_os = "macos")]
#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {}

#[cfg(target_os = "macos")]
use objc::runtime::Object;
#[cfg(target_os = "macos")]
use objc::{class, msg_send, sel, sel_impl};

#[cfg(target_os = "macos")]
fn av_media_type_audio_string() -> *mut Object {
    use std::ffi::CString;
    // `AVMediaTypeAudio` (same as Swift `AVMediaType.audio`).
    let Ok(cstr) = CString::new("soun") else {
        return std::ptr::null_mut();
    };
    unsafe { msg_send![class!(NSString), stringWithUTF8String: cstr.as_ptr()] }
}

/// `AVAuthorizationStatus` from AVFoundation (macOS 10.14+).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum MicrophoneAuthorization {
    NotDetermined,
    Restricted,
    Denied,
    Authorized,
}

impl MicrophoneAuthorization {
    #[cfg(target_os = "macos")]
    fn from_av_code(code: isize) -> Self {
        match code {
            0 => Self::NotDetermined,
            1 => Self::Restricted,
            2 => Self::Denied,
            3 => Self::Authorized,
            _ => Self::NotDetermined,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotDetermined => "not_determined",
            Self::Restricted => "restricted",
            Self::Denied => "denied",
            Self::Authorized => "authorized",
        }
    }

    #[must_use]
    pub fn is_authorized(self) -> bool {
        matches!(self, Self::Authorized)
    }
}

/// Current microphone capture authorization (AVFoundation / TCC).
pub fn microphone_authorization() -> MicrophoneAuthorization {
    #[cfg(target_os = "macos")]
    {
        let media = av_media_type_audio_string();
        if media.is_null() {
            return MicrophoneAuthorization::NotDetermined;
        }
        let code: isize = unsafe {
            msg_send![class!(AVCaptureDevice), authorizationStatusForMediaType: media]
        };
        MicrophoneAuthorization::from_av_code(code)
    }
    #[cfg(not(target_os = "macos"))]
    {
        MicrophoneAuthorization::Authorized
    }
}

/// Request microphone access (shows the system prompt when status is not determined).
pub fn request_microphone_access_prompt() {
    #[cfg(target_os = "macos")]
    {
        use block::ConcreteBlock;
        let media = av_media_type_audio_string();
        if media.is_null() {
            return;
        }
        let block = ConcreteBlock::new(move |_granted: bool| {}).copy();
        unsafe {
            let _: () = msg_send![
                class!(AVCaptureDevice),
                requestAccessForMediaType: media
                completionHandler: &*block
            ];
        }
    }
}

/// Whether this process is trusted for Accessibility (keyboard/paste).
#[must_use]
pub fn is_accessibility_trusted() -> bool {
    #[cfg(target_os = "macos")]
    unsafe {
        AXIsProcessTrusted()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Ask macOS for Accessibility trust (system prompt / guidance when allowed).
pub fn request_accessibility_prompt() {
    #[cfg(target_os = "macos")]
    {
        use core_foundation::base::TCFType;
        use core_foundation::boolean::CFBoolean;
        use core_foundation::dictionary::CFDictionary;
        use core_foundation::string::CFString;
        unsafe {
            let key = CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt as *mut _);
            let dict = CFDictionary::from_CFType_pairs(&[(key, CFBoolean::true_value())]);
            let _ = AXIsProcessTrustedWithOptions(dict.as_concrete_TypeRef());
        }
    }
}

/// Whether Input Monitoring allows global keyboard listening.
#[must_use]
pub fn is_input_monitoring_trusted() -> bool {
    #[cfg(target_os = "macos")]
    unsafe {
        CGPreflightListenEventAccess()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Request Input Monitoring permission (may show a system prompt).
pub fn request_input_monitoring_trust_prompt() {
    #[cfg(target_os = "macos")]
    unsafe {
        let _ = CGRequestListenEventAccess();
    }
}
