//! Single-instance lock (matches Python `instance.lock` behavior).

use std::fs::OpenOptions;
use std::io::Write;

use fs2::FileExt;

use crate::settings_store::{app_config_dir, instance_lock_path};

/// Guard holding the exclusive instance lock for the process lifetime.
pub struct InstanceGuard {
    _file: std::fs::File,
}

impl InstanceGuard {
    fn new(file: std::fs::File) -> Option<Self> {
        file.try_lock_exclusive().ok()?;
        Some(Self { _file: file })
    }
}

/// Return `Some(guard)` if this process is the sole instance, or `None` if another holds the lock.
#[must_use]
pub fn try_acquire_single_instance() -> Option<InstanceGuard> {
    let dir = app_config_dir();
    std::fs::create_dir_all(&dir).ok()?;
    let path = instance_lock_path();
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&path)
        .ok()?;
    file.write_all(b"lock").ok()?;
    InstanceGuard::new(file)
}

/// Test helper using a custom lock path.
#[must_use]
pub fn try_acquire_lock_path(path: &std::path::Path) -> Option<InstanceGuard> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)
        .ok()?;
    file.write_all(b"lock").ok()?;
    InstanceGuard::new(file)
}
