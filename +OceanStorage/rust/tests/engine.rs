use ocean_storage::{Engine, MAX_KEY_SIZE, MAX_VALUE_SIZE};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
static SEQ: AtomicU64 = AtomicU64::new(0);
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "ocean-storage-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn randomized_multilevel_tree_reopen_and_cache() {
    let d = Dir::new();
    let mut e = Engine::open(&d.0).unwrap();
    let mut oracle = BTreeMap::new();
    let mut state = 19u64;
    for i in 0..1600u32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let key = ((state >> 32) % 650).to_be_bytes().to_vec();
        if i % 4 == 0 {
            e.delete(&key).unwrap();
            oracle.remove(&key);
        } else {
            let value = state.to_le_bytes().to_vec();
            e.put(&key, &value).unwrap();
            oracle.insert(key.clone(), value);
        }
        assert_eq!(e.get(&key).unwrap(), oracle.get(&key).cloned());
        if i % 113 == 0 {
            assert_eq!(
                e.scan(&[], None).unwrap(),
                oracle.clone().into_iter().collect::<Vec<_>>()
            );
        }
    }
    assert_eq!(
        e.scan(&[], None).unwrap(),
        oracle.clone().into_iter().collect::<Vec<_>>()
    );
    let text = e.inspect().unwrap();
    assert!(text.contains("cache_resident=32"));
    assert!(text.matches("Branch").count() > 8);
    drop(e);
    let mut e = Engine::open(&d.0).unwrap();
    assert_eq!(
        e.scan(&[], None).unwrap(),
        oracle.into_iter().collect::<Vec<_>>()
    );
}
#[test]
fn binary_empty_limits_ranges_and_lock() {
    let d = Dir::new();
    let mut e = Engine::open(&d.0).unwrap();
    assert!(Engine::open(&d.0).is_err());
    e.put(b"", b"").unwrap();
    e.put(&[0, 255], &[255, 0]).unwrap();
    e.put(b"z", b"v").unwrap();
    assert_eq!(e.get(b"").unwrap(), Some(vec![]));
    assert_eq!(e.get(b"missing").unwrap(), None);
    assert_eq!(
        e.scan(&[0, 255], Some(b"z")).unwrap(),
        vec![(vec![0, 255], vec![255, 0])]
    );
    assert!(e.scan(b"z", Some(b"a")).unwrap().is_empty());
    assert!(e.scan(b"z", Some(b"z")).unwrap().is_empty());
    assert!(e.put(&[1; MAX_KEY_SIZE + 1], b"").is_err());
    assert!(e.put(b"", &[1; MAX_VALUE_SIZE + 1]).is_err());
    assert_eq!(e.get(b"").unwrap(), Some(vec![]));
    e.delete(b"").unwrap();
    e.delete(b"").unwrap();
    assert_eq!(e.get(b"").unwrap(), None);
    e.put(&[1; MAX_KEY_SIZE], &[2; MAX_VALUE_SIZE]).unwrap();
    drop(e);
    assert!(Engine::open(&d.0).is_ok());
}
#[test]
fn corruption_and_partial_wal() {
    let d = Dir::new();
    {
        let mut e = Engine::open(&d.0).unwrap();
        e.put(b"a", b"b").unwrap();
    }
    fs::write(d.0.join("redo.wal"), b"partial").unwrap();
    assert!(Engine::open(&d.0).is_ok());
    let mut f = OpenOptions::new()
        .write(true)
        .open(d.0.join("pages.bin"))
        .unwrap();
    f.seek(SeekFrom::Start(4096 + 50)).unwrap();
    f.write_all(&[99]).unwrap();
    f.sync_all().unwrap();
    drop(f);
    assert!(Engine::open(&d.0)
        .err()
        .unwrap()
        .to_string()
        .contains("checksum"));
}

#[test]
fn read_io_error_is_fail_stop() {
    let d = Dir::new();
    let mut e = Engine::open(&d.0).unwrap();
    e.put(b"a", b"b").unwrap();
    // Simulate external storage truncation; the cache was invalidated on commit.
    OpenOptions::new()
        .write(true)
        .open(d.0.join("pages.bin"))
        .unwrap()
        .set_len(4096)
        .unwrap();
    assert!(e.get(b"a").is_err());
    assert!(e
        .inspect()
        .unwrap_err()
        .to_string()
        .contains("reopen required"));
    assert!(e
        .put(b"c", b"d")
        .unwrap_err()
        .to_string()
        .contains("reopen required"));
}

#[test]
fn real_process_crash_recovery() {
    // Cargo builds the example separately; invoke cargo to keep the child binary
    // platform-independent and test actual process exit (no unwinding/destructors).
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "--example", "crash_writer"])
        .status()
        .unwrap();
    assert!(status.success());
    let exe = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!("crash_writer{}", std::env::consts::EXE_SUFFIX));
    for point in ["wal_written", "wal_synced", "data_partial", "data_synced"] {
        let d = Dir::new();
        {
            let mut e = Engine::open(&d.0).unwrap();
            e.put(b"before", b"acknowledged").unwrap();
            // The eighth key forces a leaf split and root metadata replacement.
            for i in 0..6 {
                e.put(&[i], &[i, 255]).unwrap();
            }
        }
        let status = std::process::Command::new(&exe)
            .arg(&d.0)
            .env("OCEAN_STORAGE_CRASH_AT", point)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(91));
        if point == "wal_synced" {
            // A valid durable WAL must repair even an unreadable metadata page.
            let mut f = OpenOptions::new()
                .write(true)
                .open(d.0.join("pages.bin"))
                .unwrap();
            f.write_all(&[0; 4096]).unwrap();
            f.sync_all().unwrap();
        }
        if point == "wal_written" {
            let wal = fs::read(d.0.join("redo.wal")).unwrap();
            // Incomplete frame leaves original committed data intact.
            fs::write(d.0.join("redo.wal"), &wal[..wal.len() - 7]).unwrap();
            {
                let mut recovered = Engine::open(&d.0).unwrap();
                assert_eq!(recovered.get(b"crash-key").unwrap(), None);
            }
            // Complete checksum corruption is an error, never discarded.
            let mut bad = wal.clone();
            bad[40] ^= 1;
            fs::write(d.0.join("redo.wal"), bad).unwrap();
            assert!(Engine::open(&d.0).is_err());
            fs::write(d.0.join("redo.wal"), wal).unwrap();
        }
        let mut e = Engine::open(&d.0).unwrap();
        assert_eq!(e.get(b"before").unwrap(), Some(b"acknowledged".to_vec()));
        for i in 0..6 {
            assert_eq!(e.get(&[i]).unwrap(), Some(vec![i, 255]));
        }
        assert_eq!(
            e.get(b"crash-key").unwrap(),
            Some(b"durable-value".to_vec())
        );
        drop(e);
        let mut e = Engine::open(&d.0).unwrap();
        assert_eq!(
            e.get(b"crash-key").unwrap(),
            Some(b"durable-value".to_vec())
        );
    }
}
