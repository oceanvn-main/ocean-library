//! A deliberately small, inspectable B+ tree with full-page redo recovery.
//! Deletes retain tombstones; pages are never reclaimed in this first milestone.
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const MAX_KEY_SIZE: usize = 128;
pub const MAX_VALUE_SIZE: usize = 256;
const PAGE: usize = 4096;
const MAX_KEYS: usize = 7;
const CACHE: usize = 32;
const MAGIC: &[u8; 8] = b"OCNPG001";
const WAL_MAGIC: &[u8; 8] = b"OCNWAL01";
pub type Result<T> = std::result::Result<T, Error>;
pub type Records = Vec<(Vec<u8>, Vec<u8>)>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Corrupt(String),
    Invalid(String),
    Locked,
    Failed,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "storage I/O: {e}"),
            Self::Corrupt(s) => write!(f, "corrupt storage: {s}"),
            Self::Invalid(s) => write!(f, "invalid input: {s}"),
            Self::Locked => write!(f, "database already has an owner"),
            Self::Failed => write!(f, "engine failed; reopen required"),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Clone, Debug)]
enum Node {
    Leaf {
        keys: Vec<Vec<u8>>,
        values: Vec<Option<Vec<u8>>>,
        next: u64,
    },
    Branch {
        keys: Vec<Vec<u8>>,
        children: Vec<u64>,
    },
}
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}
fn u64_at(b: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        b.get(at..at + 8)
            .ok_or_else(|| Error::Corrupt("truncated integer".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn seal(mut b: Vec<u8>) -> Result<Vec<u8>> {
    if b.len() > PAGE - 8 {
        return Err(Error::Corrupt("page overflow".into()));
    }
    b.resize(PAGE - 8, 0);
    let h = checksum(&b);
    b.extend(h.to_le_bytes());
    Ok(b)
}
fn verify(b: &[u8]) -> Result<()> {
    if b.len() != PAGE || checksum(&b[..PAGE - 8]) != u64_at(b, PAGE - 8)? {
        return Err(Error::Corrupt("page checksum".into()));
    }
    Ok(())
}
fn metadata(root: u64, next: u64) -> Result<Vec<u8>> {
    let mut b = MAGIC.to_vec();
    b.extend(root.to_le_bytes());
    b.extend(next.to_le_bytes());
    seal(b)
}
fn encode(n: &Node) -> Result<Vec<u8>> {
    let mut b = MAGIC.to_vec();
    let (kind, keys) = match n {
        Node::Leaf { keys, .. } => (1, keys),
        Node::Branch { keys, .. } => (2, keys),
    };
    b.push(kind);
    b.push(keys.len() as u8);
    match n {
        Node::Leaf { keys, values, next } => {
            b.extend(next.to_le_bytes());
            for (k, v) in keys.iter().zip(values) {
                b.extend((k.len() as u16).to_le_bytes());
                b.extend(
                    v.as_ref()
                        .map_or(u16::MAX, |x| x.len() as u16)
                        .to_le_bytes(),
                );
                b.extend(k);
                if let Some(v) = v {
                    b.extend(v);
                }
            }
        }
        Node::Branch { keys, children } => {
            for c in children {
                b.extend(c.to_le_bytes());
            }
            for k in keys {
                b.extend((k.len() as u16).to_le_bytes());
                b.extend(k);
            }
        }
    }
    seal(b)
}
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| Error::Corrupt("length overflow".into()))?;
        if end > PAGE - 8 {
            return Err(Error::Corrupt("page record bounds".into()));
        }
        let r = &self.b[self.at..end];
        self.at = end;
        Ok(r)
    }
    fn short(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn long(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn key(&mut self) -> Result<Vec<u8>> {
        let n = self.short()? as usize;
        if n > MAX_KEY_SIZE {
            return Err(Error::Corrupt("key length".into()));
        }
        Ok(self.take(n)?.to_vec())
    }
}
fn decode(b: &[u8]) -> Result<Node> {
    verify(b)?;
    if &b[..8] != MAGIC {
        return Err(Error::Corrupt("node format/version".into()));
    }
    let count = b[9] as usize;
    if count > MAX_KEYS {
        return Err(Error::Corrupt("node key count".into()));
    }
    let mut c = Cursor { b, at: 10 };
    let mut keys = Vec::new();
    let node = match b[8] {
        1 => {
            let next = c.long()?;
            let mut values = Vec::new();
            for _ in 0..count {
                let kn = c.short()? as usize;
                let vn = c.short()?;
                if kn > MAX_KEY_SIZE || (vn != u16::MAX && vn as usize > MAX_VALUE_SIZE) {
                    return Err(Error::Corrupt("record length".into()));
                }
                keys.push(c.take(kn)?.to_vec());
                values.push(if vn == u16::MAX {
                    None
                } else {
                    Some(c.take(vn as usize)?.to_vec())
                });
            }
            Node::Leaf {
                keys: keys.clone(),
                values,
                next,
            }
        }
        2 => {
            if count == 0 {
                return Err(Error::Corrupt("empty branch".into()));
            }
            let mut children = Vec::new();
            for _ in 0..=count {
                children.push(c.long()?);
            }
            for _ in 0..count {
                keys.push(c.key()?);
            }
            Node::Branch {
                keys: keys.clone(),
                children,
            }
        }
        _ => return Err(Error::Corrupt("node kind".into())),
    };
    if keys.windows(2).any(|p| p[0] >= p[1]) {
        return Err(Error::Corrupt("unsorted keys".into()));
    }
    Ok(node)
}

/// One process owns the directory. Clean cache is FIFO bounded to 32 pages.
pub struct Engine {
    _lock: File,
    data: File,
    wal: File,
    root: u64,
    next: u64,
    failed: bool,
    cache: BTreeMap<u64, Node>,
    fifo: VecDeque<u64>,
    dirty: BTreeMap<u64, Node>,
    reads: u64,
    hits: u64,
    writes: u64,
}
impl Engine {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(path.as_ref())?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.as_ref().join("owner.lock"))?;
        lock.try_lock().map_err(|_| Error::Locked)?;
        let data = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.as_ref().join("pages.bin"))?;
        let wal = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.as_ref().join("redo.wal"))?;
        let mut e = Self {
            _lock: lock,
            data,
            wal,
            root: 1,
            next: 2,
            failed: false,
            cache: BTreeMap::new(),
            fifo: VecDeque::new(),
            dirty: BTreeMap::new(),
            reads: 0,
            hits: 0,
            writes: 0,
        };
        e.recover()?;
        if e.data.metadata()?.len() == 0 {
            e.dirty.insert(
                1,
                Node::Leaf {
                    keys: vec![],
                    values: vec![],
                    next: 0,
                },
            );
            e.commit()?;
        }
        let b = e.read_raw(0)?;
        verify(&b)?;
        if &b[..8] != MAGIC {
            return Err(Error::Corrupt("metadata format/version".into()));
        }
        e.root = u64_at(&b, 8)?;
        e.next = u64_at(&b, 16)?;
        if e.next < 2
            || e.root == 0
            || e.root >= e.next
            || e.data.metadata()?.len()
                != e.next
                    .checked_mul(PAGE as u64)
                    .ok_or_else(|| Error::Corrupt("page count overflow".into()))?
        {
            return Err(Error::Corrupt("metadata page range/file size".into()));
        }
        // Validate every allocated page on open, after redo has repaired torn pages.
        for id in 1..e.next {
            let n = decode(&e.read_raw(id)?)?;
            e.check_refs(&n)?;
        }
        Ok(e)
    }
    fn ready(&self) -> Result<()> {
        if self.failed {
            Err(Error::Failed)
        } else {
            Ok(())
        }
    }
    fn check_refs(&self, n: &Node) -> Result<()> {
        let valid = |id: &u64| *id > 0 && *id < self.next;
        let ok = match n {
            Node::Leaf { next, .. } => *next == 0 || valid(next),
            Node::Branch { children, .. } => children.iter().all(valid),
        };
        if !ok {
            return Err(Error::Corrupt("page reference".into()));
        }
        Ok(())
    }
    fn read_raw(&mut self, id: u64) -> Result<Vec<u8>> {
        let mut b = vec![0; PAGE];
        self.data.seek(SeekFrom::Start(id * PAGE as u64))?;
        self.data.read_exact(&mut b)?;
        self.reads += 1;
        Ok(b)
    }
    fn node(&mut self, id: u64) -> Result<Node> {
        if id == 0 || id >= self.next {
            return Err(Error::Corrupt("invalid page ID".into()));
        }
        if let Some(n) = self.dirty.get(&id) {
            return Ok(n.clone());
        }
        if let Some(n) = self.cache.get(&id) {
            self.hits += 1;
            return Ok(n.clone());
        }
        let n = decode(&self.read_raw(id)?)?;
        self.check_refs(&n)?;
        if self.cache.len() == CACHE {
            let old = self.fifo.pop_front().unwrap();
            self.cache.remove(&old);
        }
        self.cache.insert(id, n.clone());
        self.fifo.push_back(id);
        Ok(n)
    }
    fn alloc(&mut self, n: Node) -> u64 {
        let id = self.next;
        self.next += 1;
        self.dirty.insert(id, n);
        id
    }
    fn insert(
        &mut self,
        id: u64,
        key: &[u8],
        value: Option<Vec<u8>>,
        depth: usize,
    ) -> Result<Option<(Vec<u8>, u64)>> {
        if depth > 64 {
            return Err(Error::Corrupt("tree depth/cycle".into()));
        }
        match self.node(id)? {
            Node::Leaf {
                mut keys,
                mut values,
                next,
            } => {
                match keys.binary_search_by(|k| k.as_slice().cmp(key)) {
                    Ok(i) => values[i] = value,
                    Err(i) => {
                        keys.insert(i, key.to_vec());
                        values.insert(i, value);
                    }
                }
                if keys.len() <= MAX_KEYS {
                    self.dirty.insert(id, Node::Leaf { keys, values, next });
                    return Ok(None);
                }
                let right_keys = keys.split_off(keys.len() / 2);
                let right_values = values.split_off(keys.len());
                let separator = right_keys[0].clone();
                let right = self.alloc(Node::Leaf {
                    keys: right_keys,
                    values: right_values,
                    next,
                });
                self.dirty.insert(
                    id,
                    Node::Leaf {
                        keys,
                        values,
                        next: right,
                    },
                );
                Ok(Some((separator, right)))
            }
            Node::Branch {
                mut keys,
                mut children,
            } => {
                let i = keys.partition_point(|k| k.as_slice() <= key);
                if let Some((separator, right)) = self.insert(children[i], key, value, depth + 1)? {
                    keys.insert(i, separator);
                    children.insert(i + 1, right);
                }
                if keys.len() <= MAX_KEYS {
                    self.dirty.insert(id, Node::Branch { keys, children });
                    return Ok(None);
                }
                let mid = keys.len() / 2;
                let right_keys = keys.split_off(mid + 1);
                let separator = keys.pop().unwrap();
                let right_children = children.split_off(mid + 1);
                let right = self.alloc(Node::Branch {
                    keys: right_keys,
                    children: right_children,
                });
                self.dirty.insert(id, Node::Branch { keys, children });
                Ok(Some((separator, right)))
            }
        }
    }
    fn finish<T>(&mut self, r: Result<T>) -> Result<T> {
        if r.is_err() {
            self.failed = true;
        }
        r
    }
    fn mutate(&mut self, key: &[u8], value: Option<Vec<u8>>) -> Result<()> {
        self.ready()?;
        let result = (|| {
            if let Some((separator, right)) = self.insert(self.root, key, value, 0)? {
                self.root = self.alloc(Node::Branch {
                    keys: vec![separator],
                    children: vec![self.root, right],
                });
            }
            self.commit()
        })();
        self.finish(result)
    }
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.ready()?;
        if key.len() > MAX_KEY_SIZE || value.len() > MAX_VALUE_SIZE {
            return Err(Error::Invalid("key/value exceeds size limit".into()));
        }
        self.mutate(key, Some(value.to_vec()))
    }
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.ready()?;
        if key.len() > MAX_KEY_SIZE {
            return Err(Error::Invalid("key exceeds size limit".into()));
        }
        if self.get(key)?.is_none() {
            return Ok(());
        }
        self.mutate(key, None)
    }
    fn leaf(&mut self, key: &[u8]) -> Result<(u64, Node)> {
        let mut id = self.root;
        for _ in 0..64 {
            match self.node(id)? {
                Node::Branch { keys, children } => {
                    id = children[keys.partition_point(|k| k.as_slice() <= key)]
                }
                n => return Ok((id, n)),
            }
        }
        Err(Error::Corrupt("tree depth/cycle".into()))
    }
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.ready()?;
        if key.len() > MAX_KEY_SIZE {
            return Err(Error::Invalid("key exceeds size limit".into()));
        }
        let result = (|| {
            if let (_, Node::Leaf { keys, values, .. }) = self.leaf(key)? {
                Ok(keys
                    .binary_search_by(|k| k.as_slice().cmp(key))
                    .ok()
                    .and_then(|i| values[i].clone()))
            } else {
                unreachable!()
            }
        })();
        self.finish(result)
    }
    pub fn scan(&mut self, start: &[u8], end: Option<&[u8]>) -> Result<Records> {
        self.ready()?;
        if start.len() > MAX_KEY_SIZE || end.is_some_and(|e| e.len() > MAX_KEY_SIZE) {
            return Err(Error::Invalid("key exceeds size limit".into()));
        }
        if end.is_some_and(|e| e <= start) {
            return Ok(vec![]);
        }
        let result = (|| {
            let (_, mut node) = self.leaf(start)?;
            let mut out = Vec::new();
            let mut visited = 0;
            loop {
                visited += 1;
                if visited >= self.next {
                    return Err(Error::Corrupt("leaf cycle".into()));
                }
                if let Node::Leaf { keys, values, next } = node {
                    for (k, v) in keys.into_iter().zip(values) {
                        if end.is_some_and(|e| k.as_slice() >= e) {
                            return Ok(out);
                        }
                        if k.as_slice() >= start {
                            if let Some(v) = v {
                                out.push((k, v));
                            }
                        }
                    }
                    if next == 0 {
                        return Ok(out);
                    }
                    node = self.node(next)?;
                } else {
                    return Err(Error::Corrupt("leaf link targets branch".into()));
                }
            }
        })();
        self.finish(result)
    }
    pub fn inspect(&mut self) -> Result<String> {
        self.ready()?;
        let result = (|| {
            let mut s = format!("OceanStorage v1 root={} allocated_pages={} page_size={} max_keys={} cache_capacity={} tombstone_deletion=true\n",self.root,self.next,PAGE,MAX_KEYS,CACHE);
            for id in 1..self.next {
                s.push_str(&format!("page {id}: {:?}\n", self.node(id)?));
            }
            s.push_str(&format!(
                "reads={} cache_hits={} page_writes={} cache_resident={}\n",
                self.reads,
                self.hits,
                self.writes,
                self.cache.len()
            ));
            Ok(s)
        })();
        self.finish(result)
    }
    fn commit(&mut self) -> Result<()> {
        let mut images = vec![(0, metadata(self.root, self.next)?)];
        for (id, n) in &self.dirty {
            images.push((*id, encode(n)?));
        }
        let mut frame = WAL_MAGIC.to_vec();
        frame.extend((images.len() as u64).to_le_bytes());
        for (id, b) in &images {
            frame.extend(id.to_le_bytes());
            frame.extend(b);
        }
        let h = checksum(&frame);
        frame.extend(h.to_le_bytes());
        frame.extend(b"COMMIT01");
        self.wal.seek(SeekFrom::Start(0))?;
        self.wal.write_all(&frame)?;
        self.wal.set_len(frame.len() as u64)?;
        crash("wal_written");
        self.wal.sync_all()?;
        crash("wal_synced");
        for (i, (id, b)) in images.iter().enumerate() {
            self.data.seek(SeekFrom::Start(id * PAGE as u64))?;
            self.data.write_all(b)?;
            self.writes += 1;
            if i == 0 {
                crash("data_partial");
            }
        }
        self.data.set_len(self.next * PAGE as u64)?;
        self.data.sync_all()?;
        crash("data_synced");
        self.wal.set_len(0)?;
        self.wal.sync_all()?;
        self.dirty.clear();
        self.cache.clear();
        self.fifo.clear();
        Ok(())
    }
    fn recover(&mut self) -> Result<()> {
        let len = self.wal.metadata()?.len();
        if len == 0 {
            return Ok(());
        }
        // A transaction touches at most one path (depth <=64), its split siblings,
        // root and metadata. Reject unbounded corrupt lengths before allocating.
        if len > 256 * (PAGE as u64 + 8) + 32 {
            return Err(Error::Corrupt("WAL too large".into()));
        }
        let mut b = Vec::new();
        self.wal.read_to_end(&mut b)?;
        if b.len() < 16 {
            self.wal.set_len(0)?;
            self.wal.sync_all()?;
            return Ok(());
        }
        if &b[..8] != WAL_MAGIC {
            return Err(Error::Corrupt("WAL format/version".into()));
        }
        let count = u64_at(&b, 8)?;
        if count == 0 || count > 256 {
            return Err(Error::Corrupt("WAL image count".into()));
        }
        let expected = 16 + count as usize * (PAGE + 8) + 16;
        if b.len() < expected {
            self.wal.set_len(0)?;
            self.wal.sync_all()?;
            return Ok(());
        }
        if b.len() != expected
            || &b[expected - 8..] != b"COMMIT01"
            || checksum(&b[..expected - 16]) != u64_at(&b, expected - 16)?
        {
            return Err(Error::Corrupt("committed WAL checksum/length".into()));
        }
        let mut images = Vec::new();
        let mut ids = std::collections::BTreeSet::new();
        for i in 0..count as usize {
            let at = 16 + i * (PAGE + 8);
            let id = u64_at(&b, at)?;
            let image = &b[at + 8..at + 8 + PAGE];
            verify(image)?;
            if !ids.insert(id) {
                return Err(Error::Corrupt("duplicate WAL page".into()));
            }
            images.push((id, image));
        }
        let meta = images
            .iter()
            .find(|(id, _)| *id == 0)
            .ok_or_else(|| Error::Corrupt("WAL metadata absent".into()))?
            .1;
        if &meta[..8] != MAGIC {
            return Err(Error::Corrupt("WAL metadata format".into()));
        }
        let next = u64_at(meta, 16)?;
        let root = u64_at(meta, 8)?;
        if next < 2
            || root == 0
            || root >= next
            || next > u64::MAX / PAGE as u64
            || images.iter().any(|(id, _)| *id >= next)
        {
            return Err(Error::Corrupt("WAL page range".into()));
        }
        for (id, image) in images {
            self.data.seek(SeekFrom::Start(id * PAGE as u64))?;
            self.data.write_all(image)?;
        }
        self.data.set_len(next * PAGE as u64)?;
        self.data.sync_all()?;
        self.wal.set_len(0)?;
        self.wal.sync_all()?;
        Ok(())
    }
}
// Explicit opt-in fault injection is exercised in separate OS processes.
fn crash(point: &str) {
    if std::env::var("OCEAN_STORAGE_CRASH_AT").ok().as_deref() == Some(point) {
        std::process::exit(91);
    }
}
