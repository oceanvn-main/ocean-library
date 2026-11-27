# OceanStorage recipes

See [the engine README](../../+OceanStorage/README.md),
[contract](../../+OceanStorage/CONTRACT.md) and
[verification log](../../+OceanStorage/BUILD_LOG.md).
Run examples from the repository root after building the Rust executable.
Each example owns its database while its client is open. Close it before another
language opens that same database.

## Rust

Build and test:

```powershell
cargo build --manifest-path '+OceanStorage/rust/Cargo.toml'
cargo test --manifest-path '+OceanStorage/rust/Cargo.toml'
```

The `ocean_storage` library exposes `Engine::open`, `put`, `get`, `delete`,
`scan` and `inspect`. Keys and values are byte slices. Scans use a start-inclusive,
end-exclusive range; `None` means no upper bound.

## Python

```python
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / '+OceanStorage' / 'clients' / 'python'))
from ocean_storage import OceanStorage

exe = root / '+OceanStorage' / 'rust' / 'target' / 'debug' / 'ocean-storage.exe'
with OceanStorage(exe, root / '+OceanStorage' / 'test-artifacts' / 'recipe-db') as db:
    db.put(b'dataset/version', b'v1')
    assert db.get(b'dataset/version') == b'v1'
    print(db.inspect())
```

On Linux/macOS use the executable name `ocean-storage` without `.exe`.

## MATLAB

```matlab
root = pwd;
exe = fullfile(root, '+OceanStorage', 'rust', 'target', 'debug', 'ocean-storage.exe');
directory = fullfile(root, '+OceanStorage', 'test-artifacts', 'recipe-db');
db = OceanStorage.Client(exe, directory);
cleanup = onCleanup(@() db.close());
db.put(uint8('dataset/version'), uint8('v1'));
[value, found] = db.get(uint8('dataset/version'));
assert(found && isequal(value, uint8('v1')));
disp(db.inspect());
db.close();
```

Use `unicode2native(text, 'UTF-8')` rather than `uint8(text)` for non-ASCII text.
`remove(key)` deletes a record; MATLAB `delete(db)` disposes of the client.

## Java

Compile the client using `javac` and put its directory on your classpath. Use
`OceanStorageClient` in a try-with-resources block, passing the executable and
database directory to its constructor. Methods accept byte arrays. The tests
directory contains the executable Java smoke example.

## Protocol without a language wrapper

```powershell
& './+OceanStorage/rust/target/debug/ocean-storage.exe' serve './+OceanStorage/test-artifacts/manual'
```

The process prints `READY` and protocol version `1`, separated by a tab. Send
tab-separated requests, for example `PUT`, `61`, `62` to store byte `b` under key
`a`; then `GET`, `61`. `INSPECT` returns a hex-encoded UTF-8 tree/page report.
The complete protocol, including empty byte fields and range bounds, is frozen
in the contract.
