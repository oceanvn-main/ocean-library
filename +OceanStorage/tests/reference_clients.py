"""Execute against real Rust engine and SQLite using repository text records."""
import pathlib
import sqlite3
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "+OceanStorage/clients/python"))
from ocean_storage import OceanStorage


def run(executable):
    sources = [ROOT / "AGENTS.md", ROOT / ".agents/architecture.md", ROOT / "+OceanStorage/CONTRACT.md"]
    records = []
    for source in sources:
        for index, line in enumerate(source.read_bytes().splitlines()):
            key = (source.name + ":" + str(index)).encode()
            records.append((key, line[:256]))
    with tempfile.TemporaryDirectory(prefix="reference-", dir=ROOT / "+OceanStorage/tests") as temporary:
        directory = pathlib.Path(temporary) / "database"
        sql = sqlite3.connect(":memory:")
        sql.execute("CREATE TABLE records(key BLOB PRIMARY KEY, value BLOB NOT NULL)")
        with OceanStorage(executable, directory) as store:
            contender = subprocess.run([str(executable), "serve", str(directory)], input="", text=True, capture_output=True, timeout=10)
            assert contender.returncode != 0 and not contender.stdout
            # Malformed lines must produce ERR, preserve synchronization and never mutate.
            for fields in [("PUT", "0", "00"), ("PUT", "gg", "00"), ("DELETE", "00", "extra"), ("PUT", "00", "00", "extra"), ("X" * 5000,)]:
                try:
                    store._request(*fields)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("malformed protocol accepted")
            assert store.scan() == []
            store.put(bytes(range(128)), bytes(range(256)))
            assert store.get(bytes(range(128))) == bytes(range(256))
            store.remove(bytes(range(128)))
            for key, value in reversed(records):
                store.put(key, value)
                sql.execute("INSERT OR REPLACE INTO records VALUES (?,?)", (key, value))
            store.put(b"", b"")
            sql.execute("INSERT INTO records VALUES (?,?)", (b"", b""))
            assert store.get(b"") == b""
            assert store.get(b"\xffmissing") is None
            try:
                store.put(bytes(129), b"rejected")
            except RuntimeError:
                pass
            else:
                raise AssertionError("oversize key accepted")
            for key, value in records[::7]:
                replacement = value[::-1]
                store.put(key, replacement)
                sql.execute("UPDATE records SET value=? WHERE key=?", (replacement, key))
            for key, _ in records[::11]:
                store.remove(key)
                sql.execute("DELETE FROM records WHERE key=?", (key,))
            expected = sql.execute("SELECT key,value FROM records ORDER BY key").fetchall()
            assert store.scan() == expected
            start, end = expected[len(expected)//4][0], expected[3*len(expected)//4][0]
            assert store.scan(start,end) == sql.execute("SELECT key,value FROM records WHERE key>=? AND key<? ORDER BY key", (start,end)).fetchall()
            assert store.scan(end,start) == []
            assert store.inspect()
        with OceanStorage(executable,directory) as store:
            assert store.scan() == expected
            # Repository-derived shared record is read by Java as well.
            shared_key, shared_value = expected[len(expected)//2]
        subprocess.run(["java", "-cp", str(ROOT / "+OceanStorage/tests/java-build"), "ClientSmoke", str(executable), str(directory), shared_key.hex(), shared_value.hex()], check=True)
        print(f"PASS Python/SQLite: {len(records)} repository records, overwrite/delete/range/reopen; Java shared record")


if __name__ == "__main__":
    run(pathlib.Path(sys.argv[1]).resolve())
