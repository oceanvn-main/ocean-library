"""Protocol-v1 process client. Failed requests are never retried automatically."""
import subprocess


class OceanStorage:
    def __init__(self, executable, directory):
        self._process = subprocess.Popen([str(executable), "serve", str(directory)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, encoding="ascii")
        if self._process.stdout.readline().rstrip("\r\n") != "READY\t1":
            error = self._process.stderr.read()
            self._process.wait()
            for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
                pipe.close()
            raise RuntimeError("Engine did not start: " + error)

    @staticmethod
    def _hex(value):
        if not isinstance(value, bytes):
            raise TypeError("keys and values must be bytes")
        return value.hex()

    def _read(self):
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("Engine connection ended; last write may have committed")
        fields = line.rstrip("\r\n").split("\t")
        if fields[0] == "ERR":
            raise RuntimeError(bytes.fromhex(fields[1]).decode("utf-8"))
        return fields

    def _request(self, *fields):
        if self._process.poll() is not None:
            raise RuntimeError("Engine is closed")
        self._process.stdin.write("\t".join(fields) + "\n")
        self._process.stdin.flush()
        return self._read()

    def put(self, key, value):
        if self._request("PUT", self._hex(key), self._hex(value)) != ["OK"]:
            raise RuntimeError("Unexpected response")

    def get(self, key):
        response = self._request("GET", self._hex(key))
        if response == ["MISSING"]:
            return None
        if response[0] != "VALUE":
            raise RuntimeError("Unexpected response")
        return bytes.fromhex(response[1])

    def remove(self, key):
        if self._request("DELETE", self._hex(key)) != ["OK"]:
            raise RuntimeError("Unexpected response")

    def scan(self, start=b"", end=None):
        row = self._request("SCAN", self._hex(start), "*" if end is None else self._hex(end))
        rows = []
        while row != ["END"]:
            if len(row) != 3 or row[0] != "ROW":
                raise RuntimeError("Unexpected scan response")
            rows.append((bytes.fromhex(row[1]), bytes.fromhex(row[2])))
            row = self._read()
        return rows

    def inspect(self):
        response = self._request("INSPECT")
        if response[0] != "TEXT":
            raise RuntimeError("Unexpected response")
        return bytes.fromhex(response[1]).decode("utf-8")

    def close(self):
        if self._process.poll() is None:
            try:
                self._request("CLOSE")
                self._process.wait(timeout=5)
            finally:
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait()
        for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
