"""G0 spike, part 2: read the MATLAB file in h5py; write a Python file for MATLAB to read."""
import json, os, tempfile
import numpy as np
import h5py

tmp = tempfile.gettempdir()
one = lambda a: a.item() if isinstance(a, np.ndarray) and a.size == 1 else a   # MATLAB writes scalars as shape (1,)
H, W, C = 2, 3, 4
logical = lambda p: (np.arange(1, H * W * C + 1).reshape((H, W, C), order="F") + 100 * p).astype(np.float32)

with h5py.File(os.path.join(tmp, "g0_matlab.h5"), "r") as f:
    d = f["/channels/x/data"]
    print("disk shape (h5py):", d.shape, "maxshape:", d.maxshape, "chunks:", d.chunks)
    v = d[1]                               # point index 1 (= MATLAB p = 2), shape (C, W, H)
    print("reversed-value rule holds:", np.array_equal(np.transpose(v), logical(2)))
    print("points:", f["/channels/x/points"][()], f["/channels/x/points"].shape)
    a = f["/channels/x"].attrs
    print("attrs:", {k: a[k] for k in a})
    print("root:", {k: f.attrs[k] for k in f.attrs if k != "meta"})
    print("meta json:", json.loads(one(f.attrs["meta"])))
    t = f["/channels/tok/data"]
    print("tokens:", t.shape, [s.decode() if isinstance(s, bytes) else s for s in t[0]])
    print("axis:", f["/axes/step"][()], f["/axes/step"].dtype)

# Python writer following the spec: disk shape [P, *reversed(valueShape)]
out = os.path.join(tmp, "g0_python.h5")
with h5py.File(out, "w") as f:
    f.attrs["format"] = "ocean-replay"; f.attrs["schemaVersion"] = "1.0"
    d = f.create_dataset("/channels/x/data", shape=(0, C, W, H), maxshape=(None, C, W, H),
                         chunks=(1, C, W, H), dtype="f4")
    for p in range(1, 4):
        d.resize(p, axis=0); d[p - 1] = np.transpose(logical(p))
    f.create_dataset("/channels/x/points", data=np.array([[0], [20], [40]], dtype="i4"))
    g = f["/channels/x"]
    g.attrs["kind"] = "image"; g.attrs["axes"] = ["step"]; g.attrs["valueShape"] = np.array([H, W, C], "i4")
    g.attrs["inTexts"] = np.uint8(1)
    f.attrs["meta"] = json.dumps({"title": "spike ñ é 中"}, ensure_ascii=False)
    s = f.create_dataset("/channels/tok/data", shape=(1, 5), maxshape=(None, 5), dtype=h5py.string_dtype("utf-8"))
    s[0] = ["the", "cat", "ñé", "中文", "."]
    f.create_dataset("/axes/step", data=np.array([0.0, 20.0, 40.0]))
print("wrote", out)
