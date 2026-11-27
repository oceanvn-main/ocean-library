"""Write the Python side of the cross-language fixtures (read by MATLAB's CrossLanguageTest).

    cd Python && python -m ocean_replay.tests.make_fixtures

Writes into +OceanTest/+OceanReplay/fixtures:
  toy_python.h5               the Python Toy, full spec, float64
  toy_python_expected.json    every stored value: coords + shape + column-major data
  spec_python.json            a spec written by Python, with the masks Python selects
  ddpm_python_tiny.h5         the diffusers adapter's recording (tiny network), for Replay Studio
"""
import json
import os

import numpy as np

from ocean_replay import Recorder, Recording, Sampling, Spec
from ocean_replay.tests.test_cross_language import FX
from ocean_replay.tests.toy import Toy, full_spec


def main():
    path = os.path.join(FX, "toy_python.h5")
    Recorder(Toy(), full_spec("float64"), path, overwrite=True, run_id="toy-python",
             title="cross-language fixture (Python)").record()
    exp = {}
    with Recording(path) as r:
        for n in r.names():
            info = r.info(n)
            I, V = r.stored(n)
            pts = []
            for p in range(V.shape[0]):
                coords = {a: float(V[p, j]) for j, a in enumerate(info.axes)}
                v, _ = r.get(n, coords, "exact")
                if info.kind == "tokens":
                    pts.append({"coords": coords, "tokens": v})
                else:
                    a = np.asarray(v, np.float64).reshape(info.value_shape)
                    pts.append({"coords": coords, "shape": info.value_shape,
                                "data": a.reshape(-1, order="F").tolist()})      # MATLAB column-major
            exp[n] = pts
    with open(os.path.join(FX, "toy_python_expected.json"), "w", encoding="utf-8") as f:
        json.dump({"channels": exp}, f, ensure_ascii=False)

    S = Sampling
    spec = (Spec(precision="float64")
            .add_group("a", 0, ["x", "y"], {"step": S.union(S.every(20), S.window(51, 1))})
            .add_group("b", 1, "z", {"step": S.list_([900, 500])})
            .add_view("v", "occlusion", {"size": 4, "stride": 2}, {"step": S.every_index(250)}))
    axis = list(range(1000, 0, -1))
    masks = {"a": spec.groups[0].at["step"].select(axis), "b": spec.groups[1].at["step"].select(axis),
             "v": spec.views[0].at["step"].select(axis)}
    with open(os.path.join(FX, "spec_python.json"), "w", encoding="utf-8") as f:
        json.dump({"spec": spec.to_dict(), "axis": axis, "masks": masks}, f)
    # a diffusion recording made by the diffusers adapter (tiny net + diffusers' DDPM step),
    # opened by MATLAB's Replay Studio with the Diffusion layout (CrossLanguageTest)
    from ocean_replay.tests.test_diffusers_adapter import make, spec_all
    Recorder(make(), spec_all(), os.path.join(FX, "ddpm_python_tiny.h5"), overwrite=True,
             run_id="ddpm-python-tiny", title="tiny DDPM recorded in Python").record()
    print("Python fixtures written to", FX)


if __name__ == "__main__":
    main()
