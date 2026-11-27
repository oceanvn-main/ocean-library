"""ocean_replay -- record once, replay in MATLAB (Python mirror of +OceanReplay).

The file format and the spec grammar are defined in
Planning/System/Replay/Replay_Format_Spec.md; this package and the MATLAB package
implement exactly the same contract, so a recording made here (e.g. on a cloud GPU)
opens in MATLAB's Replay Studio, and vice versa.

    from ocean_replay import Spec, Sampling, Recorder, Recording
    spec = Spec.from_json(open("spec.json").read())
    Recorder(my_recordable, spec, "run.h5").record()
    rec = Recording("run.h5"); value, at = rec.get("x", {"step": 500})

Only numpy + h5py are required; torch is used only by the torch adapters/views.
"""
from .info import Axis, ChannelInfo, ViewInfo, KINDS
from .sampling import Sampling
from .spec import Spec
from .writer import Hdf5Writer
from .reader import Recording
from .schema import validate
from .recorder import Recorder, NullRecorder, IRecordable, verify_invariance
from .extract import extract

__version__ = "1.0.0"
SCHEMA_VERSION = "1.0"
